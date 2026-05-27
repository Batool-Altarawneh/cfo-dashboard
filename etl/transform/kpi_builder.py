"""

This file is part of the Transform Layer in the ETL pipeline.

It receives a cleaned monthly summary DataFrame from cleaner.py and adds financial KPI columns that will be used later by:
1. the production database
2. Power BI dashboard
3. Streamlit / ML dashboard

"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# KPI calculation
# ---------------------------------------------------------------------------

def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate financial KPIs from the cleaned monthly summary data.

    Expected input grain:
        One row per year, month, department, and region.

  
    """

    # Work on a copy so the original DataFrame is not changed by accident.
    df = df.copy()

    print("Calculating financial KPIs...")

    # -----------------------------------------------------------------------
    # Step 1: Make sure calculation columns have the right data types
    # -----------------------------------------------------------------------
    # The cleaner should already handle most type issues, but I still convert here to make this function safer if it is used independently.
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    df["total_expense"] = pd.to_numeric(df["total_expense"], errors="coerce")
    df["total_budget"] = pd.to_numeric(df["total_budget"], errors="coerce")

    # Drop rows that cannot be used for KPI calculations.
    # Without year/month/department/region, we cannot calculate time-based KPIs.
    critical_columns = [
        "year",
        "month",
        "department",
        "region",
        "total_revenue",
        "total_expense",
        "total_budget",
    ]

    rows_before = len(df)
    df = df.dropna(subset=critical_columns)
    dropped_rows = rows_before - len(df)

    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows:,} rows before KPI calculation due to missing values")

    # Convert year and month to normal int after nulls are removed.
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # -----------------------------------------------------------------------
    # Step 2: Sort data in the correct time order
    # -----------------------------------------------------------------------
    # Time-based KPIs like YTD, MoM, rolling averages, and YoY depend on row order.
    # Sorting prevents calculations from comparing months in the wrong sequence.
    df = df.sort_values(["department", "region", "year", "month"])
    df = df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # KPI 1: Revenue YTD
    # -----------------------------------------------------------------------
    # YTD means Year-To-Date.
    # It is the cumulative revenue from January to the current month.
    #
    # Example:
    # Jan = 10,000
    # Feb = 15,000
    # Revenue YTD in Feb = 25,000
    #
    # I group by department, region, and year so the running total resets for every department/region and every new year.
    df["revenue_ytd"] = (
        df.groupby(["department", "region", "year"])["total_revenue"]
        .cumsum()
    )

    # -----------------------------------------------------------------------
    # KPI 2: Expense YTD
    # -----------------------------------------------------------------------
    # Same idea as revenue_ytd, but for total expenses.
    # This helps the CFO see how much each department has spent so far in the current year.
    df["expense_ytd"] = (
        df.groupby(["department", "region", "year"])["total_expense"]
        .cumsum()
    )

    # -----------------------------------------------------------------------
    # KPI 3: Profit Margin %
    # -----------------------------------------------------------------------
    # Profit margin shows what percentage of revenue is left after expenses.
    #
    # Formula:
    #   profit_margin_pct = (revenue - expense) / revenue * 100
    #
    # I use np.where to avoid division by zero when revenue is 0.
    df["profit_margin_pct"] = np.where(
        df["total_revenue"] > 0,
        ((df["total_revenue"] - df["total_expense"]) / df["total_revenue"]) * 100,
        0.0
    )

    df["profit_margin_pct"] = df["profit_margin_pct"].round(2)

    # -----------------------------------------------------------------------
    # KPI 4: Budget Variance Amount
    # -----------------------------------------------------------------------
    # Budget variance compares actual spending to the planned budget.
    #
    # Formula:
    #   budget_variance_amt = total_expense - total_budget
    #
    # Interpretation:
    #   Positive value = overspend
    #   Negative value = underspend
    df["budget_variance_amt"] = (
        df["total_expense"] - df["total_budget"]
    ).round(2)

    # -----------------------------------------------------------------------
    # KPI 5: Budget Variance %
    # -----------------------------------------------------------------------
    # This shows the budget variance as a percentage of the budget.
    #
    # Formula:
    #   budget_variance_pct = budget_variance_amt / total_budget * 100
    #
    # Example:
    #   expense = 12,000
    #   budget = 10,000
    #   variance = 2,000
    #   variance % = 20%
    df["budget_variance_pct"] = np.where(
        df["total_budget"] > 0,
        (df["budget_variance_amt"] / df["total_budget"]) * 100,
        0.0
    )

    df["budget_variance_pct"] = df["budget_variance_pct"].round(2)

    # -----------------------------------------------------------------------
    # KPI 6: Month-over-Month Revenue Change %
    # -----------------------------------------------------------------------
    # MoM compares this month's revenue with the previous month's revenue.
    #
    # Formula:
    #   revenue_mom_pct = (current_month - previous_month) / previous_month * 100
    #
    # I group by department and region so each business segment is compared
    # only with its own previous month.
    #
    # Note:
    # This compares December to January because they are consecutive months.
    # That is usually useful for business trend analysis.
    df["revenue_mom_pct"] = (
        df.groupby(["department", "region"])["total_revenue"]
        .pct_change() * 100
    )

    df["revenue_mom_pct"] = df["revenue_mom_pct"].round(2)

    # -----------------------------------------------------------------------
    # KPI 7: Rolling 3-Month Average Expense
    # -----------------------------------------------------------------------
    # A rolling average smooths out one-time spikes and makes trends easier
    # to see in the dashboard.
    #
    # For each department and region, this calculates the average expense
    # over the current month and the previous two months.
    #
    # min_periods=1 means the first and second months still get a value
    # even when there are not yet three months available.
    df["expense_rolling_3m"] = (
        df.groupby(["department", "region"])["total_expense"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    df["expense_rolling_3m"] = df["expense_rolling_3m"].round(2)

    # -----------------------------------------------------------------------
    # KPI 8: Same Period Last Year Revenue
    # -----------------------------------------------------------------------
    # Same Period Last Year means the revenue from the same month in the previous year.
    #
    # Example:
    # For March 2025, SPLY is March 2024.
    #
    # I create a lookup table by taking last year's revenue and shifting
    # the year forward by 1, so it can join to the current year.
    last_year_lookup = df[
        ["department", "region", "year", "month", "total_revenue"]
    ].copy()

    # Shift revenue forward so 2024 revenue can match 2025 rows.
    last_year_lookup["year"] = last_year_lookup["year"] + 1

    last_year_lookup = last_year_lookup.rename(
        columns={"total_revenue": "revenue_sply"}
    )

    df = df.merge(
        last_year_lookup,
        on=["department", "region", "year", "month"],
        how="left"
    )

    # -----------------------------------------------------------------------
    # KPI 9: Year-over-Year Revenue Growth %
    # -----------------------------------------------------------------------
    # YoY compares current revenue to the same month last year.
    #
    # Formula:
    #   revenue_yoy_pct = (current_revenue - revenue_sply) / revenue_sply * 100
    #
    # If there is no previous-year revenue, I return NaN instead of 0.
    # This is more accurate because it means "not available", not "no growth".
    df["revenue_yoy_pct"] = np.where(
        df["revenue_sply"] > 0,
        ((df["total_revenue"] - df["revenue_sply"]) / df["revenue_sply"]) * 100,
        np.nan
    )

    df["revenue_yoy_pct"] = df["revenue_yoy_pct"].round(2)

    # -----------------------------------------------------------------------
    # KPI 10: Expense Ratio
    # -----------------------------------------------------------------------
    # Expense ratio shows how much of the revenue is consumed by expenses.
    #
    # Formula:
    #   expense_ratio = total_expense / total_revenue
    #
    # Interpretation:
    #   0.70 = expenses are 70% of revenue
    #   1.10 = expenses are higher than revenue
    df["expense_ratio"] = np.where(
        df["total_revenue"] > 0,
        df["total_expense"] / df["total_revenue"],
        np.nan
    )

    df["expense_ratio"] = df["expense_ratio"].round(4)

    # -----------------------------------------------------------------------
    # KPI 11: Spending Velocity
    # -----------------------------------------------------------------------
    # Spending velocity measures how quickly the department is using its budget.
    #
    # Formula:
    #   spending_velocity = total_expense / total_budget
    #
    # Interpretation:
    #   0.80 = used 80% of budget
    #   1.00 = exactly on budget
    #   1.20 = 20% over budget
    #
    # This can also be useful later as a feature for a budget overrun ML model.
    df["spending_velocity"] = np.where(
        df["total_budget"] > 0,
        df["total_expense"] / df["total_budget"],
        0.0
    )

    df["spending_velocity"] = df["spending_velocity"].round(4)

    print(f"  KPIs calculated: {len(df):,} rows, {len(df.columns)} columns")

    return df


# ---------------------------------------------------------------------------
# KPI verification summary
# ---------------------------------------------------------------------------

def get_kpi_summary(df: pd.DataFrame) -> None:
    """
    Print a quick KPI summary after calculations.

    This is not a replacement for formal testing.
    It is a simple verification step to help me quickly check whether the calculated values look reasonable.
    """

    print("\n── KPI Verification Summary ──")

    # -----------------------------------------------------------------------
    # Overall revenue and expense
    # -----------------------------------------------------------------------
    total_revenue = df["total_revenue"].sum()
    total_expense = df["total_expense"].sum()
    net_result = total_revenue - total_expense

    print(f"  Total Revenue (all years):  ${total_revenue:>15,.2f}")
    print(f"  Total Expense (all years):  ${total_expense:>15,.2f}")
    print(f"  Net Result:                 ${net_result:>15,.2f}")

    # -----------------------------------------------------------------------
    # Average budget variance by department
    # -----------------------------------------------------------------------
    # This helps verify whether department-level budget patterns make sense.
    print("\n  Avg Monthly Budget Variance % by Department:")

    department_variance = (
        df.groupby("department")["budget_variance_pct"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    for department, variance_pct in department_variance.items():
        if department == "IT":
            flag = " <- expected overspend"
        elif department == "HR":
            flag = " <- expected underspend"
        else:
            flag = ""

        print(f"    {department:<12}: {variance_pct:>7.2f}%{flag}")

    # -----------------------------------------------------------------------
    # Optional sanity checks
    # -----------------------------------------------------------------------
    # These checks do not stop the pipeline.
    # They only print warnings if some values may need review.
    if df["profit_margin_pct"].isna().any():
        print("\n   Some profit margin values are missing.")

    if (df["spending_velocity"] > 1).any():
        over_budget_rows = (df["spending_velocity"] > 1).sum()
        print(f"   {over_budget_rows:,} rows are over budget based on spending velocity.")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   

    import os
    import sys

  
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    sys.path.insert(0, project_root)

    from etl.extract.source_connector import read_source_file
    from etl.transform.cleaner import clean_monthly_summary

    print("Testing kpi_builder.py")
    print("─" * 40)

    # -----------------------------------------------------------------------
    # Step 1: Read the raw monthly summary file
    # -----------------------------------------------------------------------
    monthly_summary_raw, metadata = read_source_file(
        "data/raw/monthly_summary.xlsx"
    )

    # -----------------------------------------------------------------------
    # Step 2: Clean the monthly summary data
    # -----------------------------------------------------------------------
    monthly_summary_clean = clean_monthly_summary(monthly_summary_raw)

    # -----------------------------------------------------------------------
    # Step 3: Calculate KPI columns
    # -----------------------------------------------------------------------
    monthly_summary_kpis = calculate_kpis(monthly_summary_clean)

    # -----------------------------------------------------------------------
    # Step 4: Print a simple KPI verification summary
    # -----------------------------------------------------------------------
    get_kpi_summary(monthly_summary_kpis)

    # -----------------------------------------------------------------------
    # Step 5: Print a small sample for manual review
    # -----------------------------------------------------------------------
    print("\nSample KPI columns:")

    kpi_columns = [
        "year",
        "month",
        "department",
        "region",
        "total_revenue",
        "revenue_ytd",
        "budget_variance_pct",
        "revenue_mom_pct",
        "revenue_yoy_pct",
        "spending_velocity",
    ]

    print(monthly_summary_kpis[kpi_columns].head(6).to_string(index=False))