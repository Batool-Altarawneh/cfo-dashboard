"""
This file is part of the Transform Layer in the ETL pipeline.

It receives raw/staging DataFrames and applies basic cleaning rules
before the data is used for:
1. KPI calculations
2. star schema creation
3. loading into production tables

"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------
# These lists define the values that are expected in the cleaned data.
# I keep them at the top of the file so they are easy to find and update.
# This also avoids repeating the same values in different functions.

VALID_DEPARTMENTS = ["IT", "Marketing", "Sales", "HR", "Operations"]
VALID_REGIONS = ["East", "West", "Central"]
VALID_TRANSACTION_TYPES = ["EXPENSE", "REVENUE"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def standardize_department(value):
    """
    Standardize department names into one consistent format.

    Examples:
    - ' it ' becomes 'IT'
    - 'hr' becomes 'HR'
    - 'sales' becomes 'Sales'

    I created this helper because department names have special cases like IT and HR that should stay uppercase.
    """

    if pd.isna(value):
        return value

    value = str(value).strip().title()

    department_map = {
        "It": "IT",
        "Hr": "HR",
        "Marketing": "Marketing",
        "Sales": "Sales",
        "Operations": "Operations",
    }

    return department_map.get(value, value)


def standardize_boolean(value):
    """
    Convert different possible anomaly values into True/False.

    """

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    if value in ["true", "1", "yes", "y"]:
        return True

    if value in ["false", "0", "no", "n"]:
        return False

    # If the value is unexpected, I default it to False.
    # In a production system, I might log this for data quality review.
    return False


def report_invalid_values(df, column_name, valid_values):
    """
    Print values that are not part of the expected list.

    This does not remove rows by itself.
    It only helps me see if the raw data contains unexpected categories.
    """

    invalid_values = sorted(
        set(df[column_name].dropna().unique()) - set(valid_values)
    )

    if invalid_values:
        print(f"   Unexpected values in {column_name}: {invalid_values}")


# ---------------------------------------------------------------------------
# Clean transactions
# ---------------------------------------------------------------------------

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw transactions DataFrame.

    This function prepares transaction-level data for KPI calculations and star schema building.

    Main cleaning steps:
    1. Standardize column names
    2. Remove duplicate rows
    3. Convert columns to correct data types
    4. Standardize text/category values
    5. Remove rows with missing critical fields
    6. Remove invalid transaction amounts
    7. Reset the index

    """

    
    df = df.copy()

    print("Cleaning transactions...")

    # -----------------------------------------------------------------------
    # Step 1: Standardize column names
    # -----------------------------------------------------------------------
    # Source files can sometimes have columns like 'Amount', ' amount',
    # or 'Department '. Lowercasing and stripping spaces makes the rest of
    # the code more reliable.
    df.columns = df.columns.str.lower().str.strip()

    # -----------------------------------------------------------------------
    # Step 2: Remove exact duplicate rows
    # -----------------------------------------------------------------------
    # Exact duplicates usually happen because of repeated exports or file merges.
    # I track the count before and after so I can see what was removed.
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicate_rows = rows_before - len(df)

    if duplicate_rows > 0:
        print(f"  Removed {duplicate_rows:,} duplicate rows")

    # -----------------------------------------------------------------------
    # Step 3: Convert columns to the correct data types
    # -----------------------------------------------------------------------
    # Dates must be datetime so we can use them later for YTD, MoM,and date dimension logic.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Year and month should be numeric.
    # I use pandas nullable Int64 because it can handle missing values.
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    # Amount fields must be numeric because KPI calculations depend on them.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["budget_amount"] = pd.to_numeric(df["budget_amount"], errors="coerce")

    # Convert anomaly flag safely to True/False.
    df["is_anomaly"] = df["is_anomaly"].apply(standardize_boolean)

    # -----------------------------------------------------------------------
    # Step 4: Standardize text/category values
    # -----------------------------------------------------------------------
    # These columns are used later to build dimension tables.
    # If the text is inconsistent, Power BI may show duplicate categories.
    df["department"] = df["department"].apply(standardize_department)

    df["region"] = (
        df["region"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["transaction_type"] = (
        df["transaction_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["vendor"] = (
        df["vendor"]
        .astype("string")
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    # -----------------------------------------------------------------------
    # Step 5: Check for unexpected category values
    # -----------------------------------------------------------------------
    # These checks help me catch spelling mistakes or unexpected values early.
    # For example: 'Sale' instead of 'Sales', or 'North' instead of 'East'.
    report_invalid_values(df, "department", VALID_DEPARTMENTS)
    report_invalid_values(df, "region", VALID_REGIONS)
    report_invalid_values(df, "transaction_type", VALID_TRANSACTION_TYPES)

    # -----------------------------------------------------------------------
    # Step 6: Drop rows with missing critical fields
    # -----------------------------------------------------------------------
    # These columns are required for analysis.
    # If any of them are missing, the row cannot be used safely.
    critical_columns = [
        "transaction_id",
        "date",
        "year",
        "month",
        "department",
        "region",
        "amount",
        "transaction_type",
    ]

    rows_before = len(df)
    df = df.dropna(subset=critical_columns)
    dropped_null_rows = rows_before - len(df)

    if dropped_null_rows > 0:
        print(f"  Dropped {dropped_null_rows:,} rows with missing critical values")

    # -----------------------------------------------------------------------
    # Step 7: Remove invalid amount values
    # -----------------------------------------------------------------------
    # In this project, transaction amounts should be positive.
    # The transaction_type column tells us whether the amount is revenue or expense, so we do not store expenses as negative numbers.
    invalid_amount_rows = df[df["amount"] <= 0]

    if len(invalid_amount_rows) > 0:
        print(f"  Removed {len(invalid_amount_rows):,} rows with amount <= 0")
        df = df[df["amount"] > 0]

    # -----------------------------------------------------------------------
    # Step 8: Keep only valid category values
    # -----------------------------------------------------------------------
    # After reporting invalid values, I remove them so production data stays clean.
    df = df[df["department"].isin(VALID_DEPARTMENTS)]
    df = df[df["region"].isin(VALID_REGIONS)]
    df = df[df["transaction_type"].isin(VALID_TRANSACTION_TYPES)]

    # -----------------------------------------------------------------------
    # Step 9: Reset index
    # -----------------------------------------------------------------------
    # After deleting rows, the old index may have gaps.
    # Resetting it keeps the DataFrame clean and easier to read.
    df = df.reset_index(drop=True)

    print(f"   Transactions cleaned: {len(df):,} rows remaining")

    return df


# ---------------------------------------------------------------------------
# Clean monthly summary
# ---------------------------------------------------------------------------

def clean_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw monthly summary DataFrame.

    This table is more aggregated than transactions.
    It usually contains one row per year, month, department, and region.

   
    """

    df = df.copy()

    print("Cleaning monthly summary...")

    # -----------------------------------------------------------------------
    # Step 1: Standardize column names
    # -----------------------------------------------------------------------
    df.columns = df.columns.str.lower().str.strip()

    # -----------------------------------------------------------------------
    # Step 2: Remove exact duplicate rows
    # -----------------------------------------------------------------------
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicate_rows = rows_before - len(df)

    if duplicate_rows > 0:
        print(f"  Removed {duplicate_rows:,} duplicate rows")

    # -----------------------------------------------------------------------
    # Step 3: Convert columns to correct data types
    # -----------------------------------------------------------------------
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    df["total_expense"] = pd.to_numeric(df["total_expense"], errors="coerce")
    df["total_budget"] = pd.to_numeric(df["total_budget"], errors="coerce")

    df["headcount"] = pd.to_numeric(df["headcount"], errors="coerce").astype("Int64")

    # -----------------------------------------------------------------------
    # Step 4: Standardize category values
    # -----------------------------------------------------------------------
    df["department"] = df["department"].apply(standardize_department)

    df["region"] = (
        df["region"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # -----------------------------------------------------------------------
    # Step 5: Check for unexpected category values
    # -----------------------------------------------------------------------
    report_invalid_values(df, "department", VALID_DEPARTMENTS)
    report_invalid_values(df, "region", VALID_REGIONS)

    # -----------------------------------------------------------------------
    # Step 6: Drop rows with missing critical fields
    # -----------------------------------------------------------------------
    # These fields define the grain of the monthly summary table.
    # Without them, we do not know what month/department/region the row belongs to.
    critical_columns = [
        "year",
        "month",
        "department",
        "region",
    ]

    rows_before = len(df)
    df = df.dropna(subset=critical_columns)
    dropped_null_rows = rows_before - len(df)

    if dropped_null_rows > 0:
        print(f"  Dropped {dropped_null_rows:,} rows with missing critical values")

    # -----------------------------------------------------------------------
    # Step 7: Fill missing numeric values based on business meaning
    # -----------------------------------------------------------------------
    # Some departments may not generate direct revenue.
    # For example, IT or HR can have expenses but no revenue.
    # In that case, missing revenue can reasonably be treated as 0.
    df["total_revenue"] = df["total_revenue"].fillna(0.0)

    # For expenses and budget, I prefer not to silently fill missing values unless there is a clear business rule.
    # Here, I remove rows where expense or budget is missing because they are required for variance and budget analysis.
    rows_before = len(df)
    df = df.dropna(subset=["total_expense", "total_budget"])
    dropped_financial_rows = rows_before - len(df)

    if dropped_financial_rows > 0:
        print(f"  Dropped {dropped_financial_rows:,} rows with missing expense/budget values")

    # -----------------------------------------------------------------------
    # Step 8: Remove invalid numeric values
    # -----------------------------------------------------------------------
    # Negative totals usually indicate a data quality issue in this dataset.
    df = df[df["total_revenue"] >= 0]
    df = df[df["total_expense"] >= 0]
    df = df[df["total_budget"] >= 0]

    # Headcount should also not be negative.
    df = df[(df["headcount"].isna()) | (df["headcount"] >= 0)]

    # -----------------------------------------------------------------------
    # Step 9: Keep only valid category values
    # -----------------------------------------------------------------------
    df = df[df["department"].isin(VALID_DEPARTMENTS)]
    df = df[df["region"].isin(VALID_REGIONS)]

    # -----------------------------------------------------------------------
    # Step 10: Reset index
    # -----------------------------------------------------------------------
    df = df.reset_index(drop=True)

    print(f"   Monthly summary cleaned: {len(df):,} rows remaining")

    return df


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    This block runs only when I execute this file directly.

    Example:
        python etl/transform/cleaner.py

    I use it as a quick local test to make sure the cleaning functions
    work before connecting this file to the full ETL pipeline.
    """

    import os
    import sys

    # Add the project root folder to Python path.
    # This helps Python find modules like etl.extract.source_connector when this file is run directly.
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    sys.path.insert(0, project_root)

    from etl.extract.source_connector import read_source_file

    print("Testing cleaner.py")
    print("─" * 40)

    # -----------------------------------------------------------------------
    # Test transactions cleaning
    # -----------------------------------------------------------------------
    transactions_df, transactions_metadata = read_source_file(
        "data/raw/transactions.xlsx"
    )

    transactions_clean = clean_transactions(transactions_df)

    # -----------------------------------------------------------------------
    # Test monthly summary cleaning
    # -----------------------------------------------------------------------
    monthly_summary_df, monthly_summary_metadata = read_source_file(
        "data/raw/monthly_summary.xlsx"
    )

    monthly_summary_clean = clean_monthly_summary(monthly_summary_df)

    # -----------------------------------------------------------------------
    # Print small samples to visually check the output
    # -----------------------------------------------------------------------
    print("\nTransaction sample:")
    print(
        transactions_clean[
            [
                "transaction_id",
                "date",
                "department",
                "region",
                "amount",
                "transaction_type",
            ]
        ].head(3)
    )

    print("\nMonthly summary sample:")
    print(
        monthly_summary_clean[
            [
                "year",
                "month",
                "department",
                "region",
                "total_revenue",
                "total_expense",
                "total_budget",
            ]
        ].head(3)
    )