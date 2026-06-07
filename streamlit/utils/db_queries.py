"""
This file contains all database queries used by the Streamlit dashboard.

This makes the project:
1. Easier to maintain
2. Easier to debug
3. Easier to reuse
4. Cleaner and more professional

We also use Streamlit caching so the same query does not run again and again every time the user interacts with the dashboard.
"""

import os
import sys

import pandas as pd
import streamlit as st
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Add the project root folder to Python path
# ---------------------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Import the SQLAlchemy engine
# ---------------------------------------------------------------------------


from etl.extract.db import engine


# ---------------------------------------------------------------------------
# Function 1: Load monthly summary data
# ---------------------------------------------------------------------------
# This function returns aggregated monthly financial data.
#
# It groups data by:
# - year
# - month
# - quarter
# - department
# - region
#
# It calculates:
# - total revenue
# - total expense
# - total budget

# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_monthly_summary() -> pd.DataFrame:
    """
    Load monthly revenue and expense summary by department and region.

    The result is cached for 1 hour.


    Returns:
        pd.DataFrame: Monthly summarized financial data.
    """

    query = """
        SELECT
            dt.year,
            dt.month,
            dt.month_name,
            dt.quarter,
            dt.quarter_name,

            d.dept_name AS department,
            r.region_name AS region,

            SUM(
                CASE
                    WHEN f.transaction_type = 'REVENUE'
                    THEN f.amount
                    ELSE 0
                END
            ) AS total_revenue,

            SUM(
                CASE
                    WHEN f.transaction_type = 'EXPENSE'
                    THEN f.amount
                    ELSE 0
                END
            ) AS total_expense,

            SUM(
                CASE
                    WHEN f.transaction_type = 'EXPENSE'
                    THEN f.budget_amount
                    ELSE 0
                END
            ) AS total_budget

        FROM production.fact_financials f

        JOIN production.dim_date dt
            ON f.date_key = dt.date_key

        JOIN production.dim_department d
            ON f.dept_key = d.dept_key

        JOIN production.dim_region r
            ON f.region_key = r.region_key

        GROUP BY
            dt.year,
            dt.month,
            dt.month_name,
            dt.quarter,
            dt.quarter_name,
            d.dept_name,
            r.region_name

        ORDER BY
            dt.year,
            dt.month,
            d.dept_name
    """

    # Open a database connection, run the query, and return the result as a DataFrame.
    raw_conn = engine.raw_connection()
    try:
        df = pd.read_sql_query(query, raw_conn)
    finally:
        raw_conn.close()

    return df


# ---------------------------------------------------------------------------
# Function 2: Load detailed transaction data
# ---------------------------------------------------------------------------
# This function returns transaction-level data.
#
# Unlike get_monthly_summary(), this function does not aggregate the data.
# It returns individual transactions with their related department, region, category, and date information.
#
# This is useful for:
# - anomaly detection page
# - drill-through page
# - transaction tables
# - detailed filtering
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_transactions() -> pd.DataFrame:
    """
    Load all financial transactions with readable dimension columns.

    The fact table stores keys like dept_key, region_key, category_key, and date_key. 

    we join the fact table with dimension tables to get names like:
    - department name
    - region name
    - category name
    - full date

    Returns:
        pd.DataFrame: Detailed transaction-level data.
    """

    query = """
        SELECT
            f.transaction_id,
            f.amount,
            f.budget_amount,
            f.transaction_type,
            f.vendor,
            f.is_anomaly,

            d.dept_name AS department,
            r.region_name AS region,
            c.category_name AS category,

            dt.full_date AS date,
            dt.year,
            dt.month,
            dt.month_name,
            dt.quarter

        FROM production.fact_financials f

        JOIN production.dim_department d
            ON f.dept_key = d.dept_key

        JOIN production.dim_region r
            ON f.region_key = r.region_key

        JOIN production.dim_category c
            ON f.category_key = c.category_key

        JOIN production.dim_date dt
            ON f.date_key = dt.date_key

        ORDER BY
            dt.full_date DESC
    """

    raw_conn = engine.raw_connection()
    try:
        df = pd.read_sql_query(query, raw_conn)
    finally:
        raw_conn.close()
        df["date"] = pd.to_datetime(df["date"])
    return df

    # Convert the date column to datetime so Streamlit and Plotly can filter and plot it correctly.
    

  


# ---------------------------------------------------------------------------
# Function 3: Load expense transactions only
# ---------------------------------------------------------------------------
# This function reuses get_transactions() instead of writing a new SQL query.
#
# # Because get_transactions() already loads all transaction data.
# We only need to filter it to keep expense transactions.
#
# This is useful for anomaly detection because anomalies in this project are based on unusual expenses.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_expense_transactions() -> pd.DataFrame:
    """
    Load expense transactions only.

    This function is mainly used by the anomaly detection page.

    Returns:
        pd.DataFrame: Only transactions where transaction_type is EXPENSE.
    """

    df = get_transactions()

    expense_df = df[df["transaction_type"] == "EXPENSE"].copy()

    return expense_df


# ---------------------------------------------------------------------------
# Function 4: Calculate high-level KPI summary
# ---------------------------------------------------------------------------
# This function calculates top-level numbers for the dashboard.

# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_kpi_summary(year: int = None) -> dict:
    """
    Calculate executive KPI summary.

    Args:
        year (int, optional): If provided, calculate KPIs only for that year.
                              If None, calculate KPIs for all available years.

    Returns:
        dict: A dictionary containing top-level KPI values.
    """

    # Start from the monthly summary data because it is already aggregated.
    df = get_monthly_summary()

    # If the user selected a specific year, filter the data to that year only.
    if year is not None:
        df = df[df["year"] == year]

    # Calculate total revenue.
    total_revenue = df["total_revenue"].sum()

    # Calculate total expenses.
    total_expense = df["total_expense"].sum()

    # Calculate total budget.
    total_budget = df["total_budget"].sum()

    # How much money is left after expenses.
    net_profit = total_revenue - total_expense

    # Profit margin shows net profit as a percentage of revenue.
    # We check total_revenue > 0 to avoid division by zero.
    if total_revenue > 0:
        profit_margin = (net_profit / total_revenue) * 100
    else:
        profit_margin = 0

    # Budget variance amount shows how much actual expenses differ from budget.
    # Positive value means actual expenses are higher than budget.
    # Negative value means actual expenses are lower than budget.
    budget_variance_amt = total_expense - total_budget

    # Budget variance percentage shows the variance relative to the budget.
    # We check total_budget > 0 to avoid division by zero.
    if total_budget > 0:
        budget_variance_pct = (budget_variance_amt / total_budget) * 100
    else:
        budget_variance_pct = 0

    # Return the results as a dictionary.
    # A dictionary is convenient because Streamlit pages can access values by name.
    return {
        "total_revenue": total_revenue,
        "total_expense": total_expense,
        "total_budget": total_budget,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "budget_variance_amt": budget_variance_amt,
        "budget_variance_pct": budget_variance_pct,
    }