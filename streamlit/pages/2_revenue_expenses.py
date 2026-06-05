"""
This page gives a more detailed analysis of revenue and expenses.
It helps the CFO or analyst understand monthly trends, expense categories, year-over-year expense changes, and regional revenue performance.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Revenue & Expenses - CFO Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Python import paths
# ---------------------------------------------------------------------------


current_file = os.path.abspath(__file__)
pages_dir = os.path.dirname(current_file)
streamlit_dir = os.path.dirname(pages_dir)

sys.path.insert(0, streamlit_dir)


# ---------------------------------------------------------------------------
# Import reusable helper functions
# ---------------------------------------------------------------------------


from utils.db_queries import get_monthly_summary, get_transactions
from utils.formatters import format_currency


# ---------------------------------------------------------------------------
# Load custom CSS
# ---------------------------------------------------------------------------


css_path = os.path.join(streamlit_dir, "assets", "style.css")

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------


st.markdown(
    '<div class="page-header">Revenue & Expenses - Deep Dive</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# get_monthly_summary() returns monthly aggregated financial data.
# get_transactions() returns transaction-level data.
#
# We need both because:
# - monthly summary is better for trends and KPI totals
# - transactions are better for category-level expense breakdown
# ---------------------------------------------------------------------------

df = get_monthly_summary()
df_txn = get_transactions()


# ---------------------------------------------------------------------------
# Stop page if data is missing
# ---------------------------------------------------------------------------
# This prevents errors if the database is empty or the ETL did not run.
# ---------------------------------------------------------------------------

if df.empty or df_txn.empty:
    st.warning("No data is available. Please run the ETL pipeline first.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
# These filters allow the user to analyze revenue and expenses by:
# - year
# - region
# - department
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    years = sorted(df["year"].unique(), reverse=True)
    year_options = ["All"] + [str(year) for year in years]
    default_year_index = 1 if len(year_options) > 1 else 0

    selected_year = st.selectbox(
        "Year",
        year_options,
        index=default_year_index
    )

    regions = ["All"] + sorted(df["region"].dropna().unique().tolist())

    selected_region = st.selectbox(
        "Region",
        regions
    )

    departments = ["All"] + sorted(df["department"].dropna().unique().tolist())

    selected_department = st.selectbox(
        "Department",
        departments
    )


# ---------------------------------------------------------------------------
# Apply filters to monthly summary data
# ---------------------------------------------------------------------------
# This filtered DataFrame is used for KPIs and monthly trend charts.
# ---------------------------------------------------------------------------

df_filtered = df.copy()

if selected_year != "All":
    df_filtered = df_filtered[df_filtered["year"] == int(selected_year)]

if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]

if selected_department != "All":
    df_filtered = df_filtered[df_filtered["department"] == selected_department]


# ---------------------------------------------------------------------------
# Apply filters to transaction-level data
# ---------------------------------------------------------------------------
# This filtered DataFrame is used for category-level expense breakdown.
# ---------------------------------------------------------------------------

df_txn_filtered = df_txn.copy()

if selected_year != "All":
    df_txn_filtered = df_txn_filtered[df_txn_filtered["year"] == int(selected_year)]

if selected_region != "All":
    df_txn_filtered = df_txn_filtered[df_txn_filtered["region"] == selected_region]

if selected_department != "All":
    df_txn_filtered = df_txn_filtered[df_txn_filtered["department"] == selected_department]


# ---------------------------------------------------------------------------
# Stop page if filters return no rows
# ---------------------------------------------------------------------------

if df_filtered.empty:
    st.warning("No monthly summary data found for the selected filters.")
    st.stop()

if df_txn_filtered.empty:
    st.warning("No transaction data found for the selected filters.")
    st.stop()


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------
# total_revenue:
# Total revenue after filters.
#
# total_expense:
# Total expense after filters.
#
# expense_ratio:
# Shows how much expense exists for each dollar of revenue.
# Example: 0.65x means expenses are 65% of revenue.
# ---------------------------------------------------------------------------

total_revenue = df_filtered["total_revenue"].sum()
total_expense = df_filtered["total_expense"].sum()

expense_ratio = (
    total_expense / total_revenue
    if total_revenue > 0
    else 0
)


# ---------------------------------------------------------------------------
# Year-over-year comparison
# ---------------------------------------------------------------------------
# If the user selects a specific year, compare revenue and expenses to the previous year.
#
# If "All" is selected, YoY comparison is not shown.
# ---------------------------------------------------------------------------

if selected_year != "All":
    prior_year = int(selected_year) - 1

    prior_df = df[df["year"] == prior_year]

    if selected_region != "All":
        prior_df = prior_df[prior_df["region"] == selected_region]

    if selected_department != "All":
        prior_df = prior_df[prior_df["department"] == selected_department]

    prior_revenue = prior_df["total_revenue"].sum()
    prior_expense = prior_df["total_expense"].sum()

    revenue_yoy = (
        ((total_revenue - prior_revenue) / prior_revenue) * 100
        if prior_revenue > 0
        else None
    )

    expense_yoy = (
        ((total_expense - prior_expense) / prior_expense) * 100
        if prior_expense > 0
        else None
    )

else:
    revenue_yoy = None
    expense_yoy = None


# ---------------------------------------------------------------------------
# Helper function for YoY delta labels
# ---------------------------------------------------------------------------

def format_yoy(value: float | None) -> str | None:
    """
    Format YoY percentage change for Streamlit metric cards.

    Args:
        value (float | None): YoY percentage value.

    Returns:
        str | None: Formatted YoY label.
    """

    if value is None:
        return None

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.1f}% YoY"


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Revenue YTD",
        value=format_currency(total_revenue),
        delta=format_yoy(revenue_yoy),
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Expense YTD",
        value=format_currency(total_expense),
        delta=format_yoy(expense_yoy),
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Expense Ratio",
        value=f"{expense_ratio:.2f}x",
        delta="Expense / Revenue",
        delta_color="off"
    )

st.divider()


# ---------------------------------------------------------------------------
# Chart 1: Monthly Revenue vs Expense
# ---------------------------------------------------------------------------
# This chart compares monthly expenses and revenue.
#
# Bars show expenses.
# Line shows revenue.
#
# Two y-axes are used because revenue and expenses may have different scales.
# ---------------------------------------------------------------------------

st.subheader("Monthly Revenue vs Expense")

monthly = (
    df_filtered
    .groupby(["month", "month_name"], as_index=False)
    .agg(
        total_revenue=("total_revenue", "sum"),
        total_expense=("total_expense", "sum")
    )
    .sort_values("month")
)

fig_combo = go.Figure()

fig_combo.add_trace(
    go.Bar(
        x=monthly["month_name"],
        y=monthly["total_expense"],
        name="Expense",
        marker_color="#1B3A6B",
        yaxis="y"
    )
)

fig_combo.add_trace(
    go.Scatter(
        x=monthly["month_name"],
        y=monthly["total_revenue"],
        name="Revenue",
        line=dict(color="#E67E22", width=2.5),
        mode="lines+markers",
        yaxis="y2"
    )
)

fig_combo.update_layout(
    height=260,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="", tickformat="$~s", showgrid=True),
    yaxis2=dict(
        title="",
        overlaying="y",
        side="right",
        tickformat="$~s",
        showgrid=False
    ),
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="",
    bargap=0.2
)

st.plotly_chart(fig_combo, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Row 2: Top expense categories and YoY expense comparison table
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1.4])


# ---------------------------------------------------------------------------
# Chart 2: Top Expense Categories
# ---------------------------------------------------------------------------
# This chart uses transaction-level data because category exists at the
# transaction level.
# ---------------------------------------------------------------------------

with col_left:
    st.subheader("Top Expense Categories")

    expense_transactions = df_txn_filtered[
        df_txn_filtered["transaction_type"] == "EXPENSE"
    ]

    category_expense = (
        expense_transactions
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=True)
        .tail(8)
    )

    if category_expense.empty:
        st.info("No expense category data available for the selected filters.")
    else:
        fig_category = go.Figure(
            go.Bar(
                x=category_expense["amount"],
                y=category_expense["category"],
                orientation="h",
                marker_color="#2E75B6",
                text=[
                    format_currency(value)
                    for value in category_expense["amount"]
                ],
                textposition="outside"
            )
        )

        fig_category.update_layout(
            height=320,
            margin=dict(l=0, r=80, t=10, b=0),
            xaxis=dict(tickformat="$~s", title=""),
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_category, use_container_width=True)


# ---------------------------------------------------------------------------
# Table: Year-over-Year Expense Comparison
# ---------------------------------------------------------------------------
# This table compares expenses by department across years.
# It also calculates YoY percentage change for each year after the first year.
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("Year-over-Year Expense Comparison")

    yoy_table = (
        df
        .groupby(["department", "year"])["total_expense"]
        .sum()
        .reset_index()
        .pivot(index="department", columns="year", values="total_expense")
        .round(0)
    )

    year_columns = sorted(yoy_table.columns.tolist())

    for i in range(1, len(year_columns)):
        previous_year = year_columns[i - 1]
        current_year = year_columns[i]

        yoy_column_name = f"YoY% {current_year}"

        yoy_table[yoy_column_name] = (
            (
                yoy_table[current_year] - yoy_table[previous_year]
            )
            / yoy_table[previous_year]
            * 100
        ).round(1)

    display_table = yoy_table.copy()

    for year in year_columns:
        display_table[year] = display_table[year].apply(
            lambda value: format_currency(value) if pd.notna(value) else "-"
        )

    yoy_percentage_columns = [
        column
        for column in display_table.columns
        if "YoY%" in str(column)
    ]

    for column in yoy_percentage_columns:
        display_table[column] = yoy_table[column].apply(
            lambda value: (
                f"+{value:.1f}%"
                if pd.notna(value) and value > 0
                else f"{value:.1f}%"
                if pd.notna(value)
                else "-"
            )
        )

    display_table.columns = [str(column) for column in display_table.columns]
    display_table.index.name = "Department"
    display_table = display_table.reset_index()

    st.dataframe(
        display_table,
        use_container_width=True,
        height=280,
        hide_index=True
    )

st.divider()


# ---------------------------------------------------------------------------
# Chart 3: Revenue by Region and Year
# ---------------------------------------------------------------------------
# This chart shows how much revenue each region contributed per year.
# It uses a stacked horizontal bar chart.
# ---------------------------------------------------------------------------

st.subheader("Revenue by Region and Year")

region_year = (
    df
    .groupby(["year", "region"], as_index=False)["total_revenue"]
    .sum()
)

region_colours = {
    "East": "#1B3A6B",
    "West": "#2E75B6",
    "Central": "#17A589"
}

fig_region = px.bar(
    region_year,
    x="total_revenue",
    y="year",
    color="region",
    orientation="h",
    color_discrete_map=region_colours,
    text_auto=False,
    labels={
        "total_revenue": "Revenue",
        "year": ""
    },
    barmode="stack"
)

fig_region.update_traces(
    texttemplate="%{x:$~s}",
    textposition="inside",
    textfont=dict(color="white", size=11)
)

fig_region.update_layout(
    height=200,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(tickformat="$~s", title=""),
    yaxis=dict(type="category"),
    legend=dict(orientation="h", y=1.2, title=""),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig_region, use_container_width=True)