"""
1_executive_summary.py
----------------------
Executive Summary page for the CFO Financial Dashboard.

This page gives the CFO a high-level view of the company's financial health.
It shows top-level KPIs such as revenue, expenses, net profit, and budget variance, plus summary charts for monthly revenue, expenses by department, revenue vs expense, and revenue by region.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Set Streamlit page configuration
# ---------------------------------------------------------------------------
# This controls the browser tab title, page icon, and page width.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Executive Summary - CFO Dashboard",
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
# Import reusable project functions
# ---------------------------------------------------------------------------
# db_queries.py handles database queries.
# formatters.py handles currency and percentage formatting.
# ---------------------------------------------------------------------------

from utils.db_queries import get_monthly_summary
from utils.formatters import format_currency, format_percentage


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
# Helper function: format year-over-year delta
# ---------------------------------------------------------------------------
# Streamlit st.metric can show a delta under the main KPI value.
#
# Example:
# +8.4% vs prior year
# -3.2% vs prior year
#
# This helper keeps the delta formatting consistent across KPI cards.
# ---------------------------------------------------------------------------

def format_yoy_delta(value: float | None) -> str | None:
    """
    Format a year-over-year percentage change for Streamlit metric cards.

   """

    if value is None:
        return None

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.1f}% vs prior year"


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
# This uses the .page-header class from style.css.
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="page-header">Executive Summary</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# get_monthly_summary() loads monthly aggregated revenue, expense, and budget data from PostgreSQL.
#
# The returned DataFrame should contain columns like:
# year, month, month_name, department, region, total_revenue,total_expense, and total_budget.
# ---------------------------------------------------------------------------

df = get_monthly_summary()


# ---------------------------------------------------------------------------
# Stop the page if no data is available
# ---------------------------------------------------------------------------
# This prevents the dashboard from crashing if the database query returns an empty DataFrame.
# ---------------------------------------------------------------------------

if df.empty:
    st.warning("No financial data is available. Please run the ETL pipeline first.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
# These filters allow the CFO or analyst to focus on a specific year or region.
#
# Year:
# - All
# - 2025
# - 2024
# - 2023
#
# Region:
# - All
# - East
# - West
# - Central
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    years = sorted(df["year"].unique(), reverse=True)
    year_options = ["All"] + [str(year) for year in years]

    # index=1 selects the latest year by default when available.
    # If only "All" exists, it safely falls back to index=0.
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


# ---------------------------------------------------------------------------
# Apply sidebar filters
# ---------------------------------------------------------------------------
# We create a copy of the DataFrame before filtering so the original df remains available for prior-year comparisons.
# ---------------------------------------------------------------------------

df_filtered = df.copy()

if selected_year != "All":
    df_filtered = df_filtered[df_filtered["year"] == int(selected_year)]

if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]


# ---------------------------------------------------------------------------
# Stop the page if filters return no rows
# ---------------------------------------------------------------------------
# This prevents charts from breaking when a user selects a filter combination that has no data.
# ---------------------------------------------------------------------------

if df_filtered.empty:
    st.warning("No data found for the selected filters.")
    st.stop()


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------
# These are top-level financial metrics shown at the top of the page.
#
# total_revenue:
# Total revenue for the selected filters.
#
# total_expense:
# Total expenses for the selected filters.
#
# total_budget:
# Total budget for the selected filters.
#
# net_profit:
# Revenue minus expenses.
#
# profit_margin:
# Net profit as a percentage of revenue.
#
# budget_variance_amt:
# Actual expenses minus budget.
#
# budget_variance_pct:
# Budget variance as a percentage of budget.
# ---------------------------------------------------------------------------

total_revenue = df_filtered["total_revenue"].sum()
total_expense = df_filtered["total_expense"].sum()
total_budget = df_filtered["total_budget"].sum()

net_profit = total_revenue - total_expense

profit_margin = (
    (net_profit / total_revenue) * 100
    if total_revenue > 0
    else 0
)

budget_variance_amt = total_expense - total_budget

budget_variance_pct = (
    (budget_variance_amt / total_budget) * 100
    if total_budget > 0
    else 0
)


# ---------------------------------------------------------------------------
# Prior-year comparison
# ---------------------------------------------------------------------------
# If the user selects a specific year, we compare revenue and expenses against the previous year.
#
# Example:
# selected_year = 2025
# prior_year = 2024
#
# If "All" is selected, YoY comparison does not make sense, so we keep it None.
# ---------------------------------------------------------------------------

if selected_year != "All":
    prior_year = int(selected_year) - 1

    df_prior = df[df["year"] == prior_year]

    if selected_region != "All":
        df_prior = df_prior[df_prior["region"] == selected_region]

    prior_revenue = df_prior["total_revenue"].sum()
    prior_expense = df_prior["total_expense"].sum()

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
# Row 1: KPI cards
# ---------------------------------------------------------------------------
# st.metric is a built-in Streamlit component for showing KPI values.
#
# delta_color="normal":
# Positive values show green, negative values show red.
#
# delta_color="inverse":
# Positive values show red, negative values show green.
# This is useful for expenses and budget variance because higher expense is bad.
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Revenue",
        value=format_currency(total_revenue),
        delta=format_yoy_delta(revenue_yoy),
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Total Expense",
        value=format_currency(total_expense),
        delta=format_yoy_delta(expense_yoy),
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Net Profit",
        value=format_currency(net_profit),
        delta=format_percentage(profit_margin),
        delta_color="normal"
    )

with col4:
    st.metric(
        label="Budget Variance",
        value=format_currency(budget_variance_amt),
        delta=f"{budget_variance_pct:+.1f}%",
        delta_color="inverse"
    )

st.divider()


# ---------------------------------------------------------------------------
# Row 2: Revenue by month and expense by department
# ---------------------------------------------------------------------------
# The left chart shows monthly revenue trends.
# The right chart shows which departments have the highest expenses.
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1.6, 1])


# ---------------------------------------------------------------------------
# Chart 1: Revenue by Month
# ---------------------------------------------------------------------------

with col_left:
    st.subheader("Revenue by Month")

    monthly_revenue = (
        df_filtered
        .groupby(["year", "month", "month_name"], as_index=False)["total_revenue"]
        .sum()
        .sort_values(["year", "month"])
    )

    # Convert year to string so Plotly treats it as a category, not a number.
    monthly_revenue["year"] = monthly_revenue["year"].astype(str)

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly_revenue["month_name"] = pd.Categorical(
        monthly_revenue["month_name"],
        categories=month_order,
        ordered=True
    )

    fig_revenue = px.line(
        monthly_revenue,
        x="month_name",
        y="total_revenue",
        color="year",
        markers=True,
        labels={
            "total_revenue": "Revenue",
            "month_name": "",
            "year": "Year"
        },
        color_discrete_sequence=["#1B3A6B", "#2E75B6", "#17A589"]
    )

    fig_revenue.update_traces(
        line=dict(width=2.5)
    )

    fig_revenue.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
        yaxis_tickformat="$,.0f",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_revenue, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 2: Expense by Department
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("Expense by Department")

    department_expense = (
        df_filtered
        .groupby("department", as_index=False)["total_expense"]
        .sum()
        .sort_values("total_expense", ascending=True)
    )

    department_colours = {
        "IT": "#C0392B",
        "Marketing": "#E67E22",
        "Operations": "#2E75B6",
        "Sales": "#1E8449",
        "HR": "#17A589"
    }

    bar_colours = [
        department_colours.get(department, "#2E75B6")
        for department in department_expense["department"]
    ]

    fig_department = go.Figure(
        go.Bar(
            x=department_expense["total_expense"],
            y=department_expense["department"],
            orientation="h",
            marker_color=bar_colours,
            text=[
                format_currency(value)
                for value in department_expense["total_expense"]
            ],
            textposition="outside"
        )
    )

    fig_department.update_layout(
    height=280,
    margin=dict(l=0, r=60, t=10, b=0),
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="white",
    paper_bgcolor="white"
)
    fig_department.update_xaxes(tickformat="$~s")
    st.plotly_chart(fig_department, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Row 3: Revenue vs Expense and Revenue by Region
# ---------------------------------------------------------------------------
# The left chart compares monthly expenses and revenue.
# The right chart shows revenue contribution by region.
# ---------------------------------------------------------------------------

col_left_2, col_right_2 = st.columns(2)


# ---------------------------------------------------------------------------
# Chart 3: Revenue vs Expense Monthly
# ---------------------------------------------------------------------------

with col_left_2:
    st.subheader("Revenue vs Expense Monthly")

    monthly_revenue_expense = (
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
            x=monthly_revenue_expense["month_name"],
            y=monthly_revenue_expense["total_expense"],
            name="Expense",
            marker_color="#1B3A6B"
        )
    )

    fig_combo.add_trace(
        go.Scatter(
            x=monthly_revenue_expense["month_name"],
            y=monthly_revenue_expense["total_revenue"],
            name="Revenue",
            mode="lines+markers",
            line=dict(color="#E67E22", width=2.5)
        )
    )

    fig_combo.update_layout(
    height=250,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="",
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title=""
)
    fig_combo.update_yaxes(tickformat="$~s")

    st.plotly_chart(fig_combo, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 4: Revenue by Region
# ---------------------------------------------------------------------------

with col_right_2:
    st.subheader("Revenue by Region")

    region_revenue = (
        df_filtered
        .groupby("region", as_index=False)["total_revenue"]
        .sum()
        .sort_values("total_revenue", ascending=True)
    )

    region_colours = {
        "East": "#1B3A6B",
        "West": "#2E75B6",
        "Central": "#17A589"
    }

    region_bar_colours = [
        region_colours.get(region, "#2E75B6")
        for region in region_revenue["region"]
    ]

    fig_region = go.Figure(
        go.Bar(
            x=region_revenue["total_revenue"],
            y=region_revenue["region"],
            orientation="h",
            marker_color=region_bar_colours,
            text=[
                format_currency(value)
                for value in region_revenue["total_revenue"]
            ],
            textposition="outside"
        )
    )

    fig_region.update_layout(
    height=250,
    margin=dict(l=0, r=60, t=10, b=0),
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="white",
    paper_bgcolor="white"
)
    fig_region.update_xaxes(tickformat="$~s")


    st.plotly_chart(fig_region, use_container_width=True)