"""

Transaction drill-through page for the CFO Financial Dashboard.

This page allows users to filter and explore individual expense transactions.
It is useful when a CFO or analyst wants to move from high-level KPIs into transaction-level detail.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Transaction Detail - CFO Dashboard",
    page_icon="mag_right",
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


from utils.db_queries import get_transactions
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
    '<div class="page-header">Transaction Detail - Drill Through</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Load transaction data
# ---------------------------------------------------------------------------
# This page focuses on expense transactions because the dashboard's budget, variance, and anomaly analysis are mainly based on expenses.
# ---------------------------------------------------------------------------

df = get_transactions()

if df.empty:
    st.warning("No transaction data is available. Please run the ETL pipeline first.")
    st.stop()

df_expense = df[df["transaction_type"] == "EXPENSE"].copy()

if df_expense.empty:
    st.warning("No expense transactions are available.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
# These filters let the user narrow down the transaction table by:
# - department
# - region
# - category
# - year
# - amount range
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    departments = ["All"] + sorted(df_expense["department"].dropna().unique().tolist())

    selected_department = st.selectbox(
        "Department",
        departments
    )

    regions = ["All"] + sorted(df_expense["region"].dropna().unique().tolist())

    selected_region = st.selectbox(
        "Region",
        regions
    )

    categories = ["All"] + sorted(df_expense["category"].dropna().unique().tolist())

    selected_category = st.selectbox(
        "Category",
        categories
    )

    years = sorted(df_expense["year"].dropna().unique().tolist(), reverse=True)

    selected_years = st.multiselect(
        "Year(s)",
        options=years,
        default=years
    )

    amount_min = float(df_expense["amount"].min())
    amount_max = float(df_expense["amount"].max())

    selected_amount_range = st.slider(
        "Amount range ($)",
        min_value=amount_min,
        max_value=amount_max,
        value=(amount_min, amount_max),
        format="$%.0f"
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

df_filtered = df_expense.copy()

if selected_department != "All":
    df_filtered = df_filtered[
        df_filtered["department"] == selected_department
    ]

if selected_region != "All":
    df_filtered = df_filtered[
        df_filtered["region"] == selected_region
    ]

if selected_category != "All":
    df_filtered = df_filtered[
        df_filtered["category"] == selected_category
    ]

if selected_years:
    df_filtered = df_filtered[
        df_filtered["year"].isin(selected_years)
    ]

df_filtered = df_filtered[
    (df_filtered["amount"] >= selected_amount_range[0])
    & (df_filtered["amount"] <= selected_amount_range[1])
]


# ---------------------------------------------------------------------------
# Stop if filters return no rows
# ---------------------------------------------------------------------------

if df_filtered.empty:
    st.warning("No transactions found for the selected filters.")
    st.stop()


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------
# total_spend:
# Total actual expense amount after filters.
#
# total_budget:
# Total budget amount after filters.
#
# variance:
# Actual spend minus budget.
#
# variance_pct:
# Variance as a percentage of budget.
#
# transaction_count:
# Number of transactions after filters.
#
# average_transaction:
# Average transaction amount.
# ---------------------------------------------------------------------------

total_spend = df_filtered["amount"].sum()
total_budget = df_filtered["budget_amount"].sum()

variance = total_spend - total_budget

variance_pct = (
    (variance / total_budget) * 100
    if total_budget > 0
    else 0
)

transaction_count = len(df_filtered)

average_transaction = (
    df_filtered["amount"].mean()
    if transaction_count > 0
    else 0
)


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Spend",
        value=format_currency(total_spend)
    )

with col2:
    st.metric(
        label="vs Budget",
        value=format_currency(variance),
        delta=f"{variance_pct:+.1f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Transactions",
        value=f"{transaction_count:,}"
    )

with col4:
    st.metric(
        label="Avg Transaction",
        value=format_currency(average_transaction)
    )

st.divider()


# ---------------------------------------------------------------------------
# Transaction table
# ---------------------------------------------------------------------------
# This is the main drill-through table.
# It shows detailed transaction-level information after filters are applied.
# ---------------------------------------------------------------------------

st.subheader(f"Transactions ({transaction_count:,} records)")

display_df = df_filtered[
    [
        "transaction_id",
        "date",
        "department",
        "region",
        "category",
        "vendor",
        "amount",
        "budget_amount",
        "is_anomaly"
    ]
].copy()

display_df["variance"] = (
    display_df["amount"] - display_df["budget_amount"]
)

display_df["date"] = pd.to_datetime(
    display_df["date"]
).dt.strftime("%Y-%m-%d")

display_df["amount"] = display_df["amount"].apply(format_currency)

display_df["budget_amount"] = display_df["budget_amount"].apply(format_currency)

display_df["variance"] = display_df["variance"].apply(
    lambda value: (
        f"+{format_currency(value)}"
        if value > 0
        else format_currency(value)
    )
)

display_df["is_anomaly"] = display_df["is_anomaly"].map(
    {
        True: "Yes",
        False: "No"
    }
)

display_df.columns = [
    "Transaction ID",
    "Date",
    "Department",
    "Region",
    "Category",
    "Vendor",
    "Amount",
    "Budget",
    "Anomaly",
    "Variance"
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=380,
    hide_index=True
)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
# The user can export the filtered raw transactions for further review.
# We export df_filtered, not display_df, because df_filtered keeps numeric values as numbers instead of formatted strings.
# ---------------------------------------------------------------------------

safe_department = selected_department.replace(" ", "_")
safe_years = "_".join([str(year) for year in selected_years]) if selected_years else "all"

csv = df_filtered.to_csv(index=False)

st.download_button(
    label="Export to CSV",
    data=csv,
    file_name=f"transactions_{safe_department}_{safe_years}.csv",
    mime="text/csv"
)

st.divider()


# ---------------------------------------------------------------------------
# Summary charts
# ---------------------------------------------------------------------------
# These charts summarize the filtered transaction set.
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)


# ---------------------------------------------------------------------------
# Chart 1: Spend by Category
# ---------------------------------------------------------------------------
# This chart shows the top expense categories after filters are applied.
# ---------------------------------------------------------------------------

with col_left:
    st.subheader("Spend by Category")

    category_spend = (
        df_filtered
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=True)
        .tail(8)
    )

    fig_category = go.Figure(
        go.Bar(
            x=category_spend["amount"],
            y=category_spend["category"],
            orientation="h",
            marker_color="#1B3A6B",
            text=[
                format_currency(value)
                for value in category_spend["amount"]
            ],
            textposition="outside"
        )
    )

    fig_category.update_layout(
        height=280,
        margin=dict(l=0, r=80, t=10, b=0),
        xaxis=dict(tickformat="$~s", title=""),
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_category, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 2: Monthly Spend Trend
# ---------------------------------------------------------------------------
# This chart shows how filtered spending changes over time.
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("Monthly Spend Trend")

    monthly_trend = (
        df_filtered
        .groupby(["year", "month"], as_index=False)["amount"]
        .sum()
        .sort_values(["year", "month"])
    )

    monthly_trend["date"] = pd.to_datetime(
        monthly_trend["year"].astype(str)
        + "-"
        + monthly_trend["month"].astype(str).str.zfill(2)
        + "-01"
    )

    fig_trend = go.Figure(
        go.Scatter(
            x=monthly_trend["date"],
            y=monthly_trend["amount"],
            mode="lines+markers",
            line=dict(color="#1B3A6B", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(27, 58, 107, 0.1)"
        )
    )

    fig_trend.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="",
        yaxis=dict(tickformat="$~s", title=""),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_trend, use_container_width=True)