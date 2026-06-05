"""
Budget vs Actual variance analysis page.

This page helps the CFO understand whether actual spending is over or under budget.
It includes KPI cards, XGBoost budget overrun risk predictions, department-level variance charts, a monthly variance heatmap, and a waterfall bridge showing which departments contribute most to the total variance.
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
    page_title="Budget vs Actual - CFO Dashboard",
    page_icon="dart",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Python import paths
# ---------------------------------------------------------------------------


current_file = os.path.abspath(__file__)
pages_dir = os.path.dirname(current_file)
streamlit_dir = os.path.dirname(pages_dir)
project_root = os.path.dirname(streamlit_dir)

sys.path.insert(0, streamlit_dir)
sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Import reusable helper functions
# ---------------------------------------------------------------------------

from utils.db_queries import get_monthly_summary
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
    '<div class="page-header">Budget vs Actual - Variance Analysis</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# get_monthly_summary() returns monthly revenue, expense, and budget data by department and region.
# ---------------------------------------------------------------------------

df = get_monthly_summary()

if df.empty:
    st.warning("No budget data is available. Please run the ETL pipeline first.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
# The user can filter the variance analysis by:
# - year
# - quarter
# - region
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

    quarters = ["All", "Q1", "Q2", "Q3", "Q4"]

    selected_quarter = st.selectbox(
        "Quarter",
        quarters
    )

    regions = ["All"] + sorted(df["region"].dropna().unique().tolist())

    selected_region = st.selectbox(
        "Region",
        regions
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

df_filtered = df.copy()

if selected_year != "All":
    df_filtered = df_filtered[df_filtered["year"] == int(selected_year)]

if selected_quarter != "All":
    df_filtered = df_filtered[df_filtered["quarter_name"] == selected_quarter]

if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]

if df_filtered.empty:
    st.warning("No data found for the selected filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Main variance calculations
# ---------------------------------------------------------------------------
# total_budget:
# Planned spending.
#
# total_expense:
# Actual spending.
#
# variance_amt:
# Actual spending minus planned budget.
# Positive means over budget.
# Negative means under budget.
#
# variance_pct:
# Variance amount as a percentage of total budget.
# ---------------------------------------------------------------------------

total_budget = df_filtered["total_budget"].sum()
total_expense = df_filtered["total_expense"].sum()

variance_amt = total_expense - total_budget

variance_pct = (
    (variance_amt / total_budget) * 100
    if total_budget > 0
    else 0
)


# ---------------------------------------------------------------------------
# Department-level summary
# ---------------------------------------------------------------------------
# This table powers the department charts and waterfall chart.
# ---------------------------------------------------------------------------

dept_summary = (
    df_filtered
    .groupby("department", as_index=False)
    .agg(
        total_expense=("total_expense", "sum"),
        total_budget=("total_budget", "sum")
    )
)

dept_summary["variance_amt"] = (
    dept_summary["total_expense"] - dept_summary["total_budget"]
)

dept_summary["variance_pct"] = (
    (dept_summary["variance_amt"] / dept_summary["total_budget"]) * 100
).round(2)

dept_summary = dept_summary.sort_values("variance_pct", ascending=False)


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Budget",
        value=format_currency(total_budget)
    )

with col2:
    st.metric(
        label="Total Actual",
        value=format_currency(total_expense)
    )

with col3:
    st.metric(
        label="Variance $",
        value=format_currency(variance_amt),
        delta=f"{variance_pct:+.1f}%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Variance %",
        value=f"{variance_pct:+.1f}%",
        delta="vs budget",
        delta_color="off"
    )

st.divider()


# ---------------------------------------------------------------------------
# XGBoost budget overrun predictions
# ---------------------------------------------------------------------------
# This section runs the trained XGBoost classifier and shows the predicted overrun probability by department.
#
# If the model or prediction function is not available, the dashboard still works and shows a warning instead of crashing.
# ---------------------------------------------------------------------------

st.subheader("Budget Overrun Risk - XGBoost Prediction")

with st.spinner("Running budget overrun predictions..."):
    try:
        from ml.classification.predict_overrun import predict_overrun

        year_for_prediction = (
            int(selected_year)
            if selected_year != "All"
            else int(df["year"].max())
        )

        overrun_df = predict_overrun(year_for_prediction, 12)

        if overrun_df.empty:
            st.info("No overrun predictions returned.")
        else:
            risk_columns = st.columns(5)

            for index, row in overrun_df.iterrows():
                with risk_columns[index % 5]:
                    probability = row["overrun_probability"]
                    risk_level = row["risk_level"]
                    department = row["department"]

                    badge_class = (
                        "badge-high"
                        if risk_level == "HIGH"
                        else "badge-medium"
                        if risk_level == "MEDIUM"
                        else "badge-low"
                    )

                    risk_colour = (
                        "#C0392B"
                        if risk_level == "HIGH"
                        else "#E67E22"
                        if risk_level == "MEDIUM"
                        else "#1E8449"
                    )

                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:12px; background:#F8F9FA;
                             border-radius:8px; border-left:4px solid {risk_colour}">
                            <div style="font-size:12px;color:#7F8C8D;margin-bottom:4px;">
                                {department}
                            </div>
                            <div style="font-size:24px;font-weight:600;color:{risk_colour}">
                                {probability:.0%}
                            </div>
                            <span class="badge {badge_class}">{risk_level}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    except Exception as error:
        st.warning(f"Overrun predictions unavailable: {error}")

st.divider()


# ---------------------------------------------------------------------------
# Budget vs Actual charts
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1.3, 1])


# ---------------------------------------------------------------------------
# Chart 1: Actual Spend by Department
# ---------------------------------------------------------------------------
# This chart shows which departments are spending the most.
# The dashed vertical line represents average budget per department.
# ---------------------------------------------------------------------------

with col_left:
    st.subheader("Actual Spend by Department")

    spend_data = dept_summary.sort_values("total_expense")

    department_colours = {
        "IT": "#C0392B",
        "Marketing": "#E67E22",
        "Operations": "#2E75B6",
        "Sales": "#1E8449",
        "HR": "#17A589"
    }

    spend_colours = [
        department_colours.get(department, "#2E75B6")
        for department in spend_data["department"]
    ]

    fig_spend = go.Figure(
        go.Bar(
            x=spend_data["total_expense"],
            y=spend_data["department"],
            orientation="h",
            marker_color=spend_colours,
            text=[
                format_currency(value)
                for value in spend_data["total_expense"]
            ],
            textposition="outside"
        )
    )

    fig_spend.add_vline(
        x=total_budget / max(len(spend_data), 1),
        line_dash="dash",
        line_color="#1B3A6B",
        annotation_text="Avg Budget",
        annotation_position="top right"
    )

    fig_spend.update_layout(
        height=280,
        margin=dict(l=0, r=80, t=10, b=0),
        xaxis=dict(tickformat="$~s", title=""),
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_spend, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 2: Budget Variance % by Department
# ---------------------------------------------------------------------------
# This chart shows each department's variance percentage.
#
# Color logic:
# Red    = more than 10% over budget
# Orange = more than 5% over budget
# Green  = acceptable or under budget
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("Budget Variance % by Department")

    variance_data = dept_summary.sort_values("variance_pct")

    variance_colours = [
        "#C0392B"
        if value > 10
        else "#E67E22"
        if value > 5
        else "#1E8449"
        for value in variance_data["variance_pct"]
    ]

    fig_variance = go.Figure(
        go.Bar(
            x=variance_data["variance_pct"],
            y=variance_data["department"],
            orientation="h",
            marker_color=variance_colours,
            text=[
                f"{value:+.1f}%"
                for value in variance_data["variance_pct"]
            ],
            textposition="outside"
        )
    )

    fig_variance.add_vline(
        x=0,
        line_color="#888888",
        line_dash="dash"
    )

    fig_variance.update_layout(
        height=280,
        margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(ticksuffix="%", title=""),
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_variance, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Variance heatmap
# ---------------------------------------------------------------------------
# This heatmap shows department variance by month.
#
# Rows:
# Departments
#
# Columns:
# Months
#
# Values:
# Budget variance percentage
# ---------------------------------------------------------------------------

st.subheader("Variance Heatmap - Department x Month")

heatmap_data = (
    df_filtered
    .groupby(["department", "month", "month_name"], as_index=False)
    .agg(
        total_expense=("total_expense", "sum"),
        total_budget=("total_budget", "sum")
    )
)

heatmap_data["variance_pct"] = (
    (
        heatmap_data["total_expense"] - heatmap_data["total_budget"]
    )
    / heatmap_data["total_budget"]
    * 100
).round(1)

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

heatmap_pivot = (
    heatmap_data
    .pivot(index="department", columns="month_name", values="variance_pct")
    .reindex(columns=month_order)
)

fig_heatmap = go.Figure(
    go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale=[
            [0.0, "#1E8449"],
            [0.35, "#EAF5EE"],
            [0.50, "#FEF5E7"],
            [0.65, "#E67E22"],
            [1.0, "#C0392B"]
        ],
        text=heatmap_pivot.values.round(1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        showscale=True,
        colorbar=dict(
            title="Variance %",
            ticksuffix="%",
            len=0.8
        ),
        zmid=0
    )
)

fig_heatmap.update_layout(
    height=220,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------
# This chart explains how each department contributes to the total budget variance amount.
#
# Positive variance means the department is over budget.
# Negative variance means the department is under budget.
# ---------------------------------------------------------------------------

st.subheader("Budget Variance Bridge - by Department")

waterfall_data = dept_summary.sort_values("variance_amt", ascending=False)

fig_waterfall = go.Figure(
    go.Waterfall(
        name="Variance",
        orientation="v",
        measure=["relative"] * len(waterfall_data) + ["total"],
        x=waterfall_data["department"].tolist() + ["Total"],
        y=waterfall_data["variance_amt"].tolist() + [variance_amt],
        text=[
            format_currency(value)
            for value in waterfall_data["variance_amt"].tolist() + [variance_amt]
        ],
        textposition="outside",
        increasing=dict(marker_color="#C0392B"),
        decreasing=dict(marker_color="#1E8449"),
        totals=dict(marker_color="#1B3A6B"),
        connector=dict(line=dict(color="#CCCCCC", width=1))
    )
)

fig_waterfall.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=10, b=40),
    yaxis=dict(tickformat="$~s", title=""),
    xaxis_title="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False
)

st.plotly_chart(fig_waterfall, use_container_width=True)