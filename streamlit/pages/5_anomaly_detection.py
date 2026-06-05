"""
Anomaly Detection page for the CFO Financial Dashboard.

This page uses the Isolation Forest anomaly detection model to flag suspicious expense transactions for audit review.

The goal is to help finance and audit teams quickly identify unusual spending patterns, such as unusually high amounts, unusual vendors, or transactions that do not match normal department/category behavior.
"""

import os
import sys
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Anomaly Detection - CFO Dashboard",
    page_icon="mag",
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


from utils.db_queries import get_expense_transactions
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
    '<div class="page-header">Anomaly Detection - Isolation Forest</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
# score_threshold:
# Controls how strict the review table should be.
#
# show_known_only:
# Used when we want to compare predictions against known anomaly labels.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    score_threshold = st.slider(
        "Min anomaly score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values show only the most suspicious transactions."
    )

    show_known_only = st.checkbox(
        "Show known anomalies only",
        value=False,
        help="Show only transactions that are known anomalies in the dataset."
    )


# ---------------------------------------------------------------------------
# Run anomaly detection
# ---------------------------------------------------------------------------
# detect_anomalies() loads the saved Isolation Forest model and scaler, # engineers the same features used during training, scores the transactions, and returns the original DataFrame with:
#
# - anomaly_score
# - is_anomaly_predicted
# ---------------------------------------------------------------------------

with st.spinner("Running anomaly detection..."):
    try:
        from ml.anomaly.detect import detect_anomalies

        df_transactions = get_expense_transactions()

        if df_transactions.empty:
            st.warning("No expense transactions found. Please run the ETL pipeline first.")
            st.stop()

        df_scored = detect_anomalies(df_transactions)

        if df_scored.empty:
            st.warning("Anomaly detection returned no rows.")
            st.stop()


        # -------------------------------------------------------------------
        # Validate required columns
        # -------------------------------------------------------------------
        # This protects the dashboard from breaking if detect.py changes later.
        # -------------------------------------------------------------------

        required_columns = {
            "transaction_id",
            "date",
            "department",
            "category",
            "vendor",
            "amount",
            "is_anomaly",
            "anomaly_score",
            "is_anomaly_predicted",
            "year",
            "month"
        }

        missing_columns = required_columns - set(df_scored.columns)

        if missing_columns:
            st.error(
                f"Anomaly output is missing required columns: {missing_columns}"
            )
            st.stop()


        # -------------------------------------------------------------------
        # Summary KPI calculations
        # -------------------------------------------------------------------
        # total_transactions:
        # Total number of expense transactions scored by the model.
        #
        # total_flagged:
        # Number of transactions predicted as anomalous by the model.
        #
        # known_anomalies:
        # Number of true/known anomaly labels in the dataset.
        #
        # true_positive:
        # Transactions that were both predicted as anomalous and known anomalies.
        #
        # precision:
        # Of all transactions flagged by the model, how many were actually known anomalies?
        #
        # recall:
        # Of all known anomalies, how many did the model catch?
        # -------------------------------------------------------------------

        total_transactions = len(df_scored)
        total_flagged = int(df_scored["is_anomaly_predicted"].sum())
        known_anomalies = int(df_scored["is_anomaly"].sum())

        true_positive = int(
            (
                df_scored["is_anomaly_predicted"]
                & df_scored["is_anomaly"]
            ).sum()
        )

        precision = (
            (true_positive / total_flagged) * 100
            if total_flagged > 0
            else 0
        )

        recall = (
            (true_positive / known_anomalies) * 100
            if known_anomalies > 0
            else 0
        )


        # -------------------------------------------------------------------
        # KPI row
        # -------------------------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{total_transactions:,}"
            )

        with col2:
            st.metric(
                label="Flagged for Review",
                value=f"{total_flagged:,}"
            )

        with col3:
            st.metric(
                label="Known Anomalies",
                value=f"{known_anomalies:,}"
            )

        with col4:
            st.metric(
                label="Precision",
                value=f"{precision:.1f}%"
            )

        with col5:
            st.metric(
                label="Recall",
                value=f"{recall:.1f}%"
            )

        st.divider()


        # -------------------------------------------------------------------
        # Flagged transactions table
        # -------------------------------------------------------------------
        # This table shows the transactions predicted as suspicious.
        #
        # The user can control:
        # - minimum anomaly score
        # - whether to show only known anomalies
        # -------------------------------------------------------------------

        st.subheader("Flagged Transactions")

        df_flagged = df_scored[df_scored["is_anomaly_predicted"]].copy()

        if show_known_only:
            df_flagged = df_flagged[df_flagged["is_anomaly"]]

        df_flagged = (
            df_flagged[df_flagged["anomaly_score"] >= score_threshold]
            .sort_values("anomaly_score", ascending=False)
        )

        if df_flagged.empty:
            st.info("No flagged transactions match the selected filters.")
        else:
            display_df = df_flagged[
                [
                    "transaction_id",
                    "date",
                    "department",
                    "category",
                    "vendor",
                    "amount",
                    "anomaly_score",
                    "is_anomaly"
                ]
            ].copy()

            display_df["amount"] = display_df["amount"].apply(format_currency)

            display_df["anomaly_score"] = display_df["anomaly_score"].round(4)

            display_df["date"] = pd.to_datetime(
                display_df["date"]
            ).dt.strftime("%Y-%m-%d")

            display_df["is_anomaly"] = display_df["is_anomaly"].map(
                {
                    True: "Known",
                    False: "Detected"
                }
            )

            display_df.columns = [
                "Transaction ID",
                "Date",
                "Department",
                "Category",
                "Vendor",
                "Amount",
                "Score",
                "Ground Truth"
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                height=300,
                hide_index=True
            )

        st.caption(
            f"Showing {len(df_flagged)} flagged transactions "
            f"with anomaly score >= {score_threshold}"
        )

        st.divider()


        # -------------------------------------------------------------------
        # Charts row
        # -------------------------------------------------------------------
        # Left chart:
        # Shows how many transactions were flagged per department.
        #
        # Right chart:
        # Shows the distribution of anomaly scores.
        # -------------------------------------------------------------------

        col_left, col_right = st.columns(2)


        # -------------------------------------------------------------------
        # Chart 1: Anomalies by Department
        # -------------------------------------------------------------------

        with col_left:
            st.subheader("Anomalies by Department")

            department_anomaly = (
                df_scored
                .groupby("department", as_index=False)["is_anomaly_predicted"]
                .sum()
                .rename(columns={"is_anomaly_predicted": "flagged_count"})
                .sort_values("flagged_count", ascending=True)
            )

            department_colours = {
                "IT": "#2E75B6",
                "Marketing": "#C0392B",
                "Operations": "#2E75B6",
                "Sales": "#2E75B6",
                "HR": "#2E75B6"
            }

            bar_colours = [
                department_colours.get(department, "#2E75B6")
                for department in department_anomaly["department"]
            ]

            fig_department = go.Figure(
                go.Bar(
                    x=department_anomaly["flagged_count"],
                    y=department_anomaly["department"],
                    orientation="h",
                    marker_color=bar_colours,
                    text=department_anomaly["flagged_count"],
                    textposition="outside"
                )
            )

            fig_department.update_layout(
                height=250,
                margin=dict(l=0, r=40, t=10, b=0),
                xaxis_title="Flagged transactions",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_department, use_container_width=True)


        # -------------------------------------------------------------------
        # Chart 2: Anomaly Score Distribution
        # -------------------------------------------------------------------

        with col_right:
            st.subheader("Anomaly Score Distribution")

            fig_histogram = go.Figure(
                go.Histogram(
                    x=df_scored["anomaly_score"],
                    nbinsx=30,
                    marker_color="#2E75B6",
                    opacity=0.8
                )
            )

            # Draw the threshold line.
            fig_histogram.add_vline(
                x=score_threshold,
                line_dash="dash",
                line_color="#C0392B"
            )

            # Add the threshold label separately.
            # This avoids annotation-related Plotly issues.
            fig_histogram.add_annotation(
                x=score_threshold,
                y=1,
                xref="x",
                yref="paper",
                text=f"Threshold: {score_threshold}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=11, color="#C0392B")
            )

            fig_histogram.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_histogram, use_container_width=True)

        st.divider()


        # -------------------------------------------------------------------
        # Anomaly timeline
        # -------------------------------------------------------------------
        # This chart shows how many anomalies were flagged per month.
        # It helps identify months with unusual spikes in suspicious spending.
        # -------------------------------------------------------------------

        st.subheader("Anomaly Timeline")

        timeline = (
            df_scored[df_scored["is_anomaly_predicted"]]
            .groupby(["year", "month"])
            .size()
            .reset_index(name="count")
        )

        if timeline.empty:
            st.info("No predicted anomalies found for the timeline.")
        else:
            timeline["date"] = pd.to_datetime(
                timeline["year"].astype(str)
                + "-"
                + timeline["month"].astype(str).str.zfill(2)
                + "-01"
            )

            timeline = timeline.sort_values("date")

            fig_timeline = go.Figure(
                go.Bar(
                    x=timeline["date"],
                    y=timeline["count"],
                    marker_color="#C0392B",
                    opacity=0.8
                )
            )

            fig_timeline.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Flagged count",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig_timeline, use_container_width=True)


        # -------------------------------------------------------------------
        # Model information
        # -------------------------------------------------------------------
        # This section reads the saved model metadata from isolation_forest.pkl.
        # It displays the model name, precision, recall, and feature count.
        # -------------------------------------------------------------------

        st.divider()
        st.subheader("Model Information")

        model_path = os.path.join(
            project_root,
            "ml",
            "models",
            "isolation_forest.pkl"
        )

        if not os.path.exists(model_path):
            st.warning("Isolation Forest model file was not found.")
        else:
            with open(model_path, "rb") as model_file:
                model_data = pickle.load(model_file)

            metadata = model_data.get("metrics", {})

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    label="Model",
                    value="Isolation Forest"
                )

            with c2:
                st.metric(
                    label="Precision",
                    value=f"{metadata.get('precision', 0):.2%}"
                )

            with c3:
                st.metric(
                    label="Recall",
                    value=f"{metadata.get('recall', 0):.2%}"
                )

            with c4:
                st.metric(
                    label="Features",
                    value="8 engineered features"
                )


    except Exception as error:
        st.error(f"Anomaly detection error: {error}")
        st.exception(error)