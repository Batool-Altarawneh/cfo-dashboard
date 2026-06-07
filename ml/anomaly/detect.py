"""
Here, I load the saved Isolation Forest model and use it to detect anomalous expense transactions.

This file is used by the Streamlit dashboard at runtime.
It does not retrain the model, it only loads and runs predictions.

Important design decision:
Feature statistics like vendor averages must be calculated on the full historical dataset, not just the new batch being scored.

If we only used the new batch, the vendor average would shift every run and anomaly scores would be inconsistent.

So this file always loads all expense transactions from PostgreSQL to calculate consistent feature statistics, then returns scores for the transactions passed into detect_anomalies().
"""


import os
import sys
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, project_root)

from etl.extract.db import engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(project_root, "ml", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")


# ---------------------------------------------------------------------------
# Load saved model
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """
    Load the saved Isolation Forest model from disk.

    Returns the full model_data dictionary containing:
        model:           the trained IsolationForest object
        scaler:          the fitted StandardScaler
        feature_columns: list of feature names used during training
        trained_at:      ISO timestamp of when training occurred
        contamination:   contamination value used during training
        metrics:         precision, recall, true_positives from evaluation
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_isolation_forest.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    print(f"  Model loaded: trained at {model_data['trained_at']}")
    print(
        f"  Precision: {model_data['metrics']['precision']:.2%}  "
        f"Recall: {model_data['metrics']['recall']:.2%}"
    )

    return model_data


# ---------------------------------------------------------------------------
# Load expense transactions from PostgreSQL
# ---------------------------------------------------------------------------

def load_expense_transactions() -> pd.DataFrame:
    """
    Load all expense transactions from the production database.

    I load all transactions and not just new ones because feature statistics like vendor_avg_amount must be calculated on the full historical dataset to stay consistent with training.

    If we only loaded new transactions, the vendor average would shift every run and produce inconsistent anomaly scores.
    """

    query = """
        SELECT
            f.fact_key,
            f.transaction_id,
            f.amount,
            f.budget_amount,
            f.vendor,
            f.is_anomaly,

            d.dept_name     AS department,
            r.region_name   AS region,
            c.category_name AS category,

            dt.full_date    AS transaction_date,
            dt.year,
            dt.month

        FROM production.fact_financials f

        JOIN production.dim_department d
            ON f.dept_key = d.dept_key

        JOIN production.dim_region r
            ON f.region_key = r.region_key

        JOIN production.dim_category c
            ON f.category_key = c.category_key

        JOIN production.dim_date dt
            ON f.date_key = dt.date_key

        WHERE f.transaction_type = 'EXPENSE'

        ORDER BY dt.full_date, f.transaction_id
    """

    raw_conn = engine.raw_connection()

    try:
      df = pd.read_sql_query(query, raw_conn)
    finally:
        raw_conn.close()

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    print(f"  Loaded {len(df):,} expense transactions")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the exact same 8 features used during training.

    """

    df = df.copy()

    # Feature 1: amount_zscore
    dept_stats = (
        df.groupby("department")["amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "dept_mean", "std": "dept_std"})
    )

    df = df.merge(dept_stats, on="department", how="left")

    df["dept_std"] = df["dept_std"].fillna(1.0)

    df["amount_zscore"] = (
        (df["amount"] - df["dept_mean"]) / df["dept_std"]
    ).round(4)

    # Feature 2: vendor_frequency
    vendor_frequency = (
        df.groupby("vendor")["transaction_id"]
        .count()
        .rename("vendor_frequency")
    )

    df = df.merge(vendor_frequency, on="vendor", how="left")

    # Feature 3: vendor_dept_frequency
    vendor_dept_frequency = (
        df.groupby(["vendor", "department"])["transaction_id"]
        .count()
        .rename("vendor_dept_frequency")
    )

    df = df.merge(
        vendor_dept_frequency,
        on=["vendor", "department"],
        how="left"
    )

    # Feature 4: amount_vs_vendor_avg
    vendor_average = (
        df.groupby("vendor")["amount"]
        .mean()
        .rename("vendor_avg_amount")
    )

    df = df.merge(vendor_average, on="vendor", how="left")

    df["amount_vs_vendor_avg"] = (
        df["amount"] / df["vendor_avg_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    # Feature 5: amount_vs_dept_category_avg
    dept_category_average = (
        df.groupby(["department", "category"])["amount"]
        .mean()
        .rename("dept_category_avg_amount")
    )

    df = df.merge(
        dept_category_average,
        on=["department", "category"],
        how="left"
    )

    df["amount_vs_dept_category_avg"] = (
        df["amount"] / df["dept_category_avg_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    # Feature 6: amount_vs_category_global_avg
    category_global_average = (
        df.groupby("category")["amount"]
        .mean()
        .rename("category_global_avg_amount")
    )

    df = df.merge(
        category_global_average,
        on="category",
        how="left"
    )

    df["amount_vs_category_global_avg"] = (
        df["amount"] / df["category_global_avg_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    # Feature 7: dept_monthly_spend_zscore
    monthly_spend = (
        df.groupby(["department", "year", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "monthly_total"})
    )

    monthly_dept_stats = (
        monthly_spend
        .groupby("department")["monthly_total"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "monthly_mean", "std": "monthly_std"})
    )

    monthly_spend = monthly_spend.merge(
        monthly_dept_stats,
        on="department",
        how="left"
    )

    monthly_spend["monthly_std"] = monthly_spend["monthly_std"].fillna(1.0)

    monthly_spend["dept_monthly_spend_zscore"] = (
        (monthly_spend["monthly_total"] - monthly_spend["monthly_mean"])
        / monthly_spend["monthly_std"]
    ).round(4)

    df = df.merge(
        monthly_spend[
            [
                "department",
                "year",
                "month",
                "dept_monthly_spend_zscore"
            ]
        ],
        on=["department", "year", "month"],
        how="left"
    )

    # Feature 8: is_round_number
    df["is_round_number"] = (
        (df["amount"] % 1000 == 0).astype(int)
    )

    return df


# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------

def detect_anomalies(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score transactions using the saved Isolation Forest model.
    """

    
    transactions_to_score = transactions_df.copy()

    # Load model, scaler, and feature column list
    model_data = load_model()
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_columns = model_data["feature_columns"]

    # Load full historical expense transactions from PostgreSQL.
    # This is critical because statistics like vendor averages must be calculated on the same population used in training.
    historical_df = load_expense_transactions()

    # Engineer features on the full historical dataset.
    historical_features_df = engineer_features(historical_df)

    # Prepare feature matrix using the exact feature list saved with the model.
    X = historical_features_df[feature_columns].fillna(0)

    X_scaled = scaler.transform(X)

    # Get predictions: -1 = anomaly, 1 = normal
    raw_predictions = model.predict(X_scaled)

    # Get anomaly scores : higher means more suspicious
    anomaly_scores = -model.score_samples(X_scaled)

    # Add predictions to the historical DataFrame
    scored_history = historical_features_df.copy()
    scored_history["anomaly_score"] = anomaly_scores.round(4)
    scored_history["is_anomaly_predicted"] = (raw_predictions == -1)

    # Merge the scores back to the original input rows.
    # fact_key is the safest key because it is the unique production table key.
    if "fact_key" in transactions_to_score.columns:
        result_df = transactions_to_score.merge(
            scored_history[
                [
                    "fact_key",
                    "anomaly_score",
                    "is_anomaly_predicted"
                ]
            ],
            on="fact_key",
            how="left"
        )

    elif "transaction_id" in transactions_to_score.columns:
        result_df = transactions_to_score.merge(
            scored_history[
                [
                    "transaction_id",
                    "anomaly_score",
                    "is_anomaly_predicted"
                ]
            ],
            on="transaction_id",
            how="left"
        )

    else:
        raise KeyError(
            "transactions_df must contain either 'fact_key' or 'transaction_id' "
            "so anomaly scores can be merged back to the original rows."
        )

    # If a row was not found in historical data, it will not have a score.
    # For safety, keep anomaly_score as float and default missing predictions to False.
    result_df["anomaly_score"] = result_df["anomaly_score"].astype(float)
    result_df["is_anomaly_predicted"] = (
        result_df["is_anomaly_predicted"]
        .fillna(False)
        .astype(bool)
    )

    print(f"  Transactions requested: {len(transactions_to_score):,}")
    print(f"  Transactions returned:  {len(result_df):,}")
    print(f"  Flagged as anomalous:   {result_df['is_anomaly_predicted'].sum()}")

    return result_df


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  Running Anomaly Detection")
    print("=" * 60)

    # Load transactions from PostgreSQL
    print("\n── Loading Transactions ──")
    df = load_expense_transactions()

    # Run detection
    print("\n── Scoring Transactions ──")
    df = detect_anomalies(df)

    # Show top 10 most suspicious transactions
    print("\n── Top 10 Most Suspicious Transactions ──")

    display_columns = [
        "transaction_id",
        "transaction_date",
        "department",
        "region",
        "category",
        "vendor",
        "amount",
        "anomaly_score",
        "is_anomaly",
        "is_anomaly_predicted"
    ]

    available_columns = [
        col for col in display_columns
        if col in df.columns
    ]

    top_10 = (
        df
        .sort_values("anomaly_score", ascending=False)
        .head(10)
    )

    print(top_10[available_columns].to_string(index=False))

    # Verification against known anomalies
    print("\n── Verification ──")

    if "is_anomaly" in df.columns:
        known_anomalies = df["is_anomaly"].astype(bool)
        predicted_anomalies = df["is_anomaly_predicted"].astype(bool)

        correctly_flagged = (
            predicted_anomalies & known_anomalies
        ).sum()

        total_known = known_anomalies.sum()
        total_flagged = predicted_anomalies.sum()

        if total_known > 0:
            recall = correctly_flagged / total_known
        else:
            recall = 0

        print(f"  Known anomalies:          {total_known}")
        print(f"  Flagged by model:         {total_flagged}")
        print(f"  Correctly flagged (TP):   {correctly_flagged}")
        print(f"  Recall:                   {recall:.2%}")

    else:
        print("  is_anomaly column not found, so known anomaly verification was skipped.")