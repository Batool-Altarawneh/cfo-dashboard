"""
Train an Isolation Forest model to detect suspicious or anomalous financial expense transactions.

This model is unsupervised, which means it does not need labelled data like fraud = 1 or normal = 0 during training.

Instead, it learns the normal transaction patterns and flags transactions that look statistically different from the majority.

"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import pickle
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from sqlalchemy import text
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score


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


CONTAMINATION = 0.005

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load expense transaction data
# ---------------------------------------------------------------------------

def load_expense_transactions() -> pd.DataFrame:
    """
    Load all expense transactions from the production database.

    """

    # This SQL query reads expense transactions from the fact table and joins dimension tables so we get readable names instead of keys.
  
    # Instead of dept_key = 1, we get department = "Marketing".
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

    # Open a database connection and run the query.
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    # Convert transaction_date to datetime so pandas can handle dates properly.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Print basic information so we know the data loaded correctly.
    print(f"  Loaded {len(df):,} expense transactions from production")
    print(f"  Known anomalies in data: {df['is_anomaly'].sum()}")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that help the model understand what suspicious means.
    """

  
    df = df.copy()

    # -----------------------------------------------------------------------
    # Feature 1: amount_zscore
    # -----------------------------------------------------------------------
   

    dept_stats = (
        df.groupby("department")["amount"]
        .agg(["mean", "std"])
        .rename(columns={
            "mean": "dept_mean",
            "std": "dept_std"
        })
    )

    # Add the department average and standard deviation back to each row.
    df = df.merge(dept_stats, on="department", how="left")

    # If a department has only one transaction, standard deviation may be null.
    # We replace null with 1.0 to avoid division by zero or NaN values.
    df["dept_std"] = df["dept_std"].fillna(1.0)

    # Calculate the z-score for each transaction.
    df["amount_zscore"] = (
        (df["amount"] - df["dept_mean"]) / df["dept_std"]
    ).round(4)

    # -----------------------------------------------------------------------
    # Feature 2: vendor_frequency
    # -----------------------------------------------------------------------
   

    vendor_frequency = (
        df.groupby("vendor")["transaction_id"]
        .count()
        .rename("vendor_frequency")
    )

    # Add vendor frequency to every transaction.
    df = df.merge(vendor_frequency, on="vendor", how="left")

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

    # -----------------------------------------------------------------------
    # Feature 3: amount_vs_vendor_avg
    # -----------------------------------------------------------------------
  

    vendor_average = (
        df.groupby("vendor")["amount"]
        .mean()
        .rename("vendor_avg_amount")
    )

    # Add the average amount for each vendor to every transaction.
    df = df.merge(vendor_average, on="vendor", how="left")

    # Calculate how much bigger or smaller the current transaction is compared to the vendor's usual average amount.
    #
    # replace(0, np.nan) protects us from division by zero.
    # fillna(1.0) means if we cannot calculate the ratio, we treat it as normal.
    df["amount_vs_vendor_avg"] = (
        df["amount"] / df["vendor_avg_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)



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

    # -----------------------------------------------------------------------
    # Feature 4: dept_monthly_spend_zscore
    # -----------------------------------------------------------------------
   

    monthly_spend = (
        df.groupby(["department", "year", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "monthly_total"})
    )

    # For each department, calculate its average monthly spend and monthly spending standard deviation.
    monthly_dept_stats = (
        monthly_spend
        .groupby("department")["monthly_total"]
        .agg(["mean", "std"])
        .rename(columns={
            "mean": "monthly_mean",
            "std": "monthly_std"
        })
    )

    # Add monthly average and monthly standard deviation to monthly_spend.
    monthly_spend = monthly_spend.merge(
        monthly_dept_stats,
        on="department",
        how="left"
    )

    # Avoid null standard deviation.
    monthly_spend["monthly_std"] = monthly_spend["monthly_std"].fillna(1.0)

    # Calculate the monthly spending z-score.
    monthly_spend["dept_monthly_spend_zscore"] = (
        (monthly_spend["monthly_total"] - monthly_spend["monthly_mean"])
        / monthly_spend["monthly_std"]
    ).round(4)

    # Add the department-month z-score back to each transaction.
    # Every transaction in the same department/month will get the same monthly spending z-score.
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

    # -----------------------------------------------------------------------
    # Feature 5: is_round_number
    # -----------------------------------------------------------------------
   
    df["is_round_number"] = (
        (df["amount"] % 1000 == 0).astype(int)
    )

    # Print confirmation that feature engineering worked.
    print(f"  Features engineered for {len(df):,} transactions")
    print(
        "  Feature columns created: "
        "amount_zscore, vendor_frequency, amount_vs_vendor_avg, " "amount_vs_dept_category_avg, "
        "dept_monthly_spend_zscore, is_round_number""vendor_dept_frequency, amount_vs_category_global_avg"
    )
    

    return df


# ---------------------------------------------------------------------------
# Train Isolation Forest model
# ---------------------------------------------------------------------------

def train_isolation_forest(df: pd.DataFrame) -> tuple:
    """
    Train an Isolation Forest model using the engineered features.

    Parameters:
        df:
            DataFrame that already contains the engineered feature columns.

    Returns:
        tuple:
            model:
                The trained Isolation Forest model.

            scaler:
                The fitted StandardScaler.

            feature_columns:
                List of features used for training.

    we scale the features because the features have different ranges.

    Example:
        amount_zscore might be between -3 and 10.
        vendor_frequency might be between 1 and 200.

    Scaling puts all features on a similar scale so one large-range feature does not dominate the model.
    """

    # These are the exact columns the model will learn from.
    feature_columns = [
        "amount_zscore",
        "vendor_frequency",
        "vendor_dept_frequency",
        "amount_vs_vendor_avg",
        "amount_vs_dept_category_avg",
        "amount_vs_category_global_avg",
        "dept_monthly_spend_zscore",
        "is_round_number"
    ]

    # Select only the feature columns.
    # fillna(0) makes sure the model does not receive missing values.
    X = df[feature_columns].fillna(0)

    # Create the scaler.
    scaler = StandardScaler()

    # Fit the scaler on our training data and transform the data.
    X_scaled = scaler.fit_transform(X)

    # Create the Isolation Forest model.
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        max_samples="auto"
    )

    # Train the model.
    # Since Isolation Forest is unsupervised, we only pass X_scaled.
    # We do not pass y labels.
    model.fit(X_scaled)

    print(f"  Isolation Forest trained on {len(X):,} transactions")
    print(f"  Features used: {feature_columns}")
    print(f"  Contamination: {CONTAMINATION}")

    return model, scaler, feature_columns


# ---------------------------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    scaler,
    feature_columns: list,
    df: pd.DataFrame
) -> dict:
    """
    Evaluate the trained model using the known injected anomalies.

    Isolation Forest prediction output:
        -1 means anomaly
         1 means normal

    We convert it to:
         1 means anomaly
         0 means normal

    Then we compare the prediction with the is_anomaly column.

    Metrics:
        Precision:
            Of the transactions we flagged, how many were truly anomalies?

        Recall:
            Of the true anomalies, how many did we successfully catch?

    In anomaly detection, recall is very important because missing a real suspicious transaction can be costly.
    """

    # Prepare the same feature data used during training.
    X = df[feature_columns].fillna(0)

 
    X_scaled = scaler.transform(X)

    # Get model predictions.
    # sklearn returns:
    # -1 for anomaly
    #  1 for normal
    raw_predictions = model.predict(X_scaled)

    # Convert predictions into a clearer format:
    # 1 = anomaly
    # 0 = normal
    predicted_anomaly = (raw_predictions == -1).astype(int)

    # Get anomaly scores.
    #
    # In sklearn, lower score means more anomalous.
    # We multiply by -1 so higher score means more suspicious, which is easier to understand in a dashboard.
    anomaly_scores = -model.score_samples(X_scaled)

    # This is the real label from our synthetic data.
    # We use it only for evaluation, not for training.
    actual_anomaly = df["is_anomaly"].astype(int)

    # Calculate precision and recall.
    precision = precision_score(
        actual_anomaly,
        predicted_anomaly,
        zero_division=0
    )

    recall = recall_score(
        actual_anomaly,
        predicted_anomaly,
        zero_division=0
    )

    # True positives:
    # Transactions that are actually anomalies AND predicted as anomalies.
    true_positives = (predicted_anomaly & actual_anomaly).sum()

    print("\n── Model Evaluation ──")
    print(f"  Known anomalies in data:    {actual_anomaly.sum()}")
    print(f"  Transactions flagged:       {predicted_anomaly.sum()}")
    print(f"  Correctly flagged (TP):     {true_positives}")
    print(f"  Precision:                  {precision:.2%}")
    print(f"  Recall:                     {recall:.2%}")

    # Simple quality message to help us quickly understand the result.
    if precision >= 0.5 and recall >= 0.5:
        quality = "Good"
    elif precision >= 0.3 or recall >= 0.3:
        quality = "Review"
    else:
        quality = "Poor — consider tuning contamination or features"

    print(f"  Quality:                    {quality}")

    # Return metrics as a dictionary so we can save them with the model.
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "flagged_count": int(predicted_anomaly.sum()),
        "known_anomalies": int(actual_anomaly.sum()),
        "true_positives": int(true_positives),
    }


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_model(
    model,
    scaler,
    feature_columns: list,
    metrics: dict
) -> str:
  

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "contamination": CONTAMINATION,
        "metrics": metrics,
    }

    # Full path of the model file.
    file_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")

    # Save the dictionary as a pickle file.
    with open(file_path, "wb") as file:
        pickle.dump(model_data, file)

    print(f"\n  Saved model file: {file_path}")

    return file_path


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_anomaly_model() -> dict:
    """
    Run the full anomaly detection training pipeline.

    Pipeline steps:
        1. Load expense transactions from PostgreSQL.
        2. Engineer anomaly detection features.
        3. Train Isolation Forest.
        4. Evaluate the model using known injected anomalies.
        5. Save the model and metadata.

    Returns:
        dict:
            Evaluation metrics such as precision, recall, flagged_count, and true_positives.
    """

    print("=" * 60)
    print("  Training Isolation Forest Anomaly Detection Model")
    print("=" * 60)

    print("\n── Step 1: Loading Data ──")
    df = load_expense_transactions()

    print("\n── Step 2: Engineering Features ──")
    df = engineer_features(df)

    print("\n── Step 3: Training Model ──")
    model, scaler, feature_columns = train_isolation_forest(df)

    print("\n── Step 4: Evaluating Model ──")
    metrics = evaluate_model(model, scaler, feature_columns, df)

    print("\n── Step 5: Saving Model ──")
    save_model(model, scaler, feature_columns, metrics)

    return metrics


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    metrics = train_anomaly_model()

    print("\n── Final Metrics ──")
    print(f"  Precision      : {metrics['precision']:.2%}")
    print(f"  Recall         : {metrics['recall']:.2%}")
    print(f"  Flagged        : {metrics['flagged_count']} transactions")
    print(
        f"  True Positives : {metrics['true_positives']} "
        f"of {metrics['known_anomalies']} known anomalies caught"
    )