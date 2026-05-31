"""
This file loads the saved XGBoost budget overrun classifier and predict the probability that each department will exceed its budget by the end of the current quarter.


Output per department:
    overrun_probability : number between 0.0 and 1.0
    risk_level          : HIGH / MEDIUM / LOW
    prediction          : Will Overrun / At Risk / Within Budget
"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import pickle

import numpy as np
import pandas as pd
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

MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")

# Risk thresholds.
# These thresholds convert model probability into business-friendly labels.
#
# Example:
#   probability >= 0.70 means HIGH risk
#   probability >= 0.40 and < 0.70 means MEDIUM risk
#   probability < 0.40 means LOW risk
#
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Step 1: Load saved model
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """
    Load the trained XGBoost model from disk.

    The model file contains a dictionary with:
        model:
            The trained XGBClassifier object.

        feature_columns:
            The exact feature columns used during training.

        trained_at:
            The timestamp when the model was trained.

        decision_threshold:
            The default classification threshold saved during training.

        metrics:
            Evaluation metrics from training, such as AUC, precision,
            recall, and F1 score.

    """

    # Make sure the model file exists before trying to load it.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run ml/classification/train_xgboost.py first."
        )

    # Open the pickle file and load the saved dictionary.
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Model trained at: {model_data['trained_at']}")
    print(f"AUC-ROC: {model_data['metrics']['auc_roc']:.4f}")
    print(f"CV AUC: {model_data['metrics']['cv_auc_mean']:.4f}")

    return model_data


# ---------------------------------------------------------------------------
# Step 2: Load financial data from PostgreSQL
# ---------------------------------------------------------------------------

def load_financial_data() -> pd.DataFrame:
    """
    Load monthly expense and budget data per department from PostgreSQL.

    We load historical data and not only the target month because Some features need history, such as:
        - dept_overrun_history_rate
        - prior_quarter_variance_pct
        - ytd_variance_pct

    If we only loaded one month, we could not calculate these features correctly.
    """

    query = """
        SELECT
            d.dept_name          AS department,
            dt.year              AS year,
            dt.month             AS month,
            dt.quarter           AS quarter,
            SUM(f.amount)        AS actual_expense,
            SUM(f.budget_amount) AS budget_amount
        FROM production.fact_financials f
        JOIN production.dim_department d
            ON f.dept_key = d.dept_key
        JOIN production.dim_date dt
            ON f.date_key = dt.date_key
        WHERE f.transaction_type = 'EXPENSE'
        GROUP BY
            d.dept_name,
            dt.year,
            dt.month,
            dt.quarter
        ORDER BY
            d.dept_name,
            dt.year,
            dt.month
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    print(f"Loaded {len(df):,} department-month rows from production")

    return df


# ---------------------------------------------------------------------------
# Step 3: Build quarterly summary
# ---------------------------------------------------------------------------

def build_quarterly_summary(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly financial data into quarterly financial data.

    This function is needed because some prediction features depend on
    quarter-level behavior.

    It calculates:
        quarterly_expense:
            Total expense for each department in each quarter.

        quarterly_budget:
            Total budget for each department in each quarter.

        overrun:
            1 if quarterly_expense > quarterly_budget, else 0.

        quarterly_variance_pct:
            Percentage difference between actual expense and budget.
    """

    quarterly_df = (
        monthly_df
        .groupby(["department", "year", "quarter"])
        .agg(
            quarterly_expense=("actual_expense", "sum"),
            quarterly_budget=("budget_amount", "sum"),
        )
        .reset_index()
    )

    # Historical overrun label.
    # It is used only to calculate historical features.
    quarterly_df["overrun"] = (
        quarterly_df["quarterly_expense"] > quarterly_df["quarterly_budget"]
    ).astype(int)

    # Calculate how far above or below budget each quarter was.
    quarterly_df["quarterly_variance_pct"] = (
        (quarterly_df["quarterly_expense"] - quarterly_df["quarterly_budget"])
        / quarterly_df["quarterly_budget"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    return quarterly_df


# ---------------------------------------------------------------------------
# Step 4: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineer the exact same 6 features used during training.
    """

    df = monthly_df.copy()

    
    df = df.sort_values(["department", "year", "month"])

    # -----------------------------------------------------------------------
    # Feature 1: month_of_quarter
    # -----------------------------------------------------------------------
    
    df["month_of_quarter"] = ((df["month"] - 1) % 3) + 1

    # -----------------------------------------------------------------------
    # Feature 2: spending_velocity
    # -----------------------------------------------------------------------

    df["spending_velocity"] = (
        df["actual_expense"] / df["budget_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    # -----------------------------------------------------------------------
    # Feature 3: ytd_variance_pct
    # -----------------------------------------------------------------------
    
    df["ytd_expense"] = (
        df
        .groupby(["department", "year"])["actual_expense"]
        .cumsum()
    )

    df["ytd_budget"] = (
        df
        .groupby(["department", "year"])["budget_amount"]
        .cumsum()
    )

    df["ytd_variance_pct"] = (
        (df["ytd_expense"] - df["ytd_budget"])
        / df["ytd_budget"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # -----------------------------------------------------------------------
    # Feature 4: dept_overrun_history_rate
    # -----------------------------------------------------------------------
 
    dept_history = (
        quarterly_df
        .groupby("department")["overrun"]
        .mean()
        .rename("dept_overrun_history_rate")
        .round(4)
    )

    df = df.merge(
        dept_history,
        on="department",
        how="left"
    )

    # -----------------------------------------------------------------------
    # Feature 5: budget_utilisation_rate
    # -----------------------------------------------------------------------
  
    df["quarter_to_date_expense"] = (
        df
        .groupby(["department", "year", "quarter"])["actual_expense"]
        .cumsum()
    )

    quarterly_budget_df = quarterly_df[
        ["department", "year", "quarter", "quarterly_budget"]
    ].copy()

    df = df.merge(
        quarterly_budget_df,
        on=["department", "year", "quarter"],
        how="left"
    )

    df["budget_utilisation_rate"] = (
        df["quarter_to_date_expense"]
        / df["quarterly_budget"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # -----------------------------------------------------------------------
    # Feature 6: prior_quarter_variance_pct
    # -----------------------------------------------------------------------
    
    prior_quarter_df = quarterly_df[
        ["department", "year", "quarter", "quarterly_variance_pct"]
    ].copy()


    prior_quarter_df = prior_quarter_df.rename(
        columns={
            "quarterly_variance_pct": "prior_quarter_variance_pct"
        }
    )

  
    prior_quarter_df["target_year"] = prior_quarter_df["year"]
    prior_quarter_df["target_quarter"] = prior_quarter_df["quarter"] + 1


    q4_mask = prior_quarter_df["target_quarter"] > 4

    prior_quarter_df.loc[q4_mask, "target_quarter"] = 1
    prior_quarter_df.loc[q4_mask, "target_year"] = (
        prior_quarter_df.loc[q4_mask, "year"] + 1
    )

    prior_quarter_df = prior_quarter_df[
        [
            "department",
            "target_year",
            "target_quarter",
            "prior_quarter_variance_pct",
        ]
    ].rename(
        columns={
            "target_year": "year",
            "target_quarter": "quarter",
        }
    )

    df = df.merge(
        prior_quarter_df,
        on=["department", "year", "quarter"],
        how="left"
    )


    df["prior_quarter_variance_pct"] = (
        df["prior_quarter_variance_pct"].fillna(0.0)
    )

    return df


# ---------------------------------------------------------------------------
# Step 5: Classify risk level
# ---------------------------------------------------------------------------

def classify_risk(probability: float) -> tuple[str, str]:
    """
    Convert an overrun probability into a business-friendly risk label.

    """

    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH", "Will Overrun"

    if probability >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM", "At Risk"

    return "LOW", "Within Budget"


# ---------------------------------------------------------------------------
# Step 6: Predict overrun probability
# ---------------------------------------------------------------------------

def predict_overrun(
    year: int,
    month: int
) -> pd.DataFrame:
    """
    Predict budget overrun probability for all departments for a selected month.

   
    """

    # Load the trained model and metadata.
    model_data = load_model()

    model = model_data["model"]
    feature_columns = model_data["feature_columns"]

    # Load historical financial data.
    monthly_df = load_financial_data()

    # Build quarterly summaries from the monthly data.
    quarterly_df = build_quarterly_summary(monthly_df)

    # Engineer the same features used during training.
    features_df = engineer_features(
        monthly_df=monthly_df,
        quarterly_df=quarterly_df
    )

    # Keep only the rows for the requested prediction period.
    prediction_rows = features_df[
        (features_df["year"] == year) &
        (features_df["month"] == month)
    ].copy()

    # If no rows exist, the requested period is not in the database.
    if prediction_rows.empty:
        raise ValueError(
            f"No data found for year={year}, month={month}. "
            "Check that this period exists in the production database."
        )

    # Prepare the feature matrix for prediction.
    # The columns must match the exact feature order used during training.
    X = prediction_rows[feature_columns].fillna(0)

   
    probabilities = model.predict_proba(X)[:, 1]

    results = prediction_rows[
        [
            "department",
            "year",
            "month",
            "spending_velocity",
            "ytd_variance_pct",
        ]
    ].copy()

  
    results["overrun_probability"] = probabilities.round(4)

    # Convert probability into risk labels.
    risk_labels = results["overrun_probability"].apply(classify_risk)

    results["risk_level"] = risk_labels.apply(lambda value: value[0])
    results["prediction"] = risk_labels.apply(lambda value: value[1])

    # Sort highest risk departments first.
    results = results.sort_values(
        by="overrun_probability",
        ascending=False
    ).reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Running Budget Overrun Predictions")
    print("=" * 60)

    # We predict for December 2025 because this is the last month
    # available in the synthetic dataset.
    TARGET_YEAR = 2025
    TARGET_MONTH = 11

    print(f"\nPredicting for {TARGET_YEAR}-{TARGET_MONTH:02d}")
    print(
        f"Risk thresholds: "
        f"HIGH >= {HIGH_RISK_THRESHOLD:.0%}, "
        f"MEDIUM >= {MEDIUM_RISK_THRESHOLD:.0%}, "
        f"LOW < {MEDIUM_RISK_THRESHOLD:.0%}"
    )

    results = predict_overrun(
        year=TARGET_YEAR,
        month=TARGET_MONTH
    )

    print("\nResults")
    print("-" * 60)

    print(
        results[
            [
                "department",
                "overrun_probability",
                "risk_level",
                "prediction",
                "spending_velocity",
                "ytd_variance_pct",
            ]
        ].to_string(index=False)
    )

    print("\nSummary")
    print("-" * 60)

    risk_counts = results["risk_level"].value_counts()

    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = risk_counts.get(level, 0)
        print(f"{level:<8}: {count} department(s)")