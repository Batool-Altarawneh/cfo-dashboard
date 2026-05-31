"""
Train an XGBoost classifier to predict whether a department will exceed its budget by the end of the current quarter.

This is a supervised binary classification problem.

Target variable:
    overrun = 1
        The department exceeded its quarterly budget.

    overrun = 0
        The department stayed within or equal to its quarterly budget.
"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sqlalchemy import text

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import xgboost as xgb


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


RANDOM_STATE = 42

DECISION_THRESHOLD = 0.5

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load data from PostgreSQL
# ---------------------------------------------------------------------------

def load_financial_data() -> pd.DataFrame:
    """
    Load monthly expense and budget data per department.

    We aggregate by department, year, quarter, and month because the CFO does not need prediction at transaction level here.
    The goal is to predict budget overrun at department and quarter level.

    So we summarize transaction rows into monthly department-level data.
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

    # Simple checks so we know the data loaded correctly.
    print(f"Loaded {len(df):,} department-month rows from production")
    print(f"Departments: {sorted(df['department'].unique().tolist())}")
    print(f"Years: {sorted(df['year'].unique().tolist())}")

    return df


# ---------------------------------------------------------------------------
# Step 2: Build quarterly summary and target variable
# ---------------------------------------------------------------------------

def build_quarterly_summary(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly data into quarterly data.

    The target variable is created at quarter level:
        overrun = 1 if quarterly expense > quarterly budget
        overrun = 0 otherwise

    We use quarter level because budgets are usually planned and reviewed by quarter.
    A department may overspend in one month but recover in another month, so the quarter is a more meaningful business period.
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

    # Create the target variable.
    quarterly_df["overrun"] = (
        quarterly_df["quarterly_expense"] > quarterly_df["quarterly_budget"]
    ).astype(int)

    # Calculate the quarterly variance percentage.
    quarterly_df["quarterly_variance_pct"] = (
        (quarterly_df["quarterly_expense"] - quarterly_df["quarterly_budget"])
        / quarterly_df["quarterly_budget"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    overrun_rate = quarterly_df["overrun"].mean()
    count_overrun = quarterly_df["overrun"].sum()
    count_not_overrun = (quarterly_df["overrun"] == 0).sum()

    print(f"Quarterly records: {len(quarterly_df):,}")
    print(f"Overrun rate: {overrun_rate:.1%}")
    print(f"Class distribution: {count_overrun} overrun, {count_not_overrun} within budget")

    return quarterly_df


# ---------------------------------------------------------------------------
# Step 3: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create the features used by the XGBoost model.

   
    Features created:
        1. spending_velocity
        2. ytd_variance_pct
        3. dept_overrun_history_rate
        4. month_of_quarter
        5. budget_utilisation_rate
        6. prior_quarter_variance_pct
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
    # This measures how fast the department is spending this month.
    #

    # A value above 1 means the department spent more than the monthly budget.
    df["spending_velocity"] = (
        df["actual_expense"] / df["budget_amount"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    # -----------------------------------------------------------------------
    # Feature 3: ytd_variance_pct
    # -----------------------------------------------------------------------
    # This feature checks if the department is over or under budget from the beginning of the year until the current month.

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
    # This measures how often each department exceeded its budget historically.
   
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
    # This measures how much of the quarterly budget has been used so far.
   
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
    # This feature tells the model how much the department over or under spent in the previous quarter.
  
    prior_quarter_df = quarterly_df[
        ["department", "year", "quarter", "quarterly_variance_pct"]
    ].copy()

    prior_quarter_df = prior_quarter_df.rename(
        columns={
            "quarterly_variance_pct": "prior_quarter_variance_pct"
        }
    )
    # We want to attach last quarter's variance to the next quarter.
    
    prior_quarter_df["target_year"] = prior_quarter_df["year"]
    prior_quarter_df["target_quarter"] = prior_quarter_df["quarter"] + 1


    # Handle year rollover:
    # If next_quarter becomes 5, that means the next quarter is Q1 of the following year.
    q4_mask = prior_quarter_df["target_quarter"] > 4

    prior_quarter_df.loc[q4_mask, "target_quarter"] = 1
    prior_quarter_df.loc[q4_mask, "target_year"] = (
        prior_quarter_df.loc[q4_mask, "year"] + 1
    )

    # Rename columns so they match the current quarter rows during merge.
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

    # Merge the previous quarter variance into the current quarter rows.
    df = df.merge(
        prior_quarter_df,
        on=["department", "year", "quarter"],
        how="left"
    )

    # The first quarter in the dataset has no previous quarter.
    # We fill it with 0.0 as a neutral value.
    df["prior_quarter_variance_pct"] = (
        df["prior_quarter_variance_pct"].fillna(0.0)
    )

    # -----------------------------------------------------------------------
    # Add target variable to monthly rows
    # -----------------------------------------------------------------------
    # The target is calculated at quarter level.
    # Each month inside the same quarter gets the same final overrun label.
    #
    # Example:
    #   IT Q1 overrun = 1
    #   January, February, and March rows for IT will all get overrun = 1.
    df = df.merge(
        quarterly_df[["department", "year", "quarter", "overrun"]],
        on=["department", "year", "quarter"],
        how="left"
    )

    # Remove rows where the target is missing.
    df = df.dropna(subset=["overrun"])

    df["overrun"] = df["overrun"].astype(int)

    # These are the only columns the model will use as input.
    feature_columns = [
        "spending_velocity",
        "ytd_variance_pct",
        "dept_overrun_history_rate",
        "month_of_quarter",
        "budget_utilisation_rate",
        "prior_quarter_variance_pct",
    ]

    print(f"Features engineered: {len(df):,} department-month rows")
    print(f"Feature columns: {feature_columns}")

    return df, feature_columns


# ---------------------------------------------------------------------------
# Step 4: Train XGBoost classifier
# ---------------------------------------------------------------------------

def train_xgboost(
    df: pd.DataFrame,
    feature_columns: list[str]
) -> tuple[xgb.XGBClassifier, list[str]]:
    """
    Train the XGBoost model.

    """

    X = df[feature_columns].fillna(0)

    y = df["overrun"]

    # Count both classes.
    count_negative = (y == 0).sum()
    count_positive = (y == 1).sum()

    # scale_pos_weight helps when one class appears more than the other.
    #
    # Formula:
    #   count of class 0 / count of class 1
    #
    # This tells XGBoost how much weight to give to the positive class.
    if count_positive == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = count_negative / count_positive

    print(f"Training samples: {len(X):,}")
    print(f"Overrun class count: {count_positive}")
    print(f"Non-overrun class count: {count_negative}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Create the XGBoost classifier.
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )

    # Train the model.
    model.fit(X, y)

    print("XGBoost model trained successfully")

    return model, feature_columns


# ---------------------------------------------------------------------------
# Step 5: Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    feature_columns: list[str]
) -> dict:
    """
    Evaluate the trained model.

    We use:
        - AUC-ROC
        - Precision
        - Recall
        - F1 Score

    """

    X = df[feature_columns].fillna(0)
    y = df["overrun"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Cross-validation AUC gives a more stable estimate than one train/test split.
    auc_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    # Predict class labels on the full dataset.
    y_pred = model.predict(X)

    # Predict probability of overrun.
    # [:, 1] means probability of class 1, which is overrun.
    y_pred_proba = model.predict_proba(X)[:, 1]

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y, y_pred_proba)

    print("\nModel Evaluation")
    print("-" * 60)
    print(f"Cross-validated AUC-ROC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    print(f"Full dataset AUC-ROC: {auc_roc:.4f}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.4f}")

    if auc_roc >= 0.80:
        quality = "Good"
    elif auc_roc >= 0.65:
        quality = "Needs review"
    else:
        quality = "Poor"

    print(f"Model quality: {quality}")

    # Show feature importance.
    # This tells us which features the model used the most.
    print("\nFeature Importance")
    print("-" * 60)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    for feature_name, score in importance.items():
        print(f"{feature_name:<30}: {score:.4f}")

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std": round(auc_scores.std(), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    return metrics


# ---------------------------------------------------------------------------
# Step 6: Save model
# ---------------------------------------------------------------------------

def save_model(
    model: xgb.XGBClassifier,
    feature_columns: list[str],
    metrics: dict
) -> str:
    """
    Save the trained model and useful metadata.
    """

    model_data = {
        "model": model,
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": metrics,
    }

    file_path = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")

    with open(file_path, "wb") as file:
        pickle.dump(model_data, file)

    print(f"\nModel saved to: {file_path}")

    return file_path


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_budget_overrun_model() -> dict:
    """
    Run the full training pipeline from start to finish.

    Pipeline steps:
        1. Load monthly financial data from PostgreSQL
        2. Build quarterly summary and target variable
        3. Engineer model features
        4. Train XGBoost classifier
        5. Evaluate the model
        6. Save the model
    """

    print("=" * 60)
    print("Training XGBoost Budget Overrun Classifier")
    print("=" * 60)

    print("\nStep 1: Loading financial data")
    monthly_df = load_financial_data()

    print("\nStep 2: Building quarterly summary")
    quarterly_df = build_quarterly_summary(monthly_df)

    print("\nStep 3: Engineering features")
    training_df, feature_columns = engineer_features(
        monthly_df=monthly_df,
        quarterly_df=quarterly_df
    )

    print("\nStep 4: Training model")
    model, feature_columns = train_xgboost(
        df=training_df,
        feature_columns=feature_columns
    )

    print("\nStep 5: Evaluating model")
    metrics = evaluate_model(
        model=model,
        df=training_df,
        feature_columns=feature_columns
    )

    print("\nStep 6: Saving model")
    save_model(
        model=model,
        feature_columns=feature_columns,
        metrics=metrics
    )

    return metrics


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = train_budget_overrun_model()

    print("\nFinal Metrics")
    print("-" * 60)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")