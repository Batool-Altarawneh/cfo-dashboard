"""
Purpose:
This script trains one Prophet forecasting model for each department.

In this project, the goal is to forecast future REVENUE using historical monthly revenue data from the production database.

I will use production.fact_financials
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


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------


project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, project_root)



from etl.extract.db import engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# I keep important settings in one place so the script is easier to edit later.
# For example, if I want to forecast more months or add a department, I only need to change this section.
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(project_root, "ml", "models")

DEPARTMENTS = [
    "IT",
    "Marketing",
    "Sales",
    "HR",
    "Operations"
]

REVENUE_SOURCES = [
    "Sales"
]

FORECAST_MONTHS = 6

TRAIN_END_DATE = "2025-12-31"


# Create the models folder if it does not already exist.
# exist_ok=True means:
#   - create the folder if missing
#   - do not raise an error if it already exists
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load revenue data from PostgreSQL
# ---------------------------------------------------------------------------

def load_revenue_data() -> pd.DataFrame:
    """
    Load historical revenue data from the production database.

    The Prophet model needs two main columns:
        ds -> the date column
        y  -> the numeric value we want to forecast

    In this project:
        ds = financial date
        y  = revenue amount

    I also load department and region because I need to train one model per department.
    """

    query = """
        SELECT
            dt.full_date              AS ds,
            dt.year                   AS year,
            dt.month                  AS month,
            d.dept_name               AS department,
            r.region_name             AS region,
            SUM(f.amount)             AS y
        FROM production.fact_financials f
        JOIN production.dim_date dt
            ON f.date_key = dt.date_key
        JOIN production.dim_department d
            ON f.dept_key = d.dept_key
        JOIN production.dim_region r
            ON f.region_key = r.region_key
        WHERE f.transaction_type = 'REVENUE'
          AND dt.full_date <= :end_date
        GROUP BY
            dt.full_date,
            dt.year,
            dt.month,
            d.dept_name,
            r.region_name
        ORDER BY
            d.dept_name,
            dt.full_date;
    """

    # Open a database connection and run the SQL query.
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params={"end_date": TRAIN_END_DATE}
        )

    # Prophet requires the ds column to be a real datetime column.
    df["ds"] = pd.to_datetime(df["ds"])

    print(f"  Loaded {len(df):,} revenue rows from production")

    return df


# ---------------------------------------------------------------------------
# Aggregate data to monthly revenue per department
# ---------------------------------------------------------------------------

def aggregate_monthly_by_dept(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw revenue rows into monthly revenue totals per department.

     I aggregate by month لاecause the CFO forecast is usually monthly, not daily or transaction-level.

        Instead of forecasting every single transaction,
        I forecast total monthly revenue for Sales, HR, IT, etc.

    """

    monthly = (
        df
        .groupby(["department", "year", "month"])["y"]
        .sum()
        .reset_index()
    )


    monthly["ds"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str).str.zfill(2)
        + "-01"
    )

    # Keep only the columns needed for training.
    monthly = monthly[["department", "ds", "y"]]

    # Sort the data so each department's time series is in date order.
    monthly = monthly.sort_values(["department", "ds"])

    return monthly


# ---------------------------------------------------------------------------
# Train one Prophet model
# ---------------------------------------------------------------------------

def train_prophet_model(df_dept: pd.DataFrame, dept_name: str):
    """
    Train a Prophet model for one department.

        df_dept:
            DataFrame containing monthly revenue for one department.
            Must contain:
                ds -> date
                y  -> revenue

        dept_name:
            Department name, used for logging and saving the model.

    
        A trained Prophet model.
    """

    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet is not installed. Run this command first: pip install prophet"
        )

    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        interval_width=0.80
    )

    # Add custom monthly seasonality.
    #
    # This helps the model learn patterns that repeat every month,
    # such as end-of-month or end-of-quarter revenue behavior.
    model.add_seasonality(
        name="monthly",
        period=30.5, #month periode
        fourier_order=5 #The more fourier_orders there are, the more complex the model can draw seasonal patterns.
    )

    # Train the model.
   
    model.fit(df_dept[["ds", "y"]])

    print(
        f"{dept_name}: model trained on "
        f"{len(df_dept)} monthly observations"
    )

    return model


# ---------------------------------------------------------------------------
# Evaluate model performance
# ---------------------------------------------------------------------------

def evaluate_model(df_dept: pd.DataFrame, dept_name: str) -> dict:
    """
    Evaluate the Prophet model using a simple train/test split.

        - Train on data before 2025
        - Test on data from 2025

    This simulates a real forecasting situation which is use the past to predict the future

    Metrics:
        MAE:
            Mean Absolute Error.
            Shows the average dollar error per month.

        MAPE:
            Mean Absolute Percentage Error.
            Shows the average percentage error per month.

    Example:
        MAPE = 8%
        means the model is off by around 8% on average.
    """

    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet is not installed. Run this command first: pip install prophet"
        )

    # Split the department data into train and test.
    train = df_dept[df_dept["ds"] < "2025-01-01"]
    test = df_dept[df_dept["ds"] >= "2025-01-01"]

    # If there is no test data, I cannot calculate model accuracy.
    if len(test) == 0:
        print(f"  {dept_name}: not enough test data for evaluation")
        return {}

    # Create a separate model only for evaluation.
    # This model is trained only on the training period.
    eval_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1
    )

    eval_model.fit(train[["ds", "y"]])

    # Create future dates for the test period.
    # freq='MS' means Month Start.
    # Example:
    #   2025-01-01
    #   2025-02-01
    #   2025-03-01
    future = eval_model.make_future_dataframe(
        periods=12,
        freq="MS"
    )

    # Generate predictions.
    forecast = eval_model.predict(future)

    # Keep only the predictions that match the actual test dates.
    test_predictions = forecast[
        forecast["ds"].isin(test["ds"])
    ][["ds", "yhat"]]

    # Join actual values with predicted values.
    test_actual = test[["ds", "y"]].merge(
        test_predictions,
        on="ds",
        how="inner"
    )

    if len(test_actual) == 0:
        print(f"  {dept_name}: no matching forecast dates for evaluation")
        return {}

    # MAE = average absolute difference between actual and predicted revenue.
    mae = np.mean(
        np.abs(test_actual["y"] - test_actual["yhat"])
    )

    # MAPE = average percentage error.
    # I replace 0 revenue values with NaN to avoid division by zero.
    actual_y = test_actual["y"].replace(0, np.nan)

    mape = np.mean(
        np.abs((test_actual["y"] - test_actual["yhat"]) / actual_y)
    ) * 100

    metrics = {
        "mae": round(mae, 2),
        "mape": round(mape, 2)
    }

    # Simple interpretation to make the result easier to understand.
    if mape < 10:
        quality = " Good"
    elif mape < 20:
        quality = " Review"
    else:
        quality = "Poor"

    print(
        f"  {dept_name} evaluation: "
        f"MAE=${mae:,.0f}  "
        f"MAPE={mape:.1f}%  "
        f"{quality}"
    )

    return metrics


def train_company_revenue_model(df_monthly: pd.DataFrame) -> dict:
 

    df_total = (
        df_monthly
        .groupby("ds")["y"]
        .sum()
        .reset_index()
    )

    print(f"  Company_Total: {len(df_total)} monthly revenue observations")

    # Evaluate the company revenue model using the same holdout logic.
    metrics = evaluate_model(
        df_total,
        "Company_Total"
    )

    # Train the final model on all available company revenue data.
    model = train_prophet_model(
        df_total,
        "Company_Total"
    )

    # Save the trained model as:
    #   ml/models/prophet_Company_Total.pkl
    save_model(
        model,
        "Company_Total",
        metrics
    )

    return metrics


# ---------------------------------------------------------------------------
# Save trained model
# ---------------------------------------------------------------------------

def save_model(model, dept_name: str, metrics: dict) -> str:
    """
    Save the trained Prophet model as a .pkl file.

    """

    model_data = {
        "model": model,
        "department": dept_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "train_end": TRAIN_END_DATE
    }

  
    file_path = os.path.join(
        MODELS_DIR,
        f"prophet_{dept_name}.pkl"
    )

    # wb means write binary.
    # Pickle files must be written in binary mode.
    with open(file_path, "wb") as file:
        pickle.dump(model_data, file)

    print(f"   Saved: {file_path}")

    return file_path


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_all_departments() -> dict:
    """
    Run the full Prophet training pipeline.

    """

    print("=" * 60)
    print("  Training Prophet Revenue Forecasting Models")
    print("=" * 60)

    # Step 1: Load raw revenue data from production tables.
    print("\n── Loading Revenue Data ──")
    df_raw = load_revenue_data()

    # Step 2: Convert the data to monthly revenue per department.
    df_monthly = aggregate_monthly_by_dept(df_raw)

    print(f"  Monthly observations: {len(df_monthly):,}")
    print(
        f"  Date range: "
        f"{df_monthly['ds'].min().date()} → "
        f"{df_monthly['ds'].max().date()}"
    )

    print("\n── Training Models ──")

    all_metrics = {}

    # -----------------------------------------------------------------------
    # Step 3A: Train the main company-wide revenue model first.
    # -----------------------------------------------------------------------
    # This is the most important forecasting model for the CFO dashboard.
    # It answers:
    #   "What is the expected total company revenue for the next months?"
    # -----------------------------------------------------------------------

    print("\n── Company Total Revenue Model ──")

    company_metrics = train_company_revenue_model(df_monthly)

    all_metrics["Company_Total"] = company_metrics


    # -----------------------------------------------------------------------
    # Step 3B: Train department-level revenue models only where revenue exists.
    # -----------------------------------------------------------------------
    # In this dataset, only Sales has REVENUE transactions.
    # Other departments are skipped because they have expenses, not revenue.
    # -----------------------------------------------------------------------

    print("\n── Department Revenue Models ──")

    for dept in REVENUE_SOURCES:
        df_dept = df_monthly[
            df_monthly["department"] == dept
        ].copy()

        if len(df_dept) < 12:
            print(
                f"  ⚠️ {dept}: no revenue data "
                f"({len(df_dept)} months) — skipping"
            )
            continue

        # Evaluate the department-level model.
        metrics = evaluate_model(
            df_dept,
            dept
        )

        # Train final model on all available data.
        model = train_prophet_model(
            df_dept,
            dept
        )

        # Save model + metadata.
        save_model(
            model,
            dept,
            metrics
        )

        all_metrics[dept] = metrics

    print("\n── Training Complete ──")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Departments trained: {list(all_metrics.keys())}")

    return all_metrics


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    metrics = train_all_departments()

    print("\n── Model Performance Summary ──")

    for dept, dept_metrics in metrics.items():
        if dept_metrics:
            print(
                f"  {dept:<12}: "
                f"MAPE={dept_metrics.get('mape', 'N/A')}%  "
                f"MAE=${dept_metrics.get('mae', 'N/A'):,}"
            )