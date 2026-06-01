"""

This file is the single entry point for all ML predictions in the CFO dashboard.

Instead of letting the Streamlit dashboard call each model separately, this file collects all model outputs in one clean function.

The pipeline runs:
    1. Prophet revenue forecast
    2. Isolation Forest anomaly detection
    3. XGBoost budget overrun prediction
    4. SHAP explainability charts


"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------


project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Unified prediction function
# ---------------------------------------------------------------------------

def run_all_predictions(
    year: int,
    month: int,
    forecast_periods: int = 6,
) -> dict:
    """
    Run all ML models and return their outputs in one dictionary.

   
    We return None instead of stopping the whole script because in a dashboard, one failed model should not break everything.
    For example, if Prophet fails, the CFO should still be able to see anomaly detection and overrun predictions.
    """

    # Create the main results dictionary.
    # We start with empty values, then fill them as each model runs.
    results = {
        "forecast": None,
        "anomalies": None,
        "overrun": None,
        "shap": {},
        "errors": {},
        "metadata": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "year": year,
            "month": month,
            "forecast_periods": forecast_periods,
        }
    }

    # -----------------------------------------------------------------------
    # Model 1: Prophet Revenue Forecast
    # -----------------------------------------------------------------------

    print("-- Running Prophet Forecast --")

    try:
        # Import the Prophet prediction function only when this step runs.
        # This keeps the pipeline flexible and makes failures easier to isolate.
        from ml.forecasting.predict import forecast_all_departments

        # Run the forecast model.
        # This should return a DataFrame containing historical and forecast rows.
        results["forecast"] = forecast_all_departments(
            periods=forecast_periods
        )

        # Count only the future forecast rows.
        # The forecast output is expected to have an is_forecast column.
        future_only = results["forecast"][
            results["forecast"]["is_forecast"]
        ]

        print(f"Forecast completed: {len(future_only)} future periods returned")

    except Exception as error:
        # If Prophet fails, store the error message and continue.
        # The rest of the pipeline should still run.
        results["errors"]["forecast"] = str(error)
        print(f"Forecast failed: {error}")

    # -----------------------------------------------------------------------
    # Model 2: Isolation Forest Anomaly Detection
    # -----------------------------------------------------------------------

    print("-- Running Anomaly Detection --")

    try:
        # Import the anomaly detection helper functions.
        from ml.anomaly.detect import (
            load_expense_transactions,
            detect_anomalies,
        )

        # Load only expense transactions because the anomaly model was trained to detect unusual expense behavior.
        df_transactions = load_expense_transactions()

        # Run anomaly detection.
        # The returned DataFrame should include:
        #   - anomaly_score
        #   - is_anomaly_predicted
        results["anomalies"] = detect_anomalies(df_transactions)

        # Count how many transactions were flagged as anomalies.
        flagged_count = results["anomalies"]["is_anomaly_predicted"].sum()

        print(f"Anomaly detection completed: {flagged_count} transactions flagged")

    except Exception as error:
        # If anomaly detection fails, save the error and continue.
        results["errors"]["anomalies"] = str(error)
        print(f"Anomaly detection failed: {error}")

    # -----------------------------------------------------------------------
    # Model 3: XGBoost Budget Overrun Classifier
    # -----------------------------------------------------------------------

    print("-- Running Budget Overrun Predictions --")

    try:
        # Import the XGBoost prediction function.
        from ml.classification.predict_overrun import predict_overrun

        # Run budget overrun prediction for the selected year and month.
        # The result should contain department-level probabilities and risk levels.
        results["overrun"] = predict_overrun(year, month)

        # Count how many departments were classified as HIGH risk.
        high_risk = (
            results["overrun"]["risk_level"] == "HIGH"
        ).sum()

        print(f"Overrun prediction completed: {high_risk} HIGH risk departments")

    except Exception as error:
        # If overrun prediction fails, save the error and continue.
        results["errors"]["overrun"] = str(error)
        print(f"Overrun prediction failed: {error}")

    # -----------------------------------------------------------------------
    # Model 4: SHAP Explainability Charts
    # -----------------------------------------------------------------------

    print("-- Generating SHAP Charts --")

    try:
        # Import the SHAP chart generation function.
        from ml.explainability.shap_charts import generate_all_department_charts

        # Generate SHAP waterfall charts for all departments in the selected month.
        # This returns a dictionary:
        #   key   = department name
        #   value = Matplotlib Figure
        results["shap"] = generate_all_department_charts(year, month)

        print(f"SHAP completed: {len(results['shap'])} charts generated")

    except Exception as error:
        # If SHAP fails, save the error and continue.
        # The dashboard can still show forecast, anomaly, and overrun results.
        results["errors"]["shap"] = str(error)
        print(f"SHAP charts failed: {error}")

    # -----------------------------------------------------------------------
    # Pipeline summary
    # -----------------------------------------------------------------------

    # Count how many model sections succeeded.
    # A section is considered successful if its key is not inside errors.
    successful = sum(
        1 for key in ["forecast", "anomalies", "overrun", "shap"]
        if key not in results["errors"]
    )

    print("\n-- Pipeline Complete --")
    print(f"Models successful: {successful}/4")

    # If any step failed, print which sections had errors.
    if results["errors"]:
        print(f"Errors: {list(results['errors'].keys())}")

    # Return the full result dictionary to the caller.
    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    print("=" * 60)
    print("Running Unified Prediction Pipeline")
    print("=" * 60)
    print()

    # Run the full prediction pipeline for a test month.
    results = run_all_predictions(
        year=2025,
        month=11
    )

    print("\n-- Output Summary --")

    # Summarize forecast output if Prophet succeeded.
    if results["forecast"] is not None:
        future = results["forecast"][
            results["forecast"]["is_forecast"]
        ]

        print(f"Forecast rows: {len(future)}")

    # Summarize anomaly detection output if it succeeded.
    if results["anomalies"] is not None:
        flagged = results["anomalies"]["is_anomaly_predicted"].sum()

        print(f"Anomalies flagged: {flagged}")

    # Summarize overrun prediction output if it succeeded.
    if results["overrun"] is not None:
        print("Overrun predictions:")

        print(
            results["overrun"][[
                "department",
                "overrun_probability",
                "risk_level"
            ]].to_string(index=False)
        )

    # Summarize SHAP output.
    print(f"SHAP charts: {len(results['shap'])}")

    # Print any errors that happened during the pipeline run.
    if results["errors"]:
        print("\nErrors encountered:")

        for model_name, error_message in results["errors"].items():
            print(f"{model_name}: {error_message}")