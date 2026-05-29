"""
This file loads saved Prophet models and generates revenue forecasts.

This file only loads the already-trained .pkl model files and uses them to predict future revenue.

"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import pickle

import pandas as pd


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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder where train_prophet.py saves the .pkl files.
MODELS_DIR = os.path.join(project_root, "ml", "models")

# Departments used in this CFO dashboard project.
DEPARTMENTS = [
    "IT",
    "Marketing",
    "Sales",
    "HR",
    "Operations"
]


# ---------------------------------------------------------------------------
# Load one saved model
# ---------------------------------------------------------------------------

def load_model(dept_name: str) -> dict:
    """
    Load the saved Prophet model for one department.

    Parameters:
        dept_name:
            The department name, for example:
                Sales
                IT
                HR

    Returns:
        A dictionary called model_data.

    model_data contains:
        model:
            The trained Prophet model object.

        department:
            The department this model belongs to.

        trained_at:
            The timestamp showing when the model was trained.

        metrics:
            Evaluation results such as MAE and MAPE.

        train_end:
            The last date used during training.
    """

    # Build the expected model file path.
    
    path = os.path.join(
        MODELS_DIR,
        f"prophet_{dept_name}.pkl"
    )

    # If the model file does not exist, I cannot make a forecast.
    # This usually means train_prophet.py has not been run yet.
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found for '{dept_name}'. "
            f"Run train_prophet.py first."
        )

   
    with open(path, "rb") as file:
        model_data = pickle.load(file)

    return model_data


# ---------------------------------------------------------------------------
# Forecast one department
# ---------------------------------------------------------------------------

def forecast_department(
    dept_name: str,
    periods: int = 6,
    freq: str = "MS"
) -> pd.DataFrame:
   

    # Load the saved model data for this department.
    model_data = load_model(dept_name)

    # Extract the actual Prophet model object.
    model = model_data["model"]

    # Extract the last date used during training.
    # I use this later to separate historical fitted values from future forecast values.
    train_end = model_data["train_end"]

    # Create a future DataFrame.
    #
    # Prophet needs a DataFrame containing the future dates we want predictions for.
    #
    # periods=6 and freq="MS" means:
    #   create 6 future monthly dates at the start of each month.
    future = model.make_future_dataframe(
        periods=periods,
        freq=freq
    )

    # Generate forecast results.
    #
    # Prophet returns many columns, but I only need the main forecast and the lower/upper bounds for the dashboard.
    forecast = model.predict(future)

    # Keep only the columns needed for reporting.
    result = forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    # Revenue should not be negative.
    #
    # Sometimes forecasting models may produce a small negative value
    # if the trend or uncertainty goes below zero.
    # For a business dashboard, I clip those values to 0.
    result["yhat"] = result["yhat"].clip(lower=1.0).round(2)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0).round(2)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0).round(2)

    result["department"] = dept_name

    result["trained_at"] = model_data["trained_at"]

    # Mark whether each row is a future forecast or historical fitted value.
    #
    # Prophet returns both:
    #   - historical dates it already knows
    #   - future dates it forecasted
    #
    # Anything after train_end is a real future forecast.
    result["is_forecast"] = result["ds"] > pd.Timestamp(train_end)

    return result


# ---------------------------------------------------------------------------
# Forecast all departments
# ---------------------------------------------------------------------------

def forecast_all_departments(periods: int = 6) -> pd.DataFrame:
    """
    Generate forecasts for all departments.

    This is useful for the dashboard because I can show all departments on one chart and use department as a filter.

    Parameters:
        periods:
            Number of months to forecast ahead.

    Returns:
        One combined DataFrame containing forecasts for all departments.
    """

    all_forecasts = []

    # Loop through all departments and forecast each one.
    for dept in DEPARTMENTS:
        try:
            dept_forecast = forecast_department(
                dept_name=dept,
                periods=periods
            )

            all_forecasts.append(dept_forecast)

        except FileNotFoundError:
            # If one department model is missing, I do not stop the whole script.
            # I skip that department and continue with the others.
            print(f"  No model found for {dept} — skipping")

    # If no models were loaded, then prediction cannot continue.
    if not all_forecasts:
        raise RuntimeError(
            "No forecast models found. Run train_prophet.py first."
        )

    # Combine all department forecast DataFrames into one DataFrame.
    combined_forecast = pd.concat(
        all_forecasts,
        ignore_index=True
    )

    return combined_forecast


# ---------------------------------------------------------------------------
# Get model metadata
# ---------------------------------------------------------------------------

def get_model_metadata() -> pd.DataFrame:
    """
    Return metadata for all saved models.

    This is useful for the dashboard because I can show:
        - department name
        - when the model was trained
        - last training date
        - MAPE
        - MAE

    This helps the CFO or dashboard user understand model quality.
    """

    rows = []

    for dept in DEPARTMENTS:
        path = os.path.join(
            MODELS_DIR,
            f"prophet_{dept}.pkl"
        )

        # Only read metadata if the model file exists.
        if os.path.exists(path):
            with open(path, "rb") as file:
                data = pickle.load(file)

            rows.append({
                "department": dept,
                "trained_at": data.get("trained_at", "Unknown"),
                "train_end": data.get("train_end", "Unknown"),
                "mape": data.get("metrics", {}).get("mape", None),
                "mae": data.get("metrics", {}).get("mae", None)
            })

    metadata = pd.DataFrame(rows)

    return metadata


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Testing predict.py")
    print("─" * 40)

    # Test one department first.
    # Sales is usually the most interesting because it may have Q4 seasonality.
    sales_forecast = forecast_department(
        dept_name="Sales",
        periods=6
    )

    print("\nSales forecast — next 6 months:")

    # Keep only future forecast rows, not historical fitted rows.
    future_only = sales_forecast[
        sales_forecast["is_forecast"]
    ].copy()

    print(
        future_only[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].to_string(index=False)
    )

    print("\nAll department metadata:")

    metadata = get_model_metadata()

    print(
        metadata.to_string(index=False)
    )