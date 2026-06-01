"""
This file generates SHAP waterfall charts for the XGBoost budget overrun model.

The XGBoost model can predict that a department has a high probability of going over budget, but this file helps explain why the model made that prediction.

"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import pickle

import pandas as pd
import numpy as np

import matplotlib

# We use the "Agg" backend because this script may run inside Streamlit
# This means charts can be created and saved without opening a popup window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import shap


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


sys.path.insert(0, project_root)

MODELS_DIR = os.path.join(project_root, "ml", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")


# ---------------------------------------------------------------------------
# Load XGBoost model
# ---------------------------------------------------------------------------

def load_xgboost_model() -> dict:
    """
    Load the saved XGBoost model and its metadata from disk.

    The saved pickle file does not only contain the model.
    It also contains metadata such as:
        - feature_columns
        - training date
        - evaluation metrics

    
    """

    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    return model_data


# ---------------------------------------------------------------------------
# Calculate SHAP values
# ---------------------------------------------------------------------------

def calculate_shap_values(
    features_df: pd.DataFrame,
    feature_columns: list,
    model
) -> tuple:
    """
    Calculate SHAP values for all rows in the feature DataFrame.

    SHAP values tell us how much each feature contributed to the model's prediction for each row.

    
    """

    # Select only the columns that were used during model training.
    # fillna(0) is used to avoid errors if any feature value is missing.
    X = features_df[feature_columns].fillna(0)

    # TreeExplainer is designed for tree-based models like XGBoost.
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for every row in X.
    # Each row gets one SHAP value per feature.
    shap_values = explainer.shap_values(X)

    return shap_values, explainer, X


# ---------------------------------------------------------------------------
# Generate SHAP waterfall chart for one department
# ---------------------------------------------------------------------------

def generate_waterfall_chart(
    department: str,
    year: int,
    month: int,
    features_df: pd.DataFrame,
    feature_columns: list,
    model,
    shap_values: np.ndarray,
    explainer,
    X: pd.DataFrame,
) -> plt.Figure:
    """
    Generate one SHAP waterfall chart for one department in one month.

    The waterfall chart explains how the model moved from the baseline prediction to the final prediction.

    Example:
        Baseline prediction starts at 0.50
        spending_velocity pushes it up
        ytd_variance_pct pushes it up
        prior_quarter_variance may push it down
        final prediction becomes 0.99

 
    """

    # Create a filter to find the row for the selected department and period.
    mask = (
        (features_df["department"] == department) &
        (features_df["year"] == year) &
        (features_df["month"] == month)
    )

    # Apply the filter.
    matching_rows = features_df[mask]

    # If no row exists for this department/month, stop and show a clear error.
    if len(matching_rows) == 0:
        raise ValueError(
            f"No data found for {department} in {year}-{month:02d}"
        )

    # Use the first matching row.
    # There should usually be only one row per department per month.
    row_index = matching_rows.index[0]

    # X has the same original index as features_df.
    x_position = X.index.get_loc(row_index)

    # Get the SHAP values for this specific department/month row.
    row_shap_values = shap_values[x_position]

    # Get the actual feature values for this same row.
    row_feature_values = X.iloc[x_position]

    # Calculate the model's predicted probability of budget overrun.
    # predict_proba returns probabilities for both classes:
    # [probability of no overrun, probability of overrun]
    prediction_proba = model.predict_proba(
        X.iloc[[x_position]]
    )[0, 1]

    # Create readable labels for the chart.
    # Each label shows the feature name and its actual value.
    feature_labels = [
        f"{col}\n= {row_feature_values[col]:.3f}"
        for col in feature_columns
    ]

    # Sort features by absolute SHAP value.
    # This means the most influential features appear first.
    sorted_indices = np.argsort(np.abs(row_shap_values))[::-1]

    # Reorder SHAP values and feature labels using the sorted order.
    sorted_shap = row_shap_values[sorted_indices]
    sorted_labels = [feature_labels[i] for i in sorted_indices]

    # The baseline is the model's average expected output before seeing the specific department's feature values.
    baseline = explainer.expected_value

    # Build cumulative positions for the waterfall bars.
    # Each bar starts where the previous one ended.
    cumulative = [baseline]

    for shap_value in sorted_shap:
        cumulative.append(cumulative[-1] + shap_value)

    # Create the chart area.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose bar colors:
    # Red means the feature increased the prediction.
    # Blue means the feature decreased the prediction.
    colors = [
        "#E74C3C" if shap_value > 0 else "#3498DB"
        for shap_value in sorted_shap
    ]

    # Draw horizontal bars.
    ax.barh(
        y=range(len(sorted_shap)),
        width=sorted_shap,
        left=cumulative[:-1],
        color=colors,
        height=0.6,
        alpha=0.85
    )

    # Add text labels inside each bar to show the SHAP value.
    for i, (shap_value, left_position) in enumerate(
        zip(sorted_shap, cumulative[:-1])
    ):
        label = f"+{shap_value:.3f}" if shap_value > 0 else f"{shap_value:.3f}"

        # Put the label in the middle of the bar.
        x_position_label = left_position + shap_value / 2

        ax.text(
            x_position_label,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white"
        )

    # Draw a vertical dashed line for the baseline.
    ax.axvline(
        x=baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline: {baseline:.3f}"
    )

    # Draw a vertical line for the final prediction.
    # If probability is high, use red.
    # Otherwise, use orange.
    prediction_line_color = "#E74C3C" if prediction_proba >= 0.7 else "#F39C12"

    ax.axvline(
        x=cumulative[-1],
        color=prediction_line_color,
        linestyle="-",
        linewidth=2,
        label=f"Prediction: {prediction_proba:.1%}"
    )

    # Set y-axis feature labels.
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=9)

    # Add x-axis label.
    ax.set_xlabel(
        "SHAP Value (impact on overrun prediction)",
        fontsize=10
    )

    # Convert the probability into the same risk label used by predict_overrun.py.
    # This keeps the SHAP chart title consistent with the classifier output.
    if prediction_proba >= 0.70:
        risk_level = "HIGH"
        prediction_label = "Will Overrun"
    elif prediction_proba >= 0.40:
        risk_level = "MEDIUM"
        prediction_label = "At Risk"
    else:
        risk_level = "LOW"
        prediction_label = "Within Budget"

    # Use a dynamic title instead of saying every department is "at risk".
    # HIGH means the department is expected to overrun.
    # MEDIUM means the department is not certain to overrun, but should be watched.
    # LOW means the model expects the department to stay within budget.
    ax.set_title(
        f"Why is {department} predicted {risk_level} risk?\n"
        f"{prediction_label} | "
        f"Overrun Probability: {prediction_proba:.1%} | "
        f"{year}-{month:02d}",
        fontsize=12,
        fontweight="bold"
    )

    # Show chart legend.
    ax.legend(fontsize=9)

    # Add light grid lines on the x-axis to make the chart easier to read.
    ax.grid(axis="x", alpha=0.3)

    # Adjust spacing so labels do not overlap.
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Generate SHAP charts for all departments
# ---------------------------------------------------------------------------

def generate_all_department_charts(
    year: int,
    month: int,
) -> dict:
    """
    Generate SHAP waterfall charts for all departments in a selected month.

    This function is useful for the dashboard because the CFO usually wants to compare all departments for the same month.

   
    """

    # Import prediction helper functions from the classification module.
    # These functions prepare the same features used by the XGBoost model.
    #
    # It is important to reuse the same feature engineering logic.
    # If SHAP uses different features from training, the explanation will be wrong.
    from ml.classification.predict_overrun import (
        load_financial_data,
        build_quarterly_summary,
        engineer_features,
    )

    # Load the saved model data.
    model_data = load_xgboost_model()

    # Extract the actual XGBoost model.
    model = model_data["model"]

    # Extract the feature columns used during training.
    feature_columns = model_data["feature_columns"]

    # Load monthly financial data from the database or source used by predict_overrun.
    monthly_df = load_financial_data()

    # Build quarterly summary data because some features depend on quarterly logic.
    quarterly_df = build_quarterly_summary(monthly_df)

    # Engineer the same features used by the classifier.
    features_df = engineer_features(monthly_df, quarterly_df)

    # Calculate SHAP values once for the full feature dataset.
    # This is more efficient than recalculating SHAP values department by department.
    shap_values, explainer, X = calculate_shap_values(
        features_df=features_df,
        feature_columns=feature_columns,
        model=model
    )

    # Get departments that exist for the selected year and month.
    period_depts = features_df[
        (features_df["year"] == year) &
        (features_df["month"] == month)
    ]["department"].unique().tolist()

    # If no departments exist for that period, stop with a clear error.
    if not period_depts:
        raise ValueError(
            f"No data found for {year}-{month:02d}"
        )

    # This dictionary will store the final chart for each department.
    charts = {}

    # Generate one SHAP waterfall chart for each department.
    for dept in period_depts:
        try:
            fig = generate_waterfall_chart(
                department=dept,
                year=year,
                month=month,
                features_df=features_df,
                feature_columns=feature_columns,
                model=model,
                shap_values=shap_values,
                explainer=explainer,
                X=X,
            )

            charts[dept] = fig
            print(f"Chart generated: {dept}")

        except Exception as error:
            # If one department fails, we do not stop the entire process.
            # We print the error and continue with the next department.
            print(f"Could not generate chart for {dept}: {error}")

    return charts


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    

    print("=" * 60)
    print("Generating SHAP Waterfall Charts")
    print("=" * 60)

    # Choose the period we want to test.
    # You can change these values when you want to test another month.
    TARGET_YEAR = 2025
    TARGET_MONTH = 11

    print(f"\nGenerating charts for {TARGET_YEAR}-{TARGET_MONTH:02d}")

    # Generate charts for all departments in the selected month.
    charts = generate_all_department_charts(
        year=TARGET_YEAR,
        month=TARGET_MONTH
    )

    # Create an output folder to save the generated charts.
    output_dir = os.path.join(
        project_root,
        "ml",
        "models",
        "shap_charts"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save each department chart as a PNG image.
    for dept, fig in charts.items():
        file_name = f"shap_{dept}_{TARGET_YEAR}_{TARGET_MONTH:02d}.png"
        path = os.path.join(output_dir, file_name)

        fig.savefig(path, dpi=150, bbox_inches="tight")

        # Close the figure after saving to avoid memory issues.
        plt.close(fig)

        print(f"Saved: {path}")

    print(f"\n{len(charts)} charts saved to {output_dir}")