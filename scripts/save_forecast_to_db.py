"""
Run locally to generate Prophet forecasts and save results to Supabase.


Streamlit Cloud should read pre-computed forecast results from Supabase instead of loading Prophet .pkl models directly.

Run this script locally after training Prophet models.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Make project imports work
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Load Supabase connection from .env
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_DATABASE_URL is missing. Add it to your .env file."
    )


# ---------------------------------------------------------------------------
# Connect to Supabase
# ---------------------------------------------------------------------------

cloud_engine = create_engine(
    SUPABASE_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,
)


# ---------------------------------------------------------------------------
# Import Prophet forecast functions
# ---------------------------------------------------------------------------

from ml.forecasting.predict import forecast_department, get_model_metadata


# ---------------------------------------------------------------------------
# Create schema if needed
# ---------------------------------------------------------------------------

with cloud_engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS production"))


# ---------------------------------------------------------------------------
# Generate forecasts locally
# ---------------------------------------------------------------------------

all_forecasts = []

for dept in ["Company_Total", "Sales"]:
    print(f"Generating forecast for {dept}...")

    df = forecast_department(dept, periods=12)

    if df.empty:
        print(f" No forecast returned for {dept}")
        continue

    df = df.copy()
    df["department"] = dept

    all_forecasts.append(df)


if not all_forecasts:
    raise ValueError("No forecast data was generated.")


forecast_df = pd.concat(all_forecasts, ignore_index=True)


# ---------------------------------------------------------------------------
# Clean forecast columns before saving
# ---------------------------------------------------------------------------

forecast_df["ds"] = pd.to_datetime(forecast_df["ds"]).dt.date

if "is_forecast" in forecast_df.columns:
    forecast_df["is_forecast"] = forecast_df["is_forecast"].astype(bool)

# Optional: keep only the columns Streamlit needs
forecast_columns = [
    "department",
    "ds",
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "is_forecast",
]

forecast_df = forecast_df[forecast_columns]


# ---------------------------------------------------------------------------
# Save forecast results to Supabase
# ---------------------------------------------------------------------------

forecast_df.to_sql(
    name="prophet_forecasts",
    con=cloud_engine,
    schema="production",
    if_exists="replace",
    index=False,
    chunksize=500,
)

print(f" Saved {len(forecast_df):,} forecast rows to Supabase")


# ---------------------------------------------------------------------------
# Save model metadata to Supabase
# ---------------------------------------------------------------------------

meta_df = get_model_metadata()

if meta_df.empty:
    print(" No model metadata found.")
else:
    meta_df = meta_df.copy()

    if "trained_at" in meta_df.columns:
        meta_df["trained_at"] = meta_df["trained_at"].astype(str)

    meta_df.to_sql(
        name="prophet_metadata",
        con=cloud_engine,
        schema="production",
        if_exists="replace",
        index=False,
        chunksize=500,
    )

    print(f" Saved {len(meta_df):,} model metadata rows to Supabase")


print(" Forecast export completed successfully.")