"""
One-time script to copy all local PostgreSQL data to Supabase.
We need this because Streamlit Cloud cannot connect to the PostgreSQL Docker container running on my laptop, so I need to move the database to a cloud PostgreSQL provider like Supabase.

Run this script once before deploying the Streamlit app.
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Make project imports work
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_DATABASE_URL is missing. "
        "Add it to your .env file before running this script."
    )


# ---------------------------------------------------------------------------
# Connect to Supabase
# ---------------------------------------------------------------------------

print("Connecting to Supabase...")

cloud_engine = create_engine(
    SUPABASE_URL,
    connect_args={"sslmode": "require"}
)

with cloud_engine.connect() as conn:
    conn.execute(text("SELECT 1"))

print("Connected to Supabase")


# ---------------------------------------------------------------------------
# Create schemas
# ---------------------------------------------------------------------------

print("Creating schemas...")

with cloud_engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS staging"))
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS production"))

print("Schemas created")


# ---------------------------------------------------------------------------
# Connect to local PostgreSQL
# ---------------------------------------------------------------------------
# local_engine points to the PostgreSQL database running locally in Docker.
# This is where my current project data already exists.
# ---------------------------------------------------------------------------
print("Reading data from local PostgreSQL...")

from etl.extract.db import engine as local_engine


# ---------------------------------------------------------------------------
# Tables to copy from local PostgreSQL to Supabase
# ---------------------------------------------------------------------------
# These are the tables needed by the Streamlit dashboard.
# ---------------------------------------------------------------------------
tables = [
    "production.dim_date",
    "production.dim_department",
    "production.dim_region",
    "production.dim_category",
    "production.fact_financials",
    "staging.raw_transactions",
    "staging.raw_monthly_summary",
]


# ---------------------------------------------------------------------------
# Read each local table into a pandas DataFrame
# ---------------------------------------------------------------------------
# I keep all DataFrames in a dictionary so I can write them later.
# ---------------------------------------------------------------------------
dataframes = {}

with local_engine.connect() as conn:
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        dataframes[table] = df
        print(f"  Read {len(df):,} rows from {table}")

print("Data read from local PostgreSQL")


# ---------------------------------------------------------------------------
# Write data to Supabase
# ---------------------------------------------------------------------------

print("\nWriting data to Supabase...")

write_order = [
    "staging.raw_transactions",
    "staging.raw_monthly_summary",
    "production.dim_date",
    "production.dim_department",
    "production.dim_region",
    "production.dim_category",
    "production.fact_financials",
]

for table in write_order:
    schema, table_name = table.split(".")
    df = dataframes[table]

    df.to_sql(
        name=table_name,
        con=cloud_engine,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=500,
    )

    print(f"  {table}: {len(df):,} rows written")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n All data loaded to Supabase successfully!")
print(f"   fact_financials: {len(dataframes['production.fact_financials']):,} rows")
print(f"   dim_date:        {len(dataframes['production.dim_date']):,} rows")
print(f"   dim_department:  {len(dataframes['production.dim_department']):,} rows")