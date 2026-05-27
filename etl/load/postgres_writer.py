"""

This file is responsible for loading the final transformed tables into the production PostgreSQL schema.

At this stage, the data is already cleaned and shaped by the transform layer.

Its job is only to:
1. Write dimension and fact tables to PostgreSQL.
2. Use upsert logic to avoid duplicate rows.
3. Update existing rows when source values change.
4. Provide row counts so I can quickly verify the load.

Loading strategy
----------------
This file uses PostgreSQL UPSERT:

    INSERT ... ON CONFLICT DO UPDATE

This means:
- If the row does not exist, insert it.
- If the row already exists, update selected columns.

This is safer than a normal INSERT because the pipeline can be run
multiple times without creating duplicate records.
"""

import os
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert


# ---------------------------------------------------------------------------
# Make sure Python can find the project modules
# ---------------------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Add the project root to the Python path.
# This allows imports like: from etl.extract.db import engine
sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Import database connection and table models
# ---------------------------------------------------------------------------

from etl.extract.db import (
    engine,
    DimDate,
    DimDepartment,
    DimRegion,
    DimCategory,
    FactFinancials,
)


def upsert_dataframe(
    df: pd.DataFrame,
    orm_class,
    conflict_columns: list[str],
    update_columns: list[str],
) -> dict:
    """
    Insert or update rows in a PostgreSQL production table.

    
    """

    # Convert the DataFrame into a list of dictionaries.
    # SQLAlchemy can easily insert dictionaries where each dictionary = one row.
    #
    # Example:
    # DataFrame:
    #   transaction_id | amount
    #   T001           | 500
    #
    # Becomes:
    #   [{"transaction_id": "T001", "amount": 500}]
    records = df.to_dict(orient="records")

    # This counter tracks how many rows were affected by the upsert.
    # It includes both inserted and updated rows.
    rows_upserted = 0

    # engine.begin() starts a database transaction.
    #
    # If everything succeeds:
    #   SQLAlchemy commits the transaction.
    #
    # If an error happens:
    #   SQLAlchemy rolls back the transaction.
    #
    # This protects the database from being left half-loaded.
    with engine.begin() as conn:

        # Loop through each row from the DataFrame.
        for record in records:

            # Build a PostgreSQL INSERT statement for the target table.
            #
            # **record unpacks the dictionary into column=value pairs.
            # Example:
            #   {"amount": 500, "vendor": "ABC"}
            # becomes:
            #   amount=500, vendor="ABC"
            stmt = pg_insert(orm_class).values(**record)

            # Build the columns that should be updated if a conflict occurs.
            #
            # stmt.excluded[col] means:
            #   "use the new value that we tried to insert"
            #
            # Example:
            # If amount changed from 500 to 550,
            # EXCLUDED.amount refers to the new value: 550.
            update_dict = {
                col: stmt.excluded[col]
                for col in update_columns
            }

            # Add the UPSERT rule:
            #
            # ON CONFLICT (conflict_columns)
            # DO UPDATE SET update_columns = new values
            #
            # Example:
            # If transaction_id already exists, update amount and budget_amount.
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_columns,
                set_=update_dict,
            )

            # Execute the SQL statement in PostgreSQL.
            result = conn.execute(stmt)

            # PostgreSQL returns rowcount = 1 for both INSERT and UPDATE.
            # So this counter means "rows affected", not strictly "inserted".
            if result.rowcount > 0:
                rows_upserted += 1

    # Build a simple summary dictionary.
    stats = {
        "table": f"production.{orm_class.__tablename__}",
        "rows_upserted": rows_upserted,
        "total_input_rows": len(records),
    }

    # Print a friendly message so I can see what happened when the pipeline runs.
    print(
        f"   production.{orm_class.__tablename__}: "
        f"{rows_upserted:,} rows upserted"
    )

    return stats


def write_fact_financials(df: pd.DataFrame) -> dict:
    """
    Load the fact_financials table into PostgreSQL.

    This table is loaded using transaction_id as the conflict key.

   
    transaction_id identifies one financial transaction.

    If the same transaction appears again in a future pipeline run, I do not want to insert it again.

    Instead, I want to update values such as:
    - amount
    - budget_amount
    - is_anomaly
    - vendor

    """

    df = df.copy()


    key_columns = ["date_key", "dept_key", "region_key", "category_key"]

    for col in key_columns:
        df[col] = df[col].astype(int)

    #
    df["amount"] = df["amount"].astype(float)
    df["budget_amount"] = df["budget_amount"].astype(float)

    df["is_anomaly"] = df["is_anomaly"].astype(bool)

    # Use the reusable upsert function.
    return upsert_dataframe(
        df=df,
        orm_class=FactFinancials,

        # If transaction_id already exists, PostgreSQL will update the row.
        conflict_columns=["transaction_id"],

        # Only these columns are allowed to change during update.
        # I do not update transaction_id because it identifies the row.
        update_columns=[
            "amount",
            "budget_amount",
            "is_anomaly",
            "vendor",
        ],
    )


def write_dim_department(df: pd.DataFrame) -> dict:
    """
    Load the department dimension table.

    dept_name is used as the conflict column because department names should be unique in this project.

    If the department already exists, I update descriptive attributes such as dept_head and cost_centre.
    """

    df = df.copy()

    return upsert_dataframe(
        df=df,
        orm_class=DimDepartment,
        conflict_columns=["dept_name"],
        update_columns=[
            "dept_head",
            "cost_centre",
        ],
    )


def write_dim_region(df: pd.DataFrame) -> dict:
    """
    Load the region dimension table.

    region_name is used as the conflict column because each region should appear once in the dimension table.

    If region details change, province and country can be updated.
    """

    df = df.copy()

    return upsert_dataframe(
        df=df,
        orm_class=DimRegion,
        conflict_columns=["region_name"],
        update_columns=[
            "province",
            "country",
        ],
    )


def write_dim_category(df: pd.DataFrame) -> dict:
    """
    Load the category dimension table.

    category_name is used as the conflict column because each financial  category should appear once.

    If the category group changes, the upsert will update it.
    """

    df = df.copy()

    return upsert_dataframe(
        df=df,
        orm_class=DimCategory,
        conflict_columns=["category_name"],
        update_columns=[
            "category_group",
        ],
    )


def get_production_counts() -> dict:
    """
    Return row counts for all production tables.

    This is a quick validation step after loading.

    It helps me answer questions like:
    - Did the load actually insert data?
    - Is any production table empty?
    - Do the row counts look reasonable?
    """

    # List of production tables that I want to check.
    tables = [
        "production.dim_date",
        "production.dim_department",
        "production.dim_region",
        "production.dim_category",
        "production.fact_financials",
    ]

    counts = {}

    # Open a database connection.
    with engine.connect() as conn:

        # Run SELECT COUNT(*) for each table.
        for table in tables:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()

            counts[table] = count

    return counts


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("── Production Table Counts ──")

    counts = get_production_counts()

    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")