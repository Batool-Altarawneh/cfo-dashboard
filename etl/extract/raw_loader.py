"""
Raw Loader - ETL Extract Layer

This module is responsible for loading validated raw DataFrames into the
PostgreSQL staging schema.

At this point in the pipeline:
1. source_connector.py already read the source file.
2. schema_validator.py already checked that the file is safe to load.
3. raw_loader.py writes the validated data into staging tables.

This separation keeps the pipeline easier to understand, test, and debug.
"""

import os
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

# import from db.py file
from etl.extract.db import engine, RawTransaction, RawMonthlySummary


def load_transactions(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Load validated transaction rows into staging.raw_transactions.

    This function uses PostgreSQL ON CONFLICT DO NOTHING.

    If the same file is loaded again, we do not want duplicate rows.
    Instead, PostgreSQL skips records that already exist based on the unique
    constraint uq_txn_source.

    This makes the load step idempotent, meaning it is safe to run more than
    once and still get the same final result.
    """

    # Work on a copy so we do not change the original DataFrame.
    df = df.copy()

    # Normalize column names to match the database column names.
    df.columns = df.columns.str.lower().str.strip()

    # Add source metadata to every row.
    # This allows us to trace each database record back to the file it came from.
    df["source_file"] = metadata["source_file"]
    df["loaded_at"] = metadata["loaded_at"]

    # Convert columns to the correct database-friendly types.
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["budget_amount"] = df["budget_amount"].astype(float)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)

    # Convert the DataFrame into a list of dictionaries.
    # Each dictionary represents one row to insert into PostgreSQL.
    records = df.to_dict(orient="records")

#counters
    inserted = 0
    skipped = 0

    # engine.begin() opens a database transaction.
    # If everything succeeds, it commits automatically.
    # If an error happens, it rolls back automatically.
    with engine.begin() as conn:
        for record in records:
            # Build an INSERT statement for one transaction row.
            stmt = pg_insert(RawTransaction).values(**record)

            # If the row already exists, skip it instead of failing.
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_txn_source"
            )

            result = conn.execute(stmt)

            # rowcount tells us whether PostgreSQL inserted the row or skipped it.
            if result.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

    stats = {
        "table": "staging.raw_transactions",
        "inserted": inserted,
        "skipped": skipped,
        "total": len(records),
    }

    print(
        f"staging.raw_transactions → "
        f"{inserted:,} inserted | {skipped:,} skipped"
    )

    return stats


def load_monthly_summary(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Load validated monthly summary rows into staging.raw_monthly_summary.

    This uses the same idempotent upsert pattern as transactions.
    """

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Add metadata columns for lineage.
    df["source_file"] = metadata["source_file"]
    df["loaded_at"] = metadata["loaded_at"]

    # Convert columns to database-friendly types.
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["total_revenue"] = df["total_revenue"].astype(float)
    df["total_expense"] = df["total_expense"].astype(float)
    df["total_budget"] = df["total_budget"].astype(float)
    df["headcount"] = df["headcount"].astype(int)

    records = df.to_dict(orient="records")

    inserted = 0
    skipped = 0

    with engine.begin() as conn:
        for record in records:
            stmt = pg_insert(RawMonthlySummary).values(**record)

            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_summary_source"
            )

            result = conn.execute(stmt)

            if result.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

    stats = {
        "table": "staging.raw_monthly_summary",
        "inserted": inserted,
        "skipped": skipped,
        "total": len(records),
    }

    print(
        f" staging.raw_monthly_summary → "
        f"{inserted:,} inserted | {skipped:,} skipped"
    )

    return stats


def get_staging_counts() -> dict:
    """
    Return row counts from the staging tables.

    This is a simple verification step after loading data.
    """

    with engine.connect() as conn:
        transaction_count = conn.execute(
            text("SELECT COUNT(*) FROM staging.raw_transactions")
        ).scalar()

        monthly_summary_count = conn.execute(
            text("SELECT COUNT(*) FROM staging.raw_monthly_summary")
        ).scalar()

    return {
        "staging.raw_transactions": transaction_count,
        "staging.raw_monthly_summary": monthly_summary_count,
    }


if __name__ == "__main__":
    """
    Run the full extract layer from the terminal.

    This script:
    1. Reads each raw file.
    2. Validates the file.
    3. Loads it to staging only if validation passes.
    4. Prints staging row counts at the end.
    """

    from etl.extract.source_connector import read_source_file
    from etl.extract.schema_validator import validate_file

    print("=" * 55)
    print("  Running Extract Layer")
    print("=" * 55)

    files = [
        ("data/raw/transactions.xlsx", "transactions"),
        ("data/raw/monthly_summary.xlsx", "monthly_summary"),
    ]

    for file_path, file_type in files:
        print(f"\n── {file_type.upper()} ──")

        # Step 1: Read the source file.
        df, metadata = read_source_file(file_path)

        # Step 2: Validate the raw DataFrame.
        validation_result = validate_file(df, file_type)

        print(validation_result.summary())

        # Step 3: Stop before loading if validation failed.
        if not validation_result.is_valid:
            print(f" Skipping load for {file_path} because validation failed")
            continue

        # Step 4: Load the validated DataFrame into the correct staging table.
        if file_type == "transactions":
            load_transactions(df, metadata)
        else:
            load_monthly_summary(df, metadata)

    # Step 5: Print final staging table counts.
    print("\n── Staging Row Counts ──")

    counts = get_staging_counts()

    for table_name, row_count in counts.items():
        print(f"  {table_name}: {row_count:,} rows")