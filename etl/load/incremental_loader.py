"""

This file handles incremental loading.

"""

import os
import sys
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    BigInteger,
    text,
)
from sqlalchemy.orm import declarative_base


# ---------------------------------------------------------------------------
# Make sure Python can import modules from the project root
# ---------------------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Import the PostgreSQL connection
# ---------------------------------------------------------------------------

from etl.extract.db import engine


# ---------------------------------------------------------------------------
# Define the watermark table using SQLAlchemy ORM
# ---------------------------------------------------------------------------
# This Base is only for the watermark table.
# It allows SQLAlchemy to create the table from the Python class below.
WatermarkBase = declarative_base()


class LoadWatermark(WatermarkBase):
    """
    SQLAlchemy model for the staging.load_watermarks table.

    This table stores pipeline metadata, not business data.

    It keeps one row per source, for example:
    - transactions
    - monthly_summary

    The pipeline uses this table to know where it stopped last time.
    """

    # Table name inside PostgreSQL
    __tablename__ = "load_watermarks"

    # Store this table in the staging schema because it is ETL metadata.
    __table_args__ = {"schema": "staging"}

    # Auto-increment primary key.
    # This is just a technical identifier for the watermark row.
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Name of the source being tracked.
    # unique=True means each source can only have one watermark row.
    source_name = Column(String(100), nullable=False, unique=True)

    # The last successful load timestamp.
    # The next pipeline run will only load rows newer than this value.
    last_loaded_at = Column(DateTime(timezone=True), nullable=False)

    # Number of rows loaded in the last successful run.
    # This is useful for monitoring and debugging.
    rows_loaded = Column(Integer, nullable=False, default=0)

    # Timestamp showing when this watermark row was last updated.
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


def create_watermark_table() -> None:
    """
    Create the staging.load_watermarks table if it does not already exist.

    checkfirst=True means:
    - If the table exists, do nothing.
    - If the table does not exist, create it.

    This makes the function safe to run multiple times.
    """

    WatermarkBase.metadata.create_all(engine, checkfirst=True)

    print("   Watermark table ready")


def get_watermark(source_name: str) -> datetime | None:
    """
    Get the last successful load timestamp for one source.

    Parameters
    ----------
    source_name:
        The source we want to check.
        Example: "transactions" or "monthly_summary"

    Returns
    -------
    datetime | None:
        - Returns a datetime if this source was loaded before.
        - Returns None if this is the first load for this source.

    Why return None?
    ----------------
    If there is no watermark yet, the pipeline should do a full load.
    """

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT last_loaded_at
                FROM staging.load_watermarks
                WHERE source_name = :source
                """
            ),
            {"source": source_name},
        ).fetchone()

    # If no row exists for this source, it means we have never loaded it before.
    if result is None:
        return None

    return result[0]


def set_watermark(source_name: str, rows_loaded: int) -> None:
    """
    Update the watermark after a successful load.

    This function should be called only after:
    1. Data was filtered
    2. New rows were loaded into production
    3. Data quality checks passed

    
    """

    # Get current time in UTC.
    now = datetime.now(timezone.utc)

    # engine.begin() opens a transaction.
    # If the SQL succeeds, it commits.
    # If it fails, it rolls back automatically.
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO staging.load_watermarks
                    (source_name, last_loaded_at, rows_loaded, updated_at)
                VALUES
                    (:source, :loaded_at, :rows, :updated_at)

                ON CONFLICT (source_name)
                DO UPDATE SET
                    last_loaded_at = EXCLUDED.last_loaded_at,
                    rows_loaded    = EXCLUDED.rows_loaded,
                    updated_at     = EXCLUDED.updated_at
                """
            ),
            {
                "source": source_name,
                "loaded_at": now,
                "rows": rows_loaded,
                "updated_at": now,
            },
        )

    print(
        f"   Watermark updated: {source_name} → "
        f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"({rows_loaded:,} rows)"
    )


def filter_new_records(
    df,
    timestamp_column: str,
    source_name: str,
):
    """
    Return only the records that are newer than the last watermark.

    """

   
    import pandas as pd

    # Get the last successful load time for this source.
    watermark = get_watermark(source_name)

    # If there is no watermark, this is the first load.
    # In that case, we should load all rows.
    if watermark is None:
        print(f"  No watermark found for '{source_name}' - full load")
        return df

   
    if watermark.tzinfo is not None:
        watermark = watermark.replace(tzinfo=None)

    # Convert the timestamp column to  datetime.
    # This is important because source files may read dates as strings.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # If the DataFrame timestamp column has timezone info,
    # remove it so it can be compared with the watermark.
    if df[timestamp_column].dt.tz is not None:
        df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)

    # This is the actual incremental filter.
    # Only rows newer than the watermark will continue in the pipeline.
    new_records = df[df[timestamp_column] > watermark]

    print(
        f"  Watermark for '{source_name}': "
        f"{watermark.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"  New records since watermark: "
        f"{len(new_records):,} of {len(df):,} total"
    )

    return new_records


def get_all_watermarks() -> None:
    """
    Print all current watermarks.

    This function is useful for monitoring.

    It helps me quickly answer:
    - Which sources have been loaded?
    - When was each source last loaded?
    - How many rows were loaded in the last run?
    """

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT source_name, last_loaded_at, rows_loaded
                FROM staging.load_watermarks
                ORDER BY source_name
                """
            )
        ).fetchall()

    # If the table is empty, show a clear message.
    if not rows:
        print("  No watermarks recorded yet")
        return

    print("\n── Current Watermarks ──")

    for row in rows:
        print(
            f"  {row[0]:<25} "
            f"last loaded: {row[1].strftime('%Y-%m-%d %H:%M:%S UTC')}  "
            f"rows: {row[2]:,}"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    create_watermark_table()
    get_all_watermarks()