"""
This file tells Apache Airflow how to run my full ETL pipeline.

Airflow does not replace my ETL code. It only orchestrates it.

That means Airflow is responsible for:
- running tasks in the right order
- scheduling the pipeline
- retrying failed tasks
- showing logs and task status in the UI
- stopping the pipeline if an important step fails

Pipeline order:

    extract_transactions & extract_monthly_summary -> clean_and_transform -> build_star_schema -> run_quality_checks -> update_watermarks

The two extract tasks can run at the same time because they do not depend on each other.
The clean_and_transform task waits until both extracts finish.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# ---------------------------------------------------------------------------
# Make project code importable inside the Airflow container
# ---------------------------------------------------------------------------
#


PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
#
# These paths point to the raw Excel files inside the Airflow container.
#
# I build them from PROJECT_ROOT instead of hardcoding the full path everywhere.

TRANSACTIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "transactions.xlsx"
)

MONTHLY_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "monthly_summary.xlsx"
)


# ---------------------------------------------------------------------------
# Default task settings
# ---------------------------------------------------------------------------
#
# These settings apply to every task in this DAG unless a task overrides them.

default_args = {
    # Owner name shown in the Airflow UI.
    "owner": "cfo_dashboard",

    # If True, today's task would depend on yesterday's task result.
    # I keep it False because each run should be independent.
    "depends_on_past": False,

    # Email alerts are disabled for local development.
    "email_on_failure": False,
    "email_on_retry": False,

    # If a task fails, Airflow will retry it 2 times.
    # This helps with temporary issues like database/network delays.
    "retries": 2,

    # Wait 5 minutes between retries.
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Task 0 — Ensure all database tables exist
# ---------------------------------------------------------------------------

def task_ensure_tables(**context):
    """
    Create all staging and production tables if they do not exist.

    This runs before every pipeline execution.
    checkfirst=True means existing tables are never dropped or overwritten.

    I include this in the pipeline because if the database volume is ever wiped (docker compose down -v),
    the next pipeline run recreates everything automatically instead of failing with 'relation does not exist'.
    """
    from etl.extract.db import create_all_tables
    create_all_tables()

# ---------------------------------------------------------------------------
# Task 1 — Extract transactions
# ---------------------------------------------------------------------------

def task_extract_transactions(**context):
    """
    Read transactions.xlsx, validate its structure, and load it into staging.

    This is the raw landing step for transaction-level data.

    If validation fails, I raise an error.
    Airflow will mark the task as failed and stop the downstream tasks.
    """

    from etl.extract.source_connector import read_source_file
    from etl.extract.schema_validator import validate_file
    from etl.extract.raw_loader import load_transactions

    # Read source file and collect metadata like filename/load timestamp.
    df, metadata = read_source_file(TRANSACTIONS_FILE)

    # Validate expected columns and basic file structure.
    result = validate_file(df, "transactions")

    # If the file is not valid, stop the pipeline early.
    if not result.is_valid:
        raise ValueError(
            f"Transaction validation failed:\n{result.summary()}"
        )

    # Load valid raw data into staging.raw_transactions.
    stats = load_transactions(df, metadata)

    print(f"Transactions loaded successfully: {stats}")


# ---------------------------------------------------------------------------
# Task 2 — Extract monthly summary
# ---------------------------------------------------------------------------

def task_extract_monthly_summary(**context):
    """
    Read monthly_summary.xlsx, validate it, and load it into staging.

    This task is separate from transactions extraction because each source file
    should have its own validation and loading step.
    """

    from etl.extract.source_connector import read_source_file
    from etl.extract.schema_validator import validate_file
    from etl.extract.raw_loader import load_monthly_summary

    df, metadata = read_source_file(MONTHLY_SUMMARY_FILE)

    result = validate_file(df, "monthly_summary")

    if not result.is_valid:
        raise ValueError(
            f"Monthly summary validation failed:\n{result.summary()}"
        )

    stats = load_monthly_summary(df, metadata)

    print(f"Monthly summary loaded successfully: {stats}")


# ---------------------------------------------------------------------------
# Task 3 — Clean and transform
# ---------------------------------------------------------------------------

def task_clean_and_transform(**context):
    """
    Clean the raw source files.

    read the raw Excel files again and apply the cleaning functions.

    The goal of this task is to prove that the transformation logic works after the extract tasks have completed successfully.

    
    """

    from etl.extract.source_connector import read_source_file
    from etl.transform.cleaner import clean_transactions, clean_monthly_summary

    df_transactions_raw, _ = read_source_file(TRANSACTIONS_FILE)
    df_summary_raw, _ = read_source_file(MONTHLY_SUMMARY_FILE)

    df_transactions_clean = clean_transactions(df_transactions_raw)
    df_summary_clean = clean_monthly_summary(df_summary_raw)

    print(f"Transactions cleaned: {len(df_transactions_clean):,} rows")
    print(f"Monthly summary cleaned: {len(df_summary_clean):,} rows")


# ---------------------------------------------------------------------------
# Task 4 — Build star schema
# ---------------------------------------------------------------------------

def task_build_star_schema(**context):
    """
    Build production dimension and fact tables.

    This task creates the reporting-ready star schema used by Power BI.

    It depends on clean_and_transform, so Airflow will not run this task unless the cleaning step succeeds.
    """

    from etl.extract.source_connector import read_source_file
    from etl.transform.cleaner import clean_transactions
    from etl.transform.star_schema_builder import run_star_schema_build

    df_raw, _ = read_source_file(TRANSACTIONS_FILE)

    df_clean = clean_transactions(df_raw)

    run_star_schema_build(df_clean)

    print("Star schema build completed successfully.")


# ---------------------------------------------------------------------------
# Task 5 — Run data quality checks
# ---------------------------------------------------------------------------

def task_run_quality_checks(**context):
    """
    Run data quality checks on production data.

    This is a very important step.

    If data quality checks fail, the task raises an exception.
    That makes Airflow mark the task as failed, and the pipeline stops.

    This prevents bad data from silently reaching the dashboard.
    """

    from etl.load.data_quality import run_quality_checks

    result = run_quality_checks()

    if not result.passed:
        failed_checks = [check.check_name for check in result.failed_checks]

        raise ValueError(
            f"Data quality checks failed: {failed_checks}\n"
            f"{result.summary()}"
        )

    print("All data quality checks passed.")


# ---------------------------------------------------------------------------
# Task 6 — Update watermarks
# ---------------------------------------------------------------------------

def task_update_watermarks(**context):
    """
    Update load watermarks after the pipeline succeeds.

    A watermark records what has already been loaded.

    This is useful for incremental loading because the next run can know where the previous successful run stopped.

    This task should only run after all previous tasks succeed.
    """

    from etl.load.incremental_loader import (
        create_watermark_table,
        set_watermark,
    )

    create_watermark_table()

    # These values are hardcoded for the current sample dataset.
    # Later, these could come from actual row counts returned by the loaders.
    set_watermark("transactions", 8_279)
    set_watermark("monthly_summary", 540)

    print("Watermarks updated successfully.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
#
# This block creates the Airflow DAG object.
#
# dag_id:
#   The name that will appear in the Airflow UI.
#
# start_date:
#   Airflow needs this to know when scheduling begins.
#
# schedule_interval:
#   '0 6 * * *' means run every day at 6:00 AM UTC.
#
# catchup=False:
#   If Airflow was off for several days, do not run all missed historical runs.
#
# tags:
#   Labels that make the DAG easier to find in the Airflow UI.

with DAG(
    dag_id="cfo_etl_pipeline",
    description="CFO Dashboard ETL - extract, transform, load, validate",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * *",
    catchup=False,
    tags=["cfo", "etl", "financial"],
) as dag:

    # -----------------------------------------------------------------------
    # Define Airflow tasks
    # -----------------------------------------------------------------------
    #
    # PythonOperator means:
    #   Airflow will run a Python function as a task.
    #
    # task_id:
    #   Name shown in Airflow UI.
    #
    # python_callable:
    #   The Python function Airflow should execute.
    ensure_tables = PythonOperator(
        task_id="ensure_tables",
        python_callable=task_ensure_tables,
    )

    extract_transactions = PythonOperator(
        task_id="extract_transactions",
        python_callable=task_extract_transactions,
    )

    extract_monthly_summary = PythonOperator(
        task_id="extract_monthly_summary",
        python_callable=task_extract_monthly_summary,
    )

    clean_and_transform = PythonOperator(
        task_id="clean_and_transform",
        python_callable=task_clean_and_transform,
    )

    build_star_schema = PythonOperator(
        task_id="build_star_schema",
        python_callable=task_build_star_schema,
    )

    run_quality_checks = PythonOperator(
        task_id="run_quality_checks",
        python_callable=task_run_quality_checks,
    )

    update_watermarks = PythonOperator(
        task_id="update_watermarks",
        python_callable=task_update_watermarks,
    )

    # -----------------------------------------------------------------------
    # Define task dependencies
    # -----------------------------------------------------------------------
    #
    # The >> operator means:
    #   "run the task on the left before the task on the right"
    #
    # These two extract tasks can run in parallel:
    #   extract_transactions
    #   extract_monthly_summary
    #
    # clean_and_transform waits for both extract tasks to succeed.

    ensure_tables >> [extract_transactions, extract_monthly_summary]
    [extract_transactions, extract_monthly_summary] >> clean_and_transform
    clean_and_transform >> build_star_schema
    build_star_schema >> run_quality_checks
    run_quality_checks >> update_watermarks