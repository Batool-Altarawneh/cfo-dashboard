"""
This file is part of the Transform Layer in the ETL pipeline.

It takes cleaned transaction data and reshapes it from a flat table into a production-ready star schema.

Output tables:
1. dim_date
2. dim_department
3. dim_region
4. dim_category
5. fact_financials

Why this file is separate:
- cleaner.py handles data quality.
- kpi_builder.py handles business KPI logic.
- star_schema_builder.py handles database modeling.

This separation makes the pipeline easier to maintain and explain.
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert


# ---------------------------------------------------------------------------
# Make project imports work when this file is run directly
# ---------------------------------------------------------------------------
# This file lives inside etl/transform.
# To import modules from etl/extract, I add the project root folder to Python's import path.
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, project_root)


from etl.extract.db import (
    engine,
    DimDate,
    DimDepartment,
    DimRegion,
    DimCategory,
    FactFinancials,
)


# =============================================================================
# DIMENSION BUILDERS
# =============================================================================

def build_dim_date(start_year: int = 2023, end_year: int = 2027) -> pd.DataFrame:
    """
    Build a complete date dimension.

    A date dimension is useful for Power BI time intelligence, filtering, sorting months correctly, YTD calculations, and future forecasting views.

    
    """

    rows = []

    # Start from January 1 of the start year.
    current_date = date(start_year, 1, 1)

    # End at December 31 of the end year.
    end_date = date(end_year, 12, 31)

    # Loop through every day and create one row per date.
    while current_date <= end_date:

        quarter_number = (current_date.month - 1) // 3 + 1

        rows.append(
            {
                # Integer date key in YYYYMMDD format.
                # Example: 2024-03-15 becomes 20240315.
                "date_key": int(current_date.strftime("%Y%m%d")),

                # Actual date value, useful for date filters and relationships.
                "full_date": current_date,

                "year": current_date.year,
                "quarter": quarter_number,
                "quarter_name": f"Q{quarter_number}",

                # month is used for sorting.
                # month_name is used for display.
                "month": current_date.month,
                "month_name": current_date.strftime("%B"),

                # Week number based on Monday as the first day of the week.
                "week": int(current_date.strftime("%W")),

                "day_of_month": current_date.day,

                # Python weekday: Monday = 0, Sunday = 6.
                "day_of_week": current_date.weekday(),

                # Weekend flag for dashboard filtering.
                "is_weekend": current_date.weekday() >= 5,

                # Month-end flag is useful for finance reporting.
                # If tomorrow is a different month, then today is month end.
                "is_month_end": (
                    current_date + timedelta(days=1)
                ).month != current_date.month,
            }
        )

        # Move to the next calendar day.
        current_date += timedelta(days=1)

    df_date = pd.DataFrame(rows)

    print(
        f"   dim_date built: {len(df_date):,} rows "
        f"({start_year}–{end_year})"
    )

    return df_date


def build_dim_department() -> pd.DataFrame:
    """
    Build the department dimension.

    This is static reference data for the project.
    """

    department_rows = [
        {
            "dept_name": "IT",
            "dept_head": "Sarah Chen",
            "cost_centre": "CC-001",
        },
        {
            "dept_name": "Marketing",
            "dept_head": "James Okafor",
            "cost_centre": "CC-002",
        },
        {
            "dept_name": "Sales",
            "dept_head": "Maria Tremblay",
            "cost_centre": "CC-003",
        },
        {
            "dept_name": "HR",
            "dept_head": "David Park",
            "cost_centre": "CC-004",
        },
        {
            "dept_name": "Operations",
            "dept_head": "Aisha Patel",
            "cost_centre": "CC-005",
        },
    ]

    df_department = pd.DataFrame(department_rows)

    print(f"   dim_department built: {len(df_department)} rows")

    return df_department


def build_dim_region() -> pd.DataFrame:
    """
    Build the region dimension.

    This adds Canadian geographic context to the dashboard.
    """

    region_rows = [
        {
            "region_name": "East",
            "province": "Ontario / Quebec",
            "country": "Canada",
        },
        {
            "region_name": "West",
            "province": "British Columbia / Alberta",
            "country": "Canada",
        },
        {
            "region_name": "Central",
            "province": "Manitoba / Saskatchewan",
            "country": "Canada",
        },
    ]

    df_region = pd.DataFrame(region_rows)

    print(f"   dim_region built: {len(df_region)} rows")

    return df_region


def build_dim_category(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the category dimension from transaction data.

    I derive categories from the cleaned transactions instead of hardcoding them.
    This makes the pipeline more flexible if new categories appear later.

   
    """

    unique_categories = (
        df_transactions["category"]
        .dropna()
        .unique()
    )

    def assign_category_group(category_name: str) -> str:
        """
        Assign each category to a higher-level reporting group.

        This helps the dashboard summarize categories into broader groups:
        Revenue, Capital, or Operating.
        """

        revenue_categories = {"Revenue"}
        capital_categories = {"Hardware", "Cloud Infrastructure"}

        if category_name in revenue_categories:
            return "Revenue"

        if category_name in capital_categories:
            return "Capital"

        return "Operating"

    category_rows = []

    for category_name in sorted(unique_categories):
        category_rows.append(
            {
                "category_name": category_name,
                "category_group": assign_category_group(category_name),
            }
        )

    df_category = pd.DataFrame(category_rows)

    print(f"   dim_category built: {len(df_category)} rows")

    return df_category


# =============================================================================
# DIMENSION LOADER
# =============================================================================

def load_dimension(
    df: pd.DataFrame,
    orm_class,
    natural_key_column: str,
    surrogate_key_column: str,
) -> dict:
    """
    Load a dimension table into PostgreSQL and return a key mapping.

    This function is reusable for all dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Dimension DataFrame to load.

    orm_class
        SQLAlchemy ORM class for the target dimension table.

    natural_key_column : str
        Business-readable unique column.
        Example: dept_name, region_name, category_name.

    surrogate_key_column : str
        Integer primary key column generated/stored in the database.
        Example: dept_key, region_key, category_key.

    Returns
    -------
    dict
        Mapping from natural key to surrogate key.
        Example: {"IT": 1, "Sales": 2}
    """

    records = df.to_dict(orient="records")
    inserted_count = 0

    # engine.begin() opens a transaction.
    # If something fails, SQLAlchemy can roll back the transaction.
    with engine.begin() as conn:

        for record in records:

            # PostgreSQL insert statement using SQLAlchemy.
            statement = pg_insert(orm_class).values(**record)

            # ON CONFLICT DO NOTHING makes the load safe to rerun.
            # If the row already exists because of a unique constraint,
            # PostgreSQL skips it instead of raising an error.
            statement = statement.on_conflict_do_nothing()

            result = conn.execute(statement)

            if result.rowcount > 0:
                inserted_count += 1

    # Read the full dimension back from the database.
    # I do this because the database owns the surrogate keys.
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM production.{orm_class.__tablename__}")
        ).fetchall()

    # Build a Python dictionary that maps natural keys to integer keys.
    # The fact table will use this mapping to replace text values with FKs.
    key_mapping = {
        row._mapping[natural_key_column]: row._mapping[surrogate_key_column]
        for row in rows
    }

    print(
        f"   production.{orm_class.__tablename__}: "
        f"{inserted_count} inserted | mapping: {len(key_mapping)} keys"
    )

    return key_mapping


# =============================================================================
# FACT TABLE BUILDER AND LOADER
# =============================================================================

def build_and_load_fact(
    df_transactions: pd.DataFrame,
    dept_map: dict,
    region_map: dict,
    category_map: dict,
) -> int:
    """
    Build and load the fact_financials table.

    The fact table stores measurable transaction values and foreign keys
    to dimensions.

    Parameters
    ----------
    df_transactions : pd.DataFrame
        Cleaned transaction-level data.

    dept_map : dict
        Mapping from department name to dept_key.

    region_map : dict
        Mapping from region name to region_key.

    category_map : dict
        Mapping from category name to category_key.

    Returns
    -------
    int
        Number of inserted fact rows.
    """

    df = df_transactions.copy()

    # -----------------------------------------------------------------------
    # Step 1: Build date_key
    # -----------------------------------------------------------------------
    # The date dimension uses YYYYMMDD integer keys.
    # I create the same key from the transaction date so the fact table
    # can join to dim_date.
    df["date_key"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    # -----------------------------------------------------------------------
    # Step 2: Replace text values with foreign keys
    # -----------------------------------------------------------------------
    # The staging data has text values like "IT" and "East".
    # The production fact table should store integer keys instead.
    df["dept_key"] = df["department"].map(dept_map)
    df["region_key"] = df["region"].map(region_map)
    df["category_key"] = df["category"].map(category_map)

    # -----------------------------------------------------------------------
    # Step 3: Check for unmapped keys
    # -----------------------------------------------------------------------
    # If a value is not found in the dimension mapping, map() returns NaN.
    # Those rows cannot be loaded into the fact table because they would
    # violate foreign key constraints.
    foreign_key_columns = [
        "dept_key",
        "region_key",
        "category_key",
    ]

    unmapped_rows = df[df[foreign_key_columns].isna().any(axis=1)]

    if len(unmapped_rows) > 0:
        print(
            f"  {len(unmapped_rows):,} rows have unmapped FK values "
            f"and will be skipped"
        )

        df = df.dropna(subset=foreign_key_columns)

    # -----------------------------------------------------------------------
    # Step 4: Select fact table columns
    # -----------------------------------------------------------------------
    # The fact table keeps numeric measures and foreign keys.
    # It does not keep repeated text columns like department or region.
    fact_columns = [
        "transaction_id",
        "date_key",
        "dept_key",
        "region_key",
        "category_key",
        "amount",
        "budget_amount",
        "transaction_type",
        "is_anomaly",
        "vendor",
    ]

    df_fact = df[fact_columns].copy()

    # Foreign keys must be integers before loading to PostgreSQL.
    for column in ["date_key", "dept_key", "region_key", "category_key"]:
        df_fact[column] = df_fact[column].astype(int)

    # -----------------------------------------------------------------------
    # Step 5: Load fact rows into PostgreSQL
    # -----------------------------------------------------------------------
    records = df_fact.to_dict(orient="records")

    inserted_count = 0
    skipped_count = 0

    with engine.begin() as conn:

        for record in records:

            statement = pg_insert(FactFinancials).values(**record)

            # The unique constraint prevents duplicate transactions.
            # If the same transaction_id is loaded again, PostgreSQL skips it.
            statement = statement.on_conflict_do_nothing(
                constraint="uq_fact_transaction"
            )

            result = conn.execute(statement)

            if result.rowcount > 0:
                inserted_count += 1
            else:
                skipped_count += 1

    print(
        f"  production.fact_financials: "
        f"{inserted_count:,} inserted | {skipped_count:,} skipped"
    )

    return inserted_count


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_star_schema_build(df_transactions: pd.DataFrame) -> None:
    """
    Run the full star schema build.

    Correct order:
    1. Build dimension DataFrames.
    2. Load dimension tables.
    3. Get surrogate key mappings.
    4. Build and load the fact table.

    Dimensions must be loaded before the fact table because the fact table
    depends on their primary keys through foreign key relationships.
    """

    print("\n── Building Dimensions ──")

    df_date = build_dim_date()
    df_department = build_dim_department()
    df_region = build_dim_region()
    df_category = build_dim_category(df_transactions)

    print("\n── Loading Dimensions to Production ──")

    # Date dimension does not need a mapping for the fact table here because
    # date_key is calculated directly from the transaction date.
    load_dimension(
        df=df_date,
        orm_class=DimDate,
        natural_key_column="full_date",
        surrogate_key_column="date_key",
    )

    dept_map = load_dimension(
        df=df_department,
        orm_class=DimDepartment,
        natural_key_column="dept_name",
        surrogate_key_column="dept_key",
    )

    region_map = load_dimension(
        df=df_region,
        orm_class=DimRegion,
        natural_key_column="region_name",
        surrogate_key_column="region_key",
    )

    category_map = load_dimension(
        df=df_category,
        orm_class=DimCategory,
        natural_key_column="category_name",
        surrogate_key_column="category_key",
    )

    print("\n── Building and Loading Fact Table ──")

    build_and_load_fact(
        df_transactions=df_transactions,
        dept_map=dept_map,
        region_map=region_map,
        category_map=category_map,
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    This block runs only when I execute this file directly.


    It gives me a quick way to test:
    1. reading transactions
    2. cleaning transactions
    3. building dimensions
    4. loading the fact table
    5. checking production row counts
    """

    from etl.extract.source_connector import read_source_file
    from etl.transform.cleaner import clean_transactions

    print("=" * 55)
    print("  Running Star Schema Builder")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 1: Read raw transaction data
    # -----------------------------------------------------------------------
    transactions_raw, metadata = read_source_file(
        "data/raw/transactions.xlsx"
    )

    # -----------------------------------------------------------------------
    # Step 2: Clean transaction data before building star schema
    # -----------------------------------------------------------------------
    transactions_clean = clean_transactions(transactions_raw)

    # -----------------------------------------------------------------------
    # Step 3: Build and load star schema tables
    # -----------------------------------------------------------------------
    run_star_schema_build(transactions_clean)

    # -----------------------------------------------------------------------
    # Step 4: Verify production row counts
    # -----------------------------------------------------------------------
    # This is a simple sanity check to confirm that tables were populated.
    print("\n── Production Row Counts ──")

    production_tables = [
        "production.dim_date",
        "production.dim_department",
        "production.dim_region",
        "production.dim_category",
        "production.fact_financials",
    ]

    with engine.connect() as conn:

        for table_name in production_tables:
            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()

            print(f"  {table_name}: {row_count:,} rows")