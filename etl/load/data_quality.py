"""

This file runs data quality checks on the production PostgreSQL tables.

At this point in the pipeline:
1. Raw data was extracted.
2. Data was cleaned and transformed.
3. Star schema tables were created.
4. Data was loaded into production tables.

Now we need to verify that the production tables are safe for reporting.


This file helps the pipeline fail early before bad data reaches the dashboard.

Types of checks included
------------------------
1. Row count checks
   Make sure tables are not empty and have expected row counts.

2. Null checks
   Make sure important columns like IDs and keys are not null.

3. Value range checks
   Make sure numeric values are within expected business ranges.

4. Allowed values checks
   Make sure categorical columns only contain valid values.

5. Referential integrity checks
   Make sure fact table keys exist in the related dimension tables.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text


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
# Data classes for storing quality check results
# ---------------------------------------------------------------------------

@dataclass
class QualityCheckResult:
    """
    Stores the result of one data quality check.

    Example:
    A check like "amount should not be null" will return one QualityCheckResult object.

    """

    passed: bool
    check_name: str
    message: str


@dataclass
class QualitySuiteResult:
    """
    Stores the results of all quality checks.

    """

    # default_factory=list creates a new empty list for each object.
    # This is safer than using checks=[] as a default value.
    checks: list[QualityCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """
        Return True only if every check passed.

        If even one check failed, the whole suite fails.
        """

        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[QualityCheckResult]:
        """
        Return only the checks that failed.

        This makes it easier to print a clean failure report.
        """

        return [check for check in self.checks if not check.passed]

    def add(self, result: QualityCheckResult) -> None:
        """
        Add one check result to the suite.
        """

        self.checks.append(result)

    def summary(self) -> str:
        """
        Build a readable summary report for all checks.

        This report is printed at the end of the quality process.
        """

        total = len(self.checks)
        passed = sum(1 for check in self.checks if check.passed)
        failed = total - passed

        lines = [
            "\nData Quality Report",
            "─" * 40,
            f"Checks run : {total}",
            f"Passed     : {passed}",
            f"Failed     : {failed}",
            f"Result     : {' ALL PASSED' if self.passed else ' FAILURES DETECTED'}",
        ]

        # If any checks failed, print their names and messages.
        if self.failed_checks:
            lines.append("\nFailed checks:")

            for check in self.failed_checks:
                lines.append(
                    f"   {check.check_name}: {check.message}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual quality check functions
# ---------------------------------------------------------------------------

def check_row_count(
    table: str,
    min_expected: int,
    max_expected: Optional[int] = None,
) -> QualityCheckResult:
    """
    Check whether a table has an expected number of rows.

    Parameters
    ----------
    table:
        Full table name, including schema.
        Example: "production.fact_financials"

    min_expected:
        Minimum acceptable number of rows.

    max_expected:
        Optional maximum acceptable number of rows.
        This is useful for small dimension tables where I know the exact count.

    Why this check matters
    ----------------------
    If a table suddenly has too few rows, it may mean:
    - the load failed
    - a filter removed too much data
    - the incremental logic skipped records
    - the source file was incomplete
    """

    with engine.connect() as conn:
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()

    # Fail if the table has fewer rows than expected.
    if count < min_expected:
        return QualityCheckResult(
            passed=False,
            check_name=f"row_count:{table}",
            message=(
                f"Expected at least {min_expected:,} rows, "
                f"found {count:,}"
            ),
        )

    # If max_expected was provided, also fail if the table has too many rows.
    # This can catch duplicate dimension records.
    if max_expected is not None and count > max_expected:
        return QualityCheckResult(
            passed=False,
            check_name=f"row_count:{table}",
            message=(
                f"Expected at most {max_expected:,} rows, "
                f"found {count:,}"
            ),
        )

    # If both checks pass, return a successful result.
    return QualityCheckResult(
        passed=True,
        check_name=f"row_count:{table}",
        message=f"{count:,} rows — within expected range",
    )


def check_no_nulls(table: str, column: str) -> QualityCheckResult:
    """
    Check that a column does not contain null values.

    Parameters
    ----------
    table:
        Full table name.
        Example: "production.fact_financials"

    column:
        Column that should not contain nulls.
        Example: "transaction_id"

    Why this check matters
    ----------------------
    Some columns are critical for reporting.

    For example:
    - transaction_id identifies each row
    - amount is needed for financial calculations
    - dept_key links the fact table to the department dimension
    - date_key links the fact table to the date dimension

    If these columns contain nulls, dashboard numbers or relationships
    may be incorrect.
    """

    with engine.connect() as conn:
        null_count = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {table}
                WHERE {column} IS NULL
                """
            )
        ).scalar()

    if null_count > 0:
        return QualityCheckResult(
            passed=False,
            check_name=f"no_nulls:{table}.{column}",
            message=f"{null_count:,} null values found in {column}",
        )

    return QualityCheckResult(
        passed=True,
        check_name=f"no_nulls:{table}.{column}",
        message=f"No nulls in {column}",
    )


def check_value_range(
    table: str,
    column: str,
    min_val: float,
    max_val: Optional[float] = None,
) -> QualityCheckResult:
    """
    Check that numeric values are within an expected range.

    Parameters
    ----------
    table:
        Full table name.

    column:
        Numeric column to check.

    min_val:
        Minimum allowed value.

    max_val:
        Optional maximum allowed value.

    Why this check matters
    ----------------------
    Financial values should follow business rules.

    For example:
    - amount should usually be greater than 0
    - budget_amount should usually be greater than 0
    - percentage columns may need to stay between 0 and 100
    """

    with engine.connect() as conn:
        result = conn.execute(
            text(
                f"""
                SELECT MIN({column}), MAX({column})
                FROM {table}
                """
            )
        ).fetchone()

    col_min, col_max = result[0], result[1]

    # If MIN returns None, it usually means the column is completely null
    # or the table has no rows.
    if col_min is None:
        return QualityCheckResult(
            passed=False,
            check_name=f"value_range:{table}.{column}",
            message="Column is entirely null - no values to check",
        )

    # Fail if the minimum value is below the accepted business rule.
    if col_min < min_val:
        return QualityCheckResult(
            passed=False,
            check_name=f"value_range:{table}.{column}",
            message=(
                f"Minimum value {col_min:.2f} is below "
                f"expected minimum {min_val}"
            ),
        )

    # If max_val is provided, fail if the maximum value is too high.
    if max_val is not None and col_max > max_val:
        return QualityCheckResult(
            passed=False,
            check_name=f"value_range:{table}.{column}",
            message=(
                f"Maximum value {col_max:.2f} exceeds "
                f"expected maximum {max_val}"
            ),
        )

    return QualityCheckResult(
        passed=True,
        check_name=f"value_range:{table}.{column}",
        message=f"Values in range [{col_min:.2f}, {col_max:.2f}]",
    )


def check_allowed_values(
    table: str,
    column: str,
    allowed_values: list,
) -> QualityCheckResult:
    """
    Check that a column only contains expected values.

    Parameters
    ----------
    table:
        Full table name.

    column:
        Column to check.

    allowed_values:
        List of valid values.

    Example
    -------
    transaction_type should only be:
    - "EXPENSE"
    - "REVENUE"

    Why this check matters
    ----------------------
    If unexpected categories appear, calculations and filters in Power BI may become wrong or confusing.
    """

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT DISTINCT {column}
                FROM {table}
                """
            )
        ).fetchall()

    # Convert database values into a Python set.
    # Nulls are ignored here because nulls should be checked separately
    # using check_no_nulls if the column is required.
    actual_values = {row[0] for row in rows if row[0] is not None}

    # Find values that exist in the table but are not in the allowed list.
    unexpected = actual_values - set(allowed_values)

    if unexpected:
        return QualityCheckResult(
            passed=False,
            check_name=f"allowed_values:{table}.{column}",
            message=f"Unexpected values found: {unexpected}",
        )

    return QualityCheckResult(
        passed=True,
        check_name=f"allowed_values:{table}.{column}",
        message="All values in allowed set",
    )


def check_referential_integrity(
    fact_table: str,
    fact_column: str,
    dim_table: str,
    dim_column: str,
) -> QualityCheckResult:
    """
    Check that every foreign key in the fact table exists in the dimension table.

    Parameters
    ----------
    fact_table:
        Fact table name.
        Example: "production.fact_financials"

    fact_column:
        Foreign key column in the fact table.
        Example: "dept_key"

    dim_table:
        Dimension table name.
        Example: "production.dim_department"

    dim_column:
        Primary key column in the dimension table.
        Example: "dept_key"

    Why this check matters
    ----------------------
    In a star schema, fact tables connect to dimension tables using keys.

    If a fact row has a key that does not exist in the dimension table,
    the row becomes "orphaned".

    Example:
    fact_financials has dept_key = 99,
    but dim_department does not have dept_key = 99.

    This can create blank categories or incorrect totals in Power BI.
    """

    with engine.connect() as conn:
        orphaned = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {fact_table} f
                LEFT JOIN {dim_table} d
                    ON f.{fact_column} = d.{dim_column}
                WHERE d.{dim_column} IS NULL
                """
            )
        ).scalar()

    if orphaned > 0:
        return QualityCheckResult(
            passed=False,
            check_name=f"referential_integrity:{fact_column}",
            message=(
                f"{orphaned:,} rows in {fact_table}.{fact_column} "
                f"have no match in {dim_table}.{dim_column}"
            ),
        )

    return QualityCheckResult(
        passed=True,
        check_name=f"referential_integrity:{fact_column}",
        message=f"All {fact_column} values found in {dim_table}",
    )


# ---------------------------------------------------------------------------
# Full quality suite
# ---------------------------------------------------------------------------

def run_quality_checks() -> QualitySuiteResult:
    """
    Run all production data quality checks for the CFO dashboard.

    This function defines what "good data" means for this project.

    If any check fails, the pipeline should stop before the dashboard
    refreshes with incorrect data.
    """

    suite = QualitySuiteResult()

    print("Running production data quality checks...")

    # -----------------------------------------------------------------------
    # Row count checks
    # -----------------------------------------------------------------------
    # These checks confirm that the main production tables have the expected
    # amount of data after the load.
    #
    # If these fail, it may mean the load failed, incremental filtering skipped
    # too much data, or the table has duplicate/missing records.
    suite.add(
        check_row_count(
            "production.fact_financials",
            min_expected=8_000,
        )
    )

    suite.add(
        check_row_count(
            "production.dim_department",
            min_expected=5,
            max_expected=5,
        )
    )

    suite.add(
        check_row_count(
            "production.dim_region",
            min_expected=3,
            max_expected=3,
        )
    )

    suite.add(
        check_row_count(
            "production.dim_date",
            min_expected=1_800,
        )
    )

    suite.add(
        check_row_count(
            "production.dim_category",
            min_expected=10,
        )
    )

    # -----------------------------------------------------------------------
    # Null checks on critical columns
    # -----------------------------------------------------------------------
    # These columns are required for correct reporting and relationships.
    # Nulls here usually mean something went wrong during transform or load.
    suite.add(check_no_nulls("production.fact_financials", "transaction_id"))
    suite.add(check_no_nulls("production.fact_financials", "amount"))
    suite.add(check_no_nulls("production.fact_financials", "dept_key"))
    suite.add(check_no_nulls("production.fact_financials", "region_key"))
    suite.add(check_no_nulls("production.fact_financials", "date_key"))

    suite.add(check_no_nulls("production.dim_department", "dept_name"))
    suite.add(check_no_nulls("production.dim_region", "region_name"))

    # -----------------------------------------------------------------------
    # Value range checks
    # -----------------------------------------------------------------------
    # For this project, amount and budget_amount should be positive.
    # Negative or zero values may indicate a source issue or parsing problem.
    suite.add(
        check_value_range(
            "production.fact_financials",
            "amount",
            min_val=0.01,
        )
    )

    suite.add(
        check_value_range(
            "production.fact_financials",
            "budget_amount",
            min_val=0.01,
        )
    )

    # -----------------------------------------------------------------------
    # Allowed values checks
    # -----------------------------------------------------------------------
    # transaction_type should only contain the two categories used in the
    # financial model.
    suite.add(
        check_allowed_values(
            "production.fact_financials",
            "transaction_type",
            ["EXPENSE", "REVENUE"],
        )
    )

    # -----------------------------------------------------------------------
    # Referential integrity checks
    # -----------------------------------------------------------------------
    # These checks confirm that fact table keys correctly match dimension keys.
    # This protects the star schema relationships used by Power BI.
    suite.add(
        check_referential_integrity(
            "production.fact_financials",
            "dept_key",
            "production.dim_department",
            "dept_key",
        )
    )

    suite.add(
        check_referential_integrity(
            "production.fact_financials",
            "region_key",
            "production.dim_region",
            "region_key",
        )
    )

    suite.add(
        check_referential_integrity(
            "production.fact_financials",
            "category_key",
            "production.dim_category",
            "category_key",
        )
    )

    suite.add(
        check_referential_integrity(
            "production.fact_financials",
            "date_key",
            "production.dim_date",
            "date_key",
        )
    )

    # Print the final quality report.
    print(suite.summary())

    return suite


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_quality_checks()

    if not result.passed:
        print("\n Pipeline should stop - quality checks failed")
        sys.exit(1)

    print("\n All quality checks passed - data is production ready")