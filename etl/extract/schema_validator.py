# etl/extract/schema_validator.py
"""
Schema Validator - ETL Extract Layer

This module validates raw DataFrames before they are loaded into PostgreSQL.

This validator acts as a gatekeeper:
- If the file passes validation, it can move to the staging load step.
- If the file fails validation, the pipeline stops before touching the database.

This protects the database from partial loads and bad source data.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Expected schema definitions
# ---------------------------------------------------------------------------

# These dictionaries define the columns we expect in each source file.
#

TRANSACTIONS_SCHEMA = {
    "transaction_id": "string",
    "date": "date",
    "year": "integer",
    "month": "integer",
    "department": "string",
    "region": "string",
    "category": "string",
    "vendor": "string",
    "amount": "float",
    "budget_amount": "float",
    "transaction_type": "string",
    "is_anomaly": "boolean",
}

MONTHLY_SUMMARY_SCHEMA = {
    "year": "integer",
    "month": "integer",
    "month_name": "string",
    "quarter": "string",
    "department": "string",
    "region": "string",
    "total_revenue": "float",
    "total_expense": "float",
    "total_budget": "float",
    "headcount": "integer",
}


# ---------------------------------------------------------------------------
# Allowed values for categorical columns
# ---------------------------------------------------------------------------

# These sets define the only values that are allowed for key business fields.
VALID_DEPARTMENTS = {"IT", "Marketing", "Sales", "HR", "Operations"}
VALID_REGIONS = {"East", "West", "Central"}
VALID_TRANSACTION_TYPES = {"EXPENSE", "REVENUE"}


# ---------------------------------------------------------------------------
# ValidationResult - structured validation output
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Store the result of a validation run.

    Instead of returning only True or False, we return a structured object that
    contains:
    - whether the file is valid
    - all errors found
    - all warnings found
    - row count
    - file type

    This makes the output easier to print, log, and use in the next ETL step.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    file_type: Optional[str] = None

    def add_error(self, message: str) -> None:
        """
        Add a blocking validation error.

        Any error means the file should NOT be loaded into PostgreSQL.
        """
        self.errors.append(f" ERROR: {message}")
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """
        Add a non-blocking warning.

        Warnings are useful for things we want to review, but they do not stop the pipeline by themselves.
        """
        self.warnings.append(f"  WARNING: {message}")

    def summary(self) -> str:
        """
        Return a human-readable validation summary.
        """

        lines = [
            f"Validation {'PASSED ' if self.is_valid else 'FAILED '}",
            f"File type : {self.file_type}",
            f"Row count : {self.row_count:,}",
        ]

        if self.errors:
            lines.append("\nErrors:")
            lines.extend(self.errors)

        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(self.warnings)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for consistent validation.

    Example:
    " Amount " becomes "amount"

    This prevents validation from failing just because the Excel file has  extra spaces or different letter casing in the column names.
    """

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    return df


def _find_missing_columns(df: pd.DataFrame, expected_schema: dict) -> set[str]:
    """
    Compare the DataFrame columns against the expected schema columns.
    """

    expected_columns = set(expected_schema.keys())
    actual_columns = set(df.columns)

    return expected_columns - actual_columns


def _count_non_numeric_values(series: pd.Series) -> int:
    """
    Count values that are present but cannot be converted to numbers.

    This avoids counting original null values as non-numeric.
    Null checks are handled separately.
    """

    converted = pd.to_numeric(series, errors="coerce")

    original_nulls = series.isna().sum()
    converted_nulls = converted.isna().sum()

    return converted_nulls - original_nulls


# ---------------------------------------------------------------------------
# Main validator functions
# ---------------------------------------------------------------------------

def validate_transactions(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the raw transactions DataFrame.

    Checks performed:
    1. Required columns exist.
    2. File is not empty.
    3. Critical columns do not contain nulls.
    4. Department, region, and transaction type values are valid.
    5. Amount and budget_amount can be converted to numbers.
    6. Amount values are not negative.
    7. transaction_id values are not duplicated.
    8. Vendor nulls are reported as warnings.
    """

    # Normalize columns first so checks are not affected by spaces or casing.
    df = _normalize_columns(df)

    result = ValidationResult(
        row_count=len(df),
        file_type="transactions"
    )

    # -----------------------------------------------------------------------
    # Check 1: Required columns
    # -----------------------------------------------------------------------

    missing_cols = _find_missing_columns(df, TRANSACTIONS_SCHEMA)

    if missing_cols:
        result.add_error(f"Missing required columns: {missing_cols}")

        # We stop here because later checks need these columns to exist.
        return result

    # -----------------------------------------------------------------------
    # Check 2: Empty file
    # -----------------------------------------------------------------------

    if len(df) == 0:
        result.add_error("File contains no data rows")
        return result

    # -----------------------------------------------------------------------
    # Check 3: Null checks on critical columns
    # -----------------------------------------------------------------------

    critical_cols = [
        "transaction_id",
        "date",
        "department",
        "region",
        "amount",
        "transaction_type",
    ]

    for col in critical_cols:
        null_count = df[col].isna().sum()

        if null_count > 0:
            result.add_error(f"Column '{col}' has {null_count} null values")

    # -----------------------------------------------------------------------
    # Check 4: Categorical validation
    # -----------------------------------------------------------------------

    # Validate department values.
    invalid_departments = set(df["department"].dropna().unique()) - VALID_DEPARTMENTS

    if invalid_departments:
        result.add_error(
            f"Invalid departments found: {invalid_departments}. "
            f"Expected values: {VALID_DEPARTMENTS}"
        )

    # Validate region values.
    invalid_regions = set(df["region"].dropna().unique()) - VALID_REGIONS

    if invalid_regions:
        result.add_error(
            f"Invalid regions found: {invalid_regions}. "
            f"Expected values: {VALID_REGIONS}"
        )

    # Validate transaction type values.
    invalid_transaction_types = (
        set(df["transaction_type"].dropna().unique()) - VALID_TRANSACTION_TYPES
    )

    if invalid_transaction_types:
        result.add_error(
            f"Invalid transaction types found: {invalid_transaction_types}. "
            f"Expected values: {VALID_TRANSACTION_TYPES}"
        )

    # -----------------------------------------------------------------------
    # Check 5: Numeric validation for amount and budget_amount
    # -----------------------------------------------------------------------

    for numeric_col in ["amount", "budget_amount"]:
        non_numeric_count = _count_non_numeric_values(df[numeric_col])

        if non_numeric_count > 0:
            result.add_error(
                f"Column '{numeric_col}' has {non_numeric_count} non-numeric values"
            )

    # -----------------------------------------------------------------------
    # Check 6: Amount values should not be negative
    # -----------------------------------------------------------------------

    amounts = pd.to_numeric(df["amount"], errors="coerce")
    negative_amount_count = (amounts < 0).sum()

    if negative_amount_count > 0:
        result.add_error(
            f"Column 'amount' has {negative_amount_count} negative values"
        )

    # -----------------------------------------------------------------------
    # Check 7: Duplicate transaction IDs
    # -----------------------------------------------------------------------

    duplicate_transaction_ids = df["transaction_id"].duplicated().sum()

    if duplicate_transaction_ids > 0:
        result.add_error(
            f"{duplicate_transaction_ids} duplicate transaction_id values found"
        )

    # -----------------------------------------------------------------------
    # Warnings: vendor nulls
    # -----------------------------------------------------------------------

    # A missing vendor is not always a blocking issue because revenue rows may
    # not have vendors. For now, we report it as a warning.
    null_vendor_count = df["vendor"].isna().sum()

    if null_vendor_count > 0:
        result.add_warning(
            f"{null_vendor_count} rows have null vendor. "
            f"This may be acceptable for revenue rows."
        )

    return result


def validate_monthly_summary(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the raw monthly summary DataFrame.

    Checks performed:
    1. Required columns exist.
    2. File is not empty.
    3. Critical columns do not contain nulls.
    4. Row count matches the expected business grain.
    5. Month values are between 1 and 12.
    6. Budget values are not negative.
    """

    # Normalize columns first so " Month " and "month" are treated the same.
    df = _normalize_columns(df)

    result = ValidationResult(
        row_count=len(df),
        file_type="monthly_summary"
    )

    # -----------------------------------------------------------------------
    # Check 1: Required columns
    # -----------------------------------------------------------------------

    missing_cols = _find_missing_columns(df, MONTHLY_SUMMARY_SCHEMA)

    if missing_cols:
        result.add_error(f"Missing required columns: {missing_cols}")
        return result

    # -----------------------------------------------------------------------
    # Check 2: Empty file
    # -----------------------------------------------------------------------

    if len(df) == 0:
        result.add_error("File contains no data rows")
        return result

    # -----------------------------------------------------------------------
    # Check 3: Null checks on critical columns
    # -----------------------------------------------------------------------

    critical_cols = [
        "year",
        "month",
        "department",
        "region",
        "total_expense",
        "total_budget",
    ]

    for col in critical_cols:
        null_count = df[col].isna().sum()

        if null_count > 0:
            result.add_error(f"Column '{col}' has {null_count} null values")

    # -----------------------------------------------------------------------
    # Check 4: Expected row count
    # -----------------------------------------------------------------------

    # The monthly summary is expected to have:
    # 5 departments × 3 regions × 36 months = 540 rows.
    #
    # This is a warning instead of an error because missing combinations may be
    # explainable, but it is still important to review.
    expected_rows = 540

    if len(df) != expected_rows:
        result.add_warning(
            f"Expected {expected_rows} rows, found {len(df)}. "
            f"Some department/region/month combinations may be missing."
        )

    # -----------------------------------------------------------------------
    # Check 5: Month range
    # -----------------------------------------------------------------------

    months = pd.to_numeric(df["month"], errors="coerce")
    invalid_month_count = ((months < 1) | (months > 12)).sum()

    if invalid_month_count > 0:
        result.add_error(
            f"{invalid_month_count} rows have month values outside range 1-12"
        )

    # -----------------------------------------------------------------------
    # Check 6: Budget values should not be negative
    # -----------------------------------------------------------------------

    budgets = pd.to_numeric(df["total_budget"], errors="coerce")
    negative_budget_count = (budgets < 0).sum()

    if negative_budget_count > 0:
        result.add_error(
            f"{negative_budget_count} rows have negative budget values"
        )

    return result


def validate_file(df: pd.DataFrame, file_type: str) -> ValidationResult:
    """
    Route the DataFrame to the correct validator based on file type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame returned from source_connector.py.

    file_type : str
        The type of file being validated.
        Expected values:
        - "transactions"
        - "monthly_summary"
    """

    if file_type == "transactions":
        return validate_transactions(df)

    if file_type == "monthly_summary":
        return validate_monthly_summary(df)

    raise ValueError(
        f"Unknown file_type: '{file_type}'. "
        f"Expected 'transactions' or 'monthly_summary'."
    )


# ---------------------------------------------------------------------------
# Script entry point - quick validation test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
   
    1. Reads each raw source file.
    2. Sends it to the correct validator.
    3. Prints a readable validation summary.
    """

    from source_connector import read_source_file

    files_to_validate = [
        ("data/raw/transactions.xlsx", "transactions"),
        ("data/raw/monthly_summary.xlsx", "monthly_summary"),
    ]

    for file_name, file_type in files_to_validate:
        print(f"\n{'─' * 50}")
        print(f"Validating: {file_name}")

        df, _ = read_source_file(file_name)

        validation_result = validate_file(df, file_type)

        print(validation_result.summary())