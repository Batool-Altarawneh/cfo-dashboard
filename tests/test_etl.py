"""
Unit and integration tests for the ETL pipeline.

This file is used by pytest and GitHub Actions.
Every time I push code to GitHub, the CI pipeline will run these tests to make sure the ETL logic still works correctly.

What this file tests:
1. schema_validator.py
   - Checks if raw transaction data has the correct columns and valid values.

2. cleaner.py
   - Checks if messy transaction data is cleaned correctly.
"""

import os
import sys

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestSchemaValidator:
    """
    Tests for etl/extract/schema_validator.py.

    The schema validator is responsible for checking raw data before we continue the ETL pipeline.

    It should catch problems early, such as:
    - Missing columns
    - Negative amounts
    - Invalid department names
    - Invalid transaction structure
    """

    def test_valid_transactions_pass(self):
        """
        Test that a valid transaction DataFrame passes validation.

        This is the happy path test.
        If the data has all required columns and valid values, validate_transactions should return result.is_valid = True.
        """

        from etl.extract.schema_validator import validate_transactions

        # Create a small fake transactions dataset.
        # We do not need the real database or real Excel file for this test.
        # The purpose is to check the validation logic only.
        df = pd.DataFrame({
            "transaction_id": ["TXN-001", "TXN-002"],
            "date": ["2025-01-01", "2025-01-02"],
            "year": [2025, 2025],
            "month": [1, 1],
            "department": ["IT", "HR"],
            "region": ["East", "West"],
            "category": ["Software Licences", "Training"],
            "vendor": ["Microsoft", "Coursera"],
            "amount": [5000.0, 1200.0],
            "budget_amount": [4500.0, 1300.0],
            "transaction_type": ["EXPENSE", "EXPENSE"],
            "is_anomaly": [False, False],
        })

        result = validate_transactions(df)

        # The validator should accept this data.
        # If it fails, show the validation errors to make debugging easier.
        assert result.is_valid, f"Expected valid but got errors: {result.errors}"

    def test_missing_columns_fail(self):
        """
        Test that validation fails when required columns are missing.

        This protects the ETL pipeline from continuing with incomplete data.
        """

        from etl.extract.schema_validator import validate_transactions

        # This DataFrame is intentionally incomplete.
        # It only has transaction_id and amount.
        # Required columns like date, department, region, etc. are missing.
        df = pd.DataFrame({
            "transaction_id": ["TXN-001"],
            "amount": [5000.0],
        })

        result = validate_transactions(df)

        # The validator should reject this data.
        assert not result.is_valid

        # At least one error message should mention missing columns.
        assert any("Missing" in error for error in result.errors)

    def test_negative_amounts_fail(self):
        """
        Test that validation fails when amount is negative.

        In this project, transaction amounts should not be negative.
        A negative amount may indicate bad source data or a loading issue.
        """

        from etl.extract.schema_validator import validate_transactions

        df = pd.DataFrame({
            "transaction_id": ["TXN-001"],
            "date": ["2025-01-01"],
            "year": [2025],
            "month": [1],
            "department": ["IT"],
            "region": ["East"],
            "category": ["Software Licences"],
            "vendor": ["Microsoft"],
            "amount": [-500.0],
            "budget_amount": [1000.0],
            "transaction_type": ["EXPENSE"],
            "is_anomaly": [False],
        })

        result = validate_transactions(df)

        # The validator should reject negative amounts.
        assert not result.is_valid

    def test_invalid_department_fails(self):
        """
        Test that validation fails for an invalid department.

        This makes sure the data only contains expected departments, such as IT, HR, Marketing, Operations, and Sales.
        """

        from etl.extract.schema_validator import validate_transactions

        df = pd.DataFrame({
            "transaction_id": ["TXN-001"],
            "date": ["2025-01-01"],
            "year": [2025],
            "month": [1],
            "department": ["InvalidDept"],
            "region": ["East"],
            "category": ["Software"],
            "vendor": ["Microsoft"],
            "amount": [5000.0],
            "budget_amount": [4500.0],
            "transaction_type": ["EXPENSE"],
            "is_anomaly": [False],
        })

        result = validate_transactions(df)

        # The validator should reject departments that are not part of the allowed business department list.
        assert not result.is_valid


class TestCleaner:
    """
    Tests for etl/transform/cleaner.py.

    The cleaner is responsible for preparing transaction data for analytics.

    It usually handles:
    - Converting columns to correct data types
    - Standardising text values
    - Removing invalid rows
    - Cleaning amount and date fields
    """

    def test_clean_transactions_returns_dataframe(self):
        """
        Test that clean_transactions returns a pandas DataFrame.

        This test uses intentionally messy values:
        - year and month are strings
        - department has extra spaces
        - region is lowercase
        - amount and budget_amount are strings
        - transaction_type is lowercase
        - is_anomaly is a string

        The cleaner should still process the data successfully.
        """

        from etl.transform.cleaner import clean_transactions

        df = pd.DataFrame({
            "transaction_id": ["TXN-001"],
            "date": ["2025-01-01"],
            "year": ["2025"],
            "month": ["1"],
            "department": [" IT "],
            "region": ["east"],
            "category": ["Software Licences"],
            "vendor": ["Microsoft"],
            "amount": ["5000.0"],
            "budget_amount": ["4500.0"],
            "transaction_type": ["expense"],
            "is_anomaly": ["false"],
        })

        result = clean_transactions(df)

        # The output should still be a DataFrame after cleaning.
        assert isinstance(result, pd.DataFrame)

        # The cleaned DataFrame should not be empty.
        assert len(result) > 0

    def test_clean_removes_negative_amounts(self):
        """
        Test that clean_transactions removes rows with negative amounts.

        The input has two rows:
        - TXN-001 has a valid positive amount
        - TXN-002 has an invalid negative amount

        After cleaning, only rows with positive amounts should remain.
        """

        from etl.transform.cleaner import clean_transactions

        df = pd.DataFrame({
            "transaction_id": ["TXN-001", "TXN-002"],
            "date": ["2025-01-01", "2025-01-02"],
            "year": [2025, 2025],
            "month": [1, 1],
            "department": ["IT", "IT"],
            "region": ["East", "East"],
            "category": ["Software Licences", "Hardware"],
            "vendor": ["Microsoft", "Dell"],
            "amount": [5000.0, -100.0],
            "budget_amount": [4500.0, 200.0],
            "transaction_type": ["EXPENSE", "EXPENSE"],
            "is_anomaly": [False, False],
        })

        result = clean_transactions(df)

        # Every remaining amount should be greater than zero.
        assert all(result["amount"] > 0)

    def test_department_standardisation(self):
        """
        Test that department values are standardised.

        The input department is written as ' it ' with lowercase letters and extra spaces.

        The cleaner should convert it to the standard format: 'IT'.
        """

        from etl.transform.cleaner import clean_transactions

        df = pd.DataFrame({
            "transaction_id": ["TXN-001"],
            "date": ["2025-01-01"],
            "year": [2025],
            "month": [1],
            "department": [" it "],
            "region": ["East"],
            "category": ["Software Licences"],
            "vendor": ["Microsoft"],
            "amount": [5000.0],
            "budget_amount": [4500.0],
            "transaction_type": ["EXPENSE"],
            "is_anomaly": [False],
        })

        result = clean_transactions(df)

        # After cleaning, the department name should match the official format.
        assert result.iloc[0]["department"] == "IT"