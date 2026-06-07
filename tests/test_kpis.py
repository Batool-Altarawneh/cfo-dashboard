"""
test_kpis.py
------------
Tests for KPI calculation and formatting functions.

This file is used by pytest and GitHub Actions CI.
The goal is to make sure the financial KPI logic still works correctly
after any code change.

What this file tests:
1. etl/transform/kpi_builder.py
   - Checks that KPI columns are created correctly.
   - Checks budget variance and YTD calculations.

2. streamlit/utils/formatters.py
   - Checks that numbers are displayed correctly on the dashboard.
   - Checks that variance colors match the expected business thresholds.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestKPIBuilder:
    """
    Tests for etl/transform/kpi_builder.py.

    The KPI builder is responsible for calculating financial metrics from monthly summary data.

    """

    def get_sample_monthly_df(self):
        """
        Create a small fake monthly summary DataFrame for testing.

        I am creating test data manually here instead of using the real database.
        This makes the tests faster, simpler, and more reliable.

        The sample includes:
        - 2 years: 2024 and 2025
        - 12 months per year
        - 2 departments: IT and HR
        - 1 region: East

        IT is intentionally over budget:
        - total_expense = 180,000
        - total_budget  = 150,000

        HR is intentionally under budget:
        - total_expense = 60,000
        - total_budget  = 65,000
        """

        data = []

        for year in [2024, 2025]:
            for month in range(1, 13):
                for dept in ["IT", "HR"]:
                    data.append({
                        "department": dept,
                        "region": "East",
                        "year": year,
                        "month": month,
                        "quarter": (month - 1) // 3 + 1,

                        # Revenue is higher in Q4 to simulate seasonal revenue growth.
                        "total_revenue": 500000.0 if month in [10, 11, 12] else 200000.0,

                        # IT spends more than HR in this sample.
                        "total_expense": 180000.0 if dept == "IT" else 60000.0,

                        # IT is over budget, HR is slightly under budget.
                        "total_budget": 150000.0 if dept == "IT" else 65000.0,
                    })

        return pd.DataFrame(data)

    def test_calculate_kpis_returns_dataframe(self):
        """
        Test that calculate_kpis returns a pandas DataFrame.

        This is a basic safety test.
        The function should return a DataFrame and keep the same number of rows.
        """

        from etl.transform.kpi_builder import calculate_kpis

        df = self.get_sample_monthly_df()
        result = calculate_kpis(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_spending_velocity_calculated(self):
        """
        Test that spending_velocity is created and doesnt contain null values.

        Spending velocity is important because it shows how fast a department is using its budget compared with the month or quarter timeline.
        """

        from etl.transform.kpi_builder import calculate_kpis

        df = self.get_sample_monthly_df()
        result = calculate_kpis(df)

        assert "spending_velocity" in result.columns
        assert result["spending_velocity"].notna().all()

    def test_budget_variance_correct(self):
        """
        Test that budget_variance_amt is calculated correctly.

        In the sample data, IT is always over budget:
        - total_expense = 180,000
        - total_budget  = 150,000

        So the budget variance amount should be positive for all IT rows.
        """

        from etl.transform.kpi_builder import calculate_kpis

        df = self.get_sample_monthly_df()
        result = calculate_kpis(df)

        assert "budget_variance_amt" in result.columns

        it_rows = result[result["department"] == "IT"]

        # Since IT expense is greater than budget, variance should be positive.
        assert (it_rows["budget_variance_amt"] > 0).all()

    def test_ytd_cumulative(self):
        """
        Test that expense_ytd increases over the year.

        """

        from etl.transform.kpi_builder import calculate_kpis

        df = self.get_sample_monthly_df()
        result = calculate_kpis(df)

        assert "expense_ytd" in result.columns

        it_2025 = result[
            (result["department"] == "IT") &
            (result["year"] == 2025)
        ].sort_values("month")

        assert it_2025["expense_ytd"].iloc[-1] > it_2025["expense_ytd"].iloc[0]


class TestFormatters:
    """
    Tests for streamlit/utils/formatters.py.

    These tests check dashboard display formatting.

    Formatting is important because the dashboard should show values in a clean business-friendly way, such as:
    - 21,800,000 -> $21.8M
    - 58,200     -> $58.2K
    - 8.16       -> 8.2%
    """

    def test_format_currency_millions(self):
        """
        Test that large currency values are formatted in millions.
        """

        from streamlit.utils.formatters import format_currency

        assert format_currency(21800000) == "$21.8M"

    def test_format_currency_thousands(self):
        """
        Test that medium currency values are formatted in thousands.
        """

        from streamlit.utils.formatters import format_currency

        assert format_currency(58200) == "$58.2K"

    def test_format_currency_small(self):
        """
        Test that small currency values are displayed without K or M.
        """

        from streamlit.utils.formatters import format_currency

        assert format_currency(500) == "$500"

    def test_format_percentage(self):
        """
        Test that percentage values are rounded to one decimal place.

        Example:
        8.16 -> 8.2%
        """

        from streamlit.utils.formatters import format_percentage

        assert format_percentage(8.16) == "8.2%"

    def test_variance_colour_red(self):
        """
        Test that high budget variance returns the red color.

        - Red means high risk or high overspend.
        """

        from streamlit.utils.formatters import get_variance_colour

        assert get_variance_colour(0.15) == "#C0392B"

    def test_variance_colour_amber(self):
        """
        Test that medium budget variance returns the amber color.

        - Amber means warning or moderate overspend.
        """

        from streamlit.utils.formatters import get_variance_colour

        assert get_variance_colour(0.07) == "#E67E22"

    def test_variance_colour_green(self):
        """
        Test that low budget variance returns the green color.

        - Green means acceptable or low risk.
        """

        from streamlit.utils.formatters import get_variance_colour

        assert get_variance_colour(0.03) == "#1E8449"