"""

This file contains reusable formatting helper functions for the Streamlit dashboard.


Instead of formatting numbers manually inside every Streamlit page, we keep all formatting logic here in one place.

This makes the dashboard:
1. Cleaner
2. More consistent
3. Easier to update
4. Easier to reuse across multiple pages
"""


# ---------------------------------------------------------------------------
# Function 1: Format currency values
# ---------------------------------------------------------------------------
# This function converts large numbers into readable currency format.
#
# Examples:
# 1,250,000 becomes $1.3M
# 45,000 becomes $45.0K
# 950 becomes $950
#
# This is useful for KPI cards and charts because large numbers are easier to read when they are shortened with K or M.
# ---------------------------------------------------------------------------

def format_currency(value: float, decimals: int = 1) -> str:
    """
    Format a numeric value as currency using K and M suffixes.

    """

    # If the absolute value is 1 million or more, divide by 1,000,000 and add the M suffix.
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.{decimals}f}M"

    # If the absolute value is 1 thousand or more, divide by 1,000 and add the K suffix.
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.{decimals}f}K"

    # If the value is less than 1 thousand,show it as a normal currency number with commas and no decimals.
    else:
        return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# Function 2: Format percentage values
# ---------------------------------------------------------------------------
# This function formats a number as a percentage string.
#
#
# Example:
# 12.345 becomes 12.3%
#
# It does not multiply by 100.
# ---------------------------------------------------------------------------

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a numeric value as a percentage.

    """

    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Function 3: Format budget variance values
# ---------------------------------------------------------------------------
# This function formats variance amounts and adds a plus sign for positive values.
#
#
# Positive variance means actual expense is higher than budget.
# Negative variance means actual expense is lower than budget.
#
# Example:
# 25000 becomes +$25.0K
# -12000 becomes -$12.0K
# ---------------------------------------------------------------------------

def format_variance(value: float) -> str:
    """
    Format a budget variance amount with a plus sign when positive.

    """

    # Add plus sign only when the value is positive.
    # Negative values already include the minus sign automatically.
    prefix = "+" if value > 0 else ""

    return f"{prefix}{format_currency(value)}"


# ---------------------------------------------------------------------------
# Function 4: Get variance color
# ---------------------------------------------------------------------------
# This function returns a color based on the budget variance percentage.
#
#
# Logic:
# - More than 10% over budget: red
# - More than 5% over budget: amber
# - 5% or less: green
#
# ---------------------------------------------------------------------------

def get_variance_colour(value: float) -> str:
    """
    Return a color based on budget variance percentage.

    """

    # Red means high budget overrun.
    if value > 0.10:
        return "#C0392B"

    # Amber/orange means medium warning.
    elif value > 0.05:
        return "#E67E22"

    # Green means acceptable or low variance.
    else:
        return "#1E8449"