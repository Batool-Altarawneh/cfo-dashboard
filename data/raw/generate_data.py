# Purpose of this file:
# Generate synthetic financial data for the CFO Dashboard project.
#
# This script creates two Excel files:
# 1. data/raw/transactions.xlsx
#    - detailed transaction-level financial data
#
# 2. data/raw/monthly_summary.xlsx
#    - monthly aggregated summary by department and region
#
# I am using synthetic data because this is a portfolio project,
# and I want the data to look realistic without using private company data.

import pandas as pd
import numpy as np
import random
import calendar #to know last day for each month
import os
from datetime import date #to create transaction date 

#! ------------------------------------------------------------
#! 1. Reproducibility
#! ------------------------------------------------------------
np.random.seed(7)
random.seed(7)

#! ------------------------------------------------------------
#! 2. Company Configuration 
#! ------------------------------------------------------------
# I am creating a fictional company with 5 departments and 3 regions.
# These departments and regions will be used across the whole project:
# ETL, PostgreSQL, Power BI, Streamlit, and ML models.

DEPARTMENTS = ["IT", "Marketing", "Sales", "HR", "Operations"]
REGIONS = ["East", "West", "Central"]

BASE_BUDGETS = {
    "IT": 180_000,
    "Marketing": 95_000,
    "Sales": 75_000,
    "HR": 60_000,
    "Operations": 85_000,
}

# Region weights control how much business activity each region gets.
# East has the largest share, then West, then Central.
REGION_WEIGHTS = {
    "East": 0.50,
    "West": 0.30,
    "Central": 0.20,
}

#! ------------------------------------------------------------
#! 3. Department categories and vendors
#! ------------------------------------------------------------
# Each department has its own realistic expense categories and vendors.
# This makes the dataset feel closer to real business data instead of random numbers only.
DEPT_CATEGORIES = {
 # Each department has a list of categories, and each category has a list of possible vendors.
# I used a list of vendors because one category can have multiple realistic suppliers.
# For example, (Software Licences) can come from Microsoft, AWS, GitHub, etc...
# This makes the synthetic data more realistic than assigning only one fixed vendor per category.
    "IT": [
        ("Software Licences", ["Microsoft", "AWS", "Atlassian", "GitHub", "Datadog"]),
        ("Cloud Infrastructure", ["AWS", "Azure", "GCP"]),
        ("Hardware", ["Dell", "Lenovo", "Apple", "Cisco"]),
        ("IT Support", ["CDW Canada", "Bell Canada", "Rogers"]),
    ],

    "Marketing": [
        ("Digital Advertising", ["Google Ads", "Meta Ads", "LinkedIn Ads"]),
        ("Events", ["Eventbrite", "Metro Toronto CC", "Fairmont Hotels"]),
        ("Agency Fees", ["Ogilvy Canada", "DDB Canada", "Zulu Alpha Kilo"]),
        ("Content Tools", ["HubSpot", "Hootsuite", "Canva", "Adobe"]),
    ],

    "Sales": [
        ("Travel", ["Air Canada", "WestJet", "Marriott", "Hertz"]),
        ("Client Entertainment", ["Canoe Restaurant", "Hy's Steakhouse"]),
        ("Sales Tools", ["Salesforce", "ZoomInfo", "Outreach"]),
        ("Commissions", ["Internal Payroll"]),
    ],

    "HR": [
        ("Recruiting", ["LinkedIn Talent", "Indeed Canada", "Hays Recruiting"]),
        ("Training", ["Udemy Business", "Coursera", "Dale Carnegie"]),
        ("Benefits Admin", ["Manulife", "Sun Life", "Green Shield Canada"]),
        ("HR Software", ["Workday", "BambooHR", "ADP Canada"]),
    ],

    "Operations": [
        ("Facilities", ["WeWork", "CBRE Canada", "Brookfield Properties"]),
        ("Logistics", ["Purolator", "FedEx Canada", "Canada Post"]),
        ("Office Supplies", ["Staples Canada", "Bureau en Gros"]),
        ("Utilities", ["Toronto Hydro", "Enbridge Gas", "Rogers Business"]),
    ],
}


# Month names will be used later in the monthly summary table.
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

#! ------------------------------------------------------------
#! 4. Create the date range
#! ------------------------------------------------------------
# The project needs 3 years of data: 2023, 2024, and 2025.
# This gives enough history for trends, YoY growth, and forecasting.

months = [(year, month) for year in [2023, 2024, 2025] for month in range(1, 13)]


def random_date_in_month(year, month):
    """
    Return a random date inside a specific year and month.

    I need this because each transaction should have a realistic date, not just the first day of the month.
    """

    # Get the last valid day of the month.
    # Example: February has 28 or 29 days, April has 30, January has 31.
    last_day = calendar.monthrange(year, month)[1]

    # Pick one random day between 1 and the last day of that month.
    random_day = random.randint(1, last_day)

    return date(year, month, random_day)

#! ------------------------------------------------------------
#! 5. Generate expense transactions
#! ------------------------------------------------------------
# Each row in this list will become one transaction in the final DataFrame.
transactions = []

# numeric counter to create unique transaction IDs.
txn_id = 1


# Loop through every month, department, and region.
# This creates a complete dataset across 3 years.
for year, month in months:
    for dept in DEPARTMENTS:
        for region in REGIONS:

            # Generate between 12 and 18 expense transactions for each
            # department-region-month combination.
            num_transactions = random.randint(12, 18)

            for _ in range(num_transactions):

                # Pick a realistic category and vendor for this department.
                category_name, vendors = random.choice(DEPT_CATEGORIES[dept])
                vendor = random.choice(vendors)

                # Estimate an average transaction amount based on the department budget.
                # Then use a normal distribution to make the amounts vary naturally.
                avg_amount = BASE_BUDGETS[dept] / num_transactions
                amount = abs(np.random.normal(avg_amount, avg_amount * 0.25))

                # ------------------------------------------------------------
                # Business Pattern 1:
                # IT consistently goes over budget by around 15% to 22%.
                
                # ------------------------------------------------------------
                if dept == "IT":
                    amount *= np.random.uniform(1.15, 1.22)

                # ------------------------------------------------------------
                # Business Pattern 2:
                # HR usually spends less than expected.
                # ------------------------------------------------------------
                if dept == "HR":
                    amount *= np.random.uniform(0.88, 0.92)

                # Create a budget amount.
                # For IT, I intentionally make budget lower than actual spending
                # so that the over-budget pattern appears clearly.
                if dept == "IT":
                    budget_amount = amount / np.random.uniform(1.15, 1.22)
                else:
                    budget_amount = amount * np.random.uniform(0.90, 1.10)

                # Add the transaction to the list as a dictionary.
                transactions.append({
                    "transaction_id": f"TXN-{txn_id:05d}",
                    "date": random_date_in_month(year, month),
                    "year": year,
                    "month": month,
                    "department": dept,
                    "region": region,
                    "category": category_name,
                    "vendor": vendor,
                    "amount": round(amount, 2),
                    "budget_amount": round(budget_amount, 2),
                    "transaction_type": "EXPENSE",
                    "is_anomaly": False,
                })

                txn_id += 1

#! ------------------------------------------------------------
#! 6. Generate revenue transactions
#! ------------------------------------------------------------
# For this project, revenue is assigned to Sales.
# This makes the CFO dashboard easier to understand:
# Sales brings revenue, while all departments have expenses.

for year, month in months:
    for region in REGIONS:

        # Base revenue depends on the region.
        # East has the largest share because its weight is 50%.
        base_revenue = 500_000 * REGION_WEIGHTS[region]

        # ------------------------------------------------------------
        # Business Pattern 3:
        # Sales revenue increases in Q4 because of the holiday season.
        # January and February are slightly lower.
        # ------------------------------------------------------------
        if month in [10, 11, 12]:
            seasonal_factor = np.random.uniform(1.40, 1.60)
        elif month in [1, 2]:
            seasonal_factor = np.random.uniform(0.88, 0.95)
        else:
            seasonal_factor = np.random.uniform(0.97, 1.08)

        # Add 8% year-over-year growth.
        # 2023 = baseline
        # 2024 = +8%
        # 2025 = +16%
        yoy_growth = 1 + 0.08 * (year - 2023)

        # Add a little random noise so revenue does not look too perfect.
        revenue = abs(
            base_revenue * seasonal_factor * yoy_growth
            + np.random.normal(0, base_revenue * 0.03)
        )

        transactions.append({
            "transaction_id": f"TXN-{txn_id:05d}",
            "date": random_date_in_month(year, month),
            "year": year,
            "month": month,
            "department": "Sales",
            "region": region,
            "category": "Revenue",
            "vendor": "Client Payments",
            "amount": round(revenue, 2),
            "budget_amount": round(revenue * 0.95, 2),
            "transaction_type": "REVENUE",
            "is_anomaly": False,
        })

        txn_id += 1

#! ------------------------------------------------------------
#! 7. Convert list of transactions to a DataFrame
#! ------------------------------------------------------------
df = pd.DataFrame(transactions)


#! ------------------------------------------------------------
#! 8. Inject Marketing anomalies
#! ------------------------------------------------------------
# I am manually creating known anomalies in Marketing expenses.
# This is useful later for the Isolation Forest anomaly detection model,because I will already know which rows are suspicious.

marketing_expense_mask = (
    (df["department"] == "Marketing")
    & (df["transaction_type"] == "EXPENSE")
)

suspicious_vendors = [
    "Apex Solutions Inc",
    "Global Trade Partners",
    "NovaTech Services",
]


# For every year and quarter, select 3 Marketing expense transactions
# and make them unusually large.
for year in [2023, 2024, 2025]:
    for quarter in [1, 2, 3, 4]:

        # Convert quarter number into its 3 months.
        # Q1 = 1, 2, 3
        # Q2 = 4, 5, 6
        # Q3 = 7, 8, 9
        # Q4 = 10, 11, 12
        quarter_months = [(quarter - 1) * 3 + 1 + i for i in range(3)]

        # Filter Marketing expense transactions for the current year and quarter.
        mask = (
            marketing_expense_mask
            & (df["year"] == year)
            & (df["month"].isin(quarter_months))
        )

        candidate_indexes = df[mask].index.tolist()

        # Only inject anomalies if there are enough candidate rows.
        if len(candidate_indexes) >= 3:

            # Pick 3 random transactions and turn them into anomalies.
            for idx in random.sample(candidate_indexes, 3):

                # Make the amount 3x to 5x larger than normal.
                df.at[idx, "amount"] = round(
                    df.at[idx, "amount"] * np.random.uniform(3.0, 5.0),
                    2
                )

                # Replace the vendor with a suspicious-looking vendor.
                df.at[idx, "vendor"] = random.choice(suspicious_vendors)

                # Mark the row as an anomaly.
                df.at[idx, "is_anomaly"] = True

#! ------------------------------------------------------------
#! 9. Export transaction-level data
#! ------------------------------------------------------------
# Sort the dataset so it is easier to read in Excel. (from oldest to recent, then by dep)
df = df.sort_values(["date", "department"]).reset_index(drop=True)

# Create the data/raw folder if it does not already exist.
os.makedirs("data/raw", exist_ok=True)

# Save the detailed transaction-level dataset.
df.to_excel("data/raw/transactions.xlsx", index=False)

print(
    f"transactions.xlsx  →  {len(df):,} rows  |  "
    f"{df['is_anomaly'].sum()} anomalies injected"
)

#! ------------------------------------------------------------
#! 10. Build monthly summary table
#! ------------------------------------------------------------
# The transaction table is detailed.
# Now I also want a monthly summary table because it is useful for:
# - Power BI KPI cards
# - monthly trend charts
# - forecasting models
# - budget vs actual analysis

expense_df = df[df["transaction_type"] == "EXPENSE"]
revenue_df = df[df["transaction_type"] == "REVENUE"]


# Base headcount by department.
# This will be useful later for the budget overrun classifier.
BASE_HEADCOUNT = {
    "IT": 45,
    "Marketing": 22,
    "Sales": 38,
    "HR": 15,
    "Operations": 30,
}

summary_rows = []


for year, month in months:
    for dept in DEPARTMENTS:
        for region in REGIONS:

            # Filter expense transactions for this exact month, department, and region.
            expense_mask = (
                (expense_df["year"] == year)
                & (expense_df["month"] == month)
                & (expense_df["department"] == dept)
                & (expense_df["region"] == region)
            )

            # Filter revenue transactions for this exact month, department, and region.
            # Since revenue is assigned to Sales, most non-Sales departments will have 0 revenue.
            revenue_mask = (
                (revenue_df["year"] == year)
                & (revenue_df["month"] == month)
                & (revenue_df["department"] == dept)
                & (revenue_df["region"] == region)
            )

            # Headcount grows slightly every year and is distributed by region.
            headcount = int(
                BASE_HEADCOUNT[dept]
                * (1 + 0.03 * (year - 2023))
                * REGION_WEIGHTS[region]
            )

            summary_rows.append({
                "year": year,
                "month": month,
                "month_name": MONTH_NAMES[month],
                "quarter": f"Q{(month - 1) // 3 + 1}",
                "department": dept,
                "region": region,
                "total_revenue": round(revenue_df[revenue_mask]["amount"].sum(), 2),
                "total_expense": round(expense_df[expense_mask]["amount"].sum(), 2),
                "total_budget": round(expense_df[expense_mask]["budget_amount"].sum(), 2),

                # max(1, ...) prevents headcount from becoming zero in smaller regions.
                "headcount": max(1, headcount),
            })


# Convert summary rows to a DataFrame.
summary_df = pd.DataFrame(summary_rows)

# Save monthly summary to Excel.
summary_df.to_excel("data/raw/monthly_summary.xlsx", index=False)

print(f"monthly_summary.xlsx  →  {len(summary_df):,} rows")
print("\n Done  check data/raw/")
