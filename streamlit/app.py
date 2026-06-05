
import streamlit as st
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Page configuration — must be first Streamlit call
st.set_page_config(
    page_title  = "CFO Financial Dashboard",
    page_icon   = "💰",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Landing page content
st.markdown(
    '<div class="page-header">💰 NorthStar Analytics - CFO Financial Dashboard</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Descriptive Analytics
    - Executive Summary
    - Revenue & Expenses
    - Budget vs Actual
    - Transaction Detail
    """)

with col2:
    st.markdown("""
    ### 🤖 Predictive Analytics
    - Revenue Forecast (Prophet)
    - Budget Overrun Risk (XGBoost)
    - Anomaly Detection (Isolation Forest)
    """)

with col3:
    st.markdown("""
    ### 🏗️ Data Pipeline
    - PostgreSQL Star Schema
    - Airflow ETL Orchestration
    - 19/19 Quality Checks Passing
    - 3 Years · 5 Departments · 3 Regions
    """)

st.info("👈 Use the sidebar to navigate between pages")