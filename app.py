
"""RetailPulse — native Streamlit multipage entry."""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="RetailPulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="padding:1.25rem 1.5rem;
    background:linear-gradient(135deg,#1E3A8A,#1D4ED8,#06B6D4);
    border-radius:14px;color:white;margin-bottom:1.25rem;">
        <div style="font-size:1.6rem;font-weight:700;">
            🛒 RetailPulse
        </div>
        <div>AI-Powered Customer Analytics &amp; Demand Forecasting Platform</div>
        <div style="font-size:0.8rem;opacity:0.8;margin-top:0.25rem;">
            UCI Online Retail II · Sales, RFM, Churn, Forecasting &amp; MLOps
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Choose a section from the navigation menu in the sidebar.")

pages = {
    "Analytics": [
        st.Page(
            str(ROOT / "pages" / "1_Sales_Dashboard.py"),
            title="Sales Dashboard",
            default=True,
        ),
        st.Page(
            str(ROOT / "pages" / "2_Customer_Analytics.py"),
            title="Customer Analytics",
        ),
        st.Page(
            str(ROOT / "pages" / "3_Demand_Forecasting.py"),
            title="Demand Forecasting",
        ),
        st.Page(
            str(ROOT / "pages" / "4_Churn_Analysis.py"),
            title="Churn Risk",
        ),
        st.Page(
            str(ROOT / "pages" / "5_Inventory_Optimization.py"),
            title="Inventory Optimization",
        ),
        st.Page(
            str(ROOT / "pages" / "6_Model_Monitoring.py"),
            title="Model Monitoring & MLOps",
        ),
    ]
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
