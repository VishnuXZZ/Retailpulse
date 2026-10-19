"""RetailPulse — multi-page Streamlit entry."""
import os, sys
from pathlib import Path
import streamlit as st

# when launched via `streamlit run app.py`, ensure project root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="RetailPulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Sales Dashboard": "pages/1_Sales_Dashboard.py",
    "Customer Analytics": "pages/2_Customer_Analytics.py",
    "Demand Forecasting": "pages/3_Demand_Forecasting.py",
    "Churn Risk": "pages/4_Churn_Analysis.py",
    "Inventory Optimization": "pages/5_Inventory_Optimization.py",
    "Model Monitoring & MLOps": "pages/6_Model_Monitoring.py",
}

st.markdown(
    """
    <div style="padding:1.25rem 1.5rem; background:linear-gradient(135deg,#1E3A8A 0%,#1D4ED8 50%,#06B6D4 100%); border-radius:14px; color:#fff; margin-bottom:1.25rem;">
        <div style="font-size:1.6rem; font-weight:700;">🛒 RetailPulse</div>
        <div style="opacity:0.9;">AI-Powered Customer Analytics &amp; Demand Forecasting Platform</div>
        <div style="font-size:0.8rem; opacity:0.7; margin-top:0.25rem;">UCI Online Retail II &nbsp;·&nbsp; Predictive Demand, RFM, Churn, Inventory, MLOps</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Navigation")
selection = st.radio("Choose a dashboard", list(PAGES.keys()), label_visibility="collapsed")

st.markdown("---")
st.markdown("**Source dataset:** UCI Online Retail II (`online_retail_II.xlsx`, 1.07 M transactions, 2009-12 – 2011-12).")
st.markdown("**Pipeline artefacts:** run `python -m src.build_processed` to regenerate every model, segment and recommendation from raw.")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart--v1.png", width=64)
    st.markdown("## 🛒 RetailPulse")
    st.caption("Zidio Development · Data Science & Analytics · March 2026")
    st.markdown("---")
    sel_side = st.radio("Pages", list(PAGES.keys()))
    if sel_side and sel_side != selection:
        st.rerun()

# Run the chosen page by file execution (acts like a multi-page Streamlit app)
sel = selection or sel_side
target = ROOT / PAGES[sel]
# exec the page in __main__ namespace
if target.exists():
    exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), globals())
else:
    st.error(f"Page not found: {target}")
