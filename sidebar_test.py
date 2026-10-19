
import streamlit as st

st.set_page_config(
    page_title="Sidebar Diagnostic",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RetailPulse Sidebar Diagnostic")
st.write("If you can read this, the test app is running.")

with st.sidebar:
    st.title("Sidebar Test")
    st.write("Can you see this message?")
    st.radio("Choose a page", ["Sales", "Customers", "Forecasting"])
