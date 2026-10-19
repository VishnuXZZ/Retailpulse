
"""RetailPulse — Sales Dashboard."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SALES_PARQUET = Path("data/processed/sales_clean.parquet")

st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14243b, #101827);
    border: 1px solid #263b57;
    padding: 18px 18px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #a8bad0;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.15rem, 1.7vw, 1.8rem);
    color: #f2f7ff;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 650;
    margin: 0.6rem 0 0.2rem;
}
.section-subtitle {
    color: #93a7c1;
    font-size: 0.85rem;
    margin-bottom: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading cleaned sales...")
def load_sales():
    return pd.read_parquet(SALES_PARQUET)

if not SALES_PARQUET.exists():
    st.error(f"Sales dataset not found: {SALES_PARQUET}")
    st.stop()

sales = load_sales()

st.title("🛍️ Sales Intelligence")
st.caption(
    "Explore revenue performance, product demand, and customer activity "
    "using cleaned retail transactions."
)

# Filters
with st.expander("⚙️ Filters", expanded=True):
    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])

    min_d = sales["InvoiceDate"].min().date()
    max_d = sales["InvoiceDate"].max().date()

    with f1:
        date_range = st.date_input(
            "Invoice date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

    countries = sorted(sales["Country"].dropna().unique().tolist())
    default_country = ["United Kingdom"] if "United Kingdom" in countries else countries[:1]

    with f2:
        country_sel = st.multiselect(
            "Countries",
            countries,
            default=default_country,
        )

    with f3:
        max_revenue = float(sales["RevenueGross"].max())
        rev_cap = st.slider(
            "Maximum transaction-line revenue (£)",
            min_value=0.0,
            max_value=max_revenue,
            value=min(500.0, max_revenue),
            step=10.0,
        )

if len(date_range) != 2:
    st.info("Choose both a start and end date.")
    st.stop()

if date_range[0] > date_range[1]:
    st.warning("The start date must be on or before the end date.")
    st.stop()

filt = sales[
    (sales["InvoiceDate"].dt.date >= date_range[0])
    & (sales["InvoiceDate"].dt.date <= date_range[1])
    & (sales["Country"].isin(country_sel))
    & (sales["RevenueGross"] <= rev_cap)
].copy()

if filt.empty:
    st.warning("No transactions match these filters. Try widening your selection.")
    st.stop()

# KPIs
revenue = filt["RevenueGross"].sum()
invoices = filt["Invoice"].nunique()
units = int(filt["Quantity"].sum())
customers = int(filt["CustomerID"].nunique())
aov = filt.groupby("Invoice")["RevenueGross"].sum().mean()

st.markdown("### Performance overview")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue", f"£{revenue:,.0f}")
k2.metric("Invoices", f"{invoices:,}")
k3.metric("Units sold", f"{units:,}")
k4.metric("Customers", f"{customers:,}")
k5.metric("Average order value", f"£{aov:,.2f}")

# Monthly trend
st.markdown("### Sales trends")
trend = (
    filt.groupby(filt["InvoiceDate"].dt.to_period("M").astype(str))
    .agg(
        Revenue=("RevenueGross", "sum"),
        Invoices=("Invoice", "nunique"),
    )
    .reset_index()
)
trend.columns = ["YearMonth", "Revenue", "Invoices"]
trend["YearMonth"] = pd.to_datetime(trend["YearMonth"])

left, right = st.columns([1.55, 1])

with left:
    fig = px.area(
        trend,
        x="YearMonth",
        y="Revenue",
        markers=True,
        title="Monthly revenue",
        color_discrete_sequence=["#38a5ff"],
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title="Revenue (£)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    dow_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    dow = (
        filt.groupby("DayOfWeek")["RevenueGross"]
        .sum()
        .reindex(dow_order)
        .fillna(0)
        .reset_index()
    )
    dow.columns = ["Day", "Revenue"]

    fig_dow = px.bar(
        dow,
        x="Day",
        y="Revenue",
        title="Revenue by weekday",
        color="Revenue",
        color_continuous_scale=["#1d4ed8", "#38bdf8"],
    )
    fig_dow.update_layout(
        template="plotly_dark",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title="Revenue (£)",
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# Product and country analysis
st.markdown("### Product & market performance")
left, right = st.columns(2)

top_products = (
    filt.groupby(["StockCode", "Description"], dropna=False)
    .agg(
        Revenue=("RevenueGross", "sum"),
        Units=("Quantity", "sum"),
    )
    .sort_values("Revenue", ascending=False)
    .head(15)
    .reset_index()
)
top_products["Product"] = (
    top_products["Description"].fillna("Unknown").astype(str).str.slice(0, 45)
)

with left:
    fig_products = px.bar(
        top_products.sort_values("Revenue"),
        x="Revenue",
        y="Product",
        orientation="h",
        title="Top 15 products by revenue",
        hover_data=["StockCode", "Units"],
        color="Revenue",
        color_continuous_scale=["#1d4ed8", "#38bdf8"],
    )
    fig_products.update_layout(
        template="plotly_dark",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Revenue (£)",
        yaxis_title=None,
    )
    st.plotly_chart(fig_products, use_container_width=True)

country_rev = (
    filt.groupby("Country")["RevenueGross"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .sort_values()
    .reset_index()
)

with right:
    fig_country = px.bar(
        country_rev,
        x="RevenueGross",
        y="Country",
        orientation="h",
        title="Top 15 countries by revenue",
        color="RevenueGross",
        color_continuous_scale=["#6d28d9", "#c084fc"],
    )
    fig_country.update_layout(
        template="plotly_dark",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Revenue (£)",
        yaxis_title=None,
    )
    st.plotly_chart(fig_country, use_container_width=True)

# Transactions
st.markdown("### Transaction explorer")
st.caption(f"{len(filt):,} transaction lines match the current filters.")

search = st.text_input("Search product description or stock code")
table = filt.copy()

if search.strip():
    mask = (
        table["Description"].astype(str).str.contains(search, case=False, na=False)
        | table["StockCode"].astype(str).str.contains(search, case=False, na=False)
    )
    table = table[mask]

st.dataframe(
    table.head(500),
    use_container_width=True,
    height=350,
    hide_index=True,
)

st.download_button(
    "⬇️ Download filtered transactions (CSV)",
    data=filt.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_filtered_sales.csv",
    mime="text/csv",
)

st.caption(
    "Note: the maximum line-revenue filter affects the displayed KPIs and charts. "
    "The CSV download contains all rows matching the date and country filters."
)
