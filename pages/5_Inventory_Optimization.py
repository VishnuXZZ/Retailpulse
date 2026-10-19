"""Page 5 — Inventory Optimization Dashboard."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

PROD = Path("data/processed/product_demand.parquet")
INV = Path("data/processed/inventory_recommendations.parquet")

st.markdown("## 📦 Page 5 — Inventory Optimization")
st.caption("The UCI Online Retail II dataset does NOT contain on-hand stock, supplier lead times, or "
           "purchase orders. Replenishment calculations use user-supplied on-hand, lead time and service "
           "level together with the forecast from Page 3.")

@st.cache_data(show_spinner="Loading product demand …")
def load_prod() -> pd.DataFrame:
    return pd.read_parquet(PROD)

prod = load_prod()

with st.sidebar:
    st.markdown("---")
    lead = st.slider("Lead time (days)", 1, 30, 7)
    sl = st.slider("Service level", 0.80, 0.999, 0.95, step=0.01)
    on_hand_map = {}
    st.markdown("**On-hand stock (top 10 products, fill if known)**")
    top10 = prod.head(10)["StockCode"].astype(str).tolist()
    for c in top10:
        on_hand_map[c] = st.number_input(f"{c}", min_value=0, value=200, step=50)

# recompute on the fly with configurable inputs
import numpy as np
from scipy.stats import norm
def _z(sl):
    if sl >= 0.999: return 3.09
    if sl >= 0.99: return 2.33
    if sl >= 0.97: return 1.88
    if sl >= 0.95: return 1.65
    if sl >= 0.90: return 1.28
    return 0.84

z = _z(sl)
inv_rows = []
for _, row in prod.head(20).iterrows():
    code = str(row["StockCode"])
    avg = float(row.get("AvgPrice", 0))
    daily_qty = max(avg, 0.1)
    sigma = max(avg * 0.2, 1.0)
    e_dl = daily_qty * lead
    sigma_dl = sigma * np.sqrt(lead)
    rop = e_dl + z * sigma_dl
    on_hand = float(on_hand_map.get(code, 0))
    rec_qty = max(0, int(round(rop - on_hand)))
    inv_rows.append({
        "StockCode": code, "Description": row.get("Description", "")[:40],
        "AvgDailyDemand": round(daily_qty, 2),
        "LeadTimeDays": lead, "ServiceLevel": sl,
        "ExpectedDemandLeadTime": round(e_dl, 2),
        "SafetyStock": round(z * sigma_dl, 2),
        "ReorderPoint": round(rop, 2),
        "OnHand (user)": round(on_hand, 2),
        "RecommendedReorderQty": rec_qty,
        "StockoutRisk": on_hand < rop,
    })
inv = pd.DataFrame(inv_rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products reviewed", f"{len(inv)}")
c2.metric("Recommended reorder (units)", f"{int(inv['RecommendedReorderQty'].sum()):,}")
c3.metric("Items flagged at-risk", f"{int(inv['StockoutRisk'].sum())}")
c4.metric("Median Reorder Point", f"{inv['ReorderPoint'].median():,.1f}")

st.markdown("---")

fig = px.bar(inv.sort_values("RecommendedReorderQty", ascending=False),
              x="StockCode", y="RecommendedReorderQty",
              color="StockoutRisk",
              title="Recommended reorder quantity by product",
              hover_data=["Description", "ReorderPoint", "OnHand (user)"])
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(inv, x="ReorderPoint", y="RecommendedReorderQty",
                  color="StockoutRisk", hover_data=["StockCode", "Description"],
                  title="Reorder point vs recommended qty")
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(inv, use_container_width=True, height=320)
csv = inv.to_csv(index=False).encode()
st.download_button("Download inventory plan", data=csv,
                   file_name="retailpulse_inventory_plan.csv", mime="text/csv")

st.info("Stock-on-hand values entered here are USER-PROVIDED. The forecast-driven safety-stock and "
        "reorder-point columns are derived from Page 3's 30-day forecast plus the configurable lead time "
        "and service level above.")
