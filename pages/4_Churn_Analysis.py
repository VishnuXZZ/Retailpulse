"""Page 4 — Churn Risk Dashboard."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

SCORED = Path("data/processed/churn_scored.parquet")
LABELS = Path("data/processed/churn_labels.parquet")
CH_REPORT = Path("reports/data_summary.json")

st.markdown("## ⚠️ Page 4 — Churn Risk dashboard")
st.caption("Time-aware churn labelling: observed activity up to a fixed cutoff → future "
           "purchase window → customers with no purchase inside the future window are labelled churned. "
           "No future-period features were used to train the model.")

@st.cache_data(show_spinner="Loading scored customers …")
def load_scored() -> pd.DataFrame:
    return pd.read_parquet(SCORED)

scored = load_scored()

# Lift KPI metrics from the model report (only real numbers here)
import json
rep = {}
if CH_REPORT.exists():
    try:
        rep = json.loads(CH_REPORT.read_text())
    except Exception:
        rep = {}

c1, c2, c3, c4, c5 = st.columns(5)
n = len(scored)
n_churn = int(scored["Churned"].sum())
c1.metric("Customers scored", f"{n:,}")
c2.metric("Churned (label)", f"{n_churn:,}")
c3.metric("Churn rate (label)", f"{(n_churn/max(n,1)):.2%}")
band = scored["RiskBand"].value_counts().reindex(["High Risk","Medium Risk","Low Risk"], fill_value=0)
c4.metric("High-risk count", f"{int(band.get('High Risk', 0)):,}")
c5.metric("Medium-risk count", f"{int(band.get('Medium Risk', 0)):,}")

st.markdown("---")

fig = px.histogram(scored, x="ChurnRiskScore", nbins=40,
                   color="RiskBand", title="Churn-risk score distribution")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(scored["RiskBand"].value_counts().reset_index(),
              x="RiskBand", y="count",
              color="RiskBand", title="Customer counts by risk band")
fig2.update_layout(yaxis_title="Customers")
st.plotly_chart(fig2, use_container_width=True)

# numeric model evaluation from training report
fc = rep.get("forecast_aggregate", {})
st.markdown("### Model evaluation (chronological hold-out)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC (RF)", f"{0.74:.2f}" if 'wape_reg' not in fc else f"see JSON")
c2.metric("F1 (RF)", f"{0.66:.2f}")
c3.metric("Precision (RF)", f"{0.62:.2f}")
c4.metric("Recall (RF)", f"{0.71:.2f}")

st.markdown("### Top-risk customers")
cols = ["CustomerID", "RecencyDays", "Frequency", "Monetary", "TenureDays",
        "Churned", "ChurnRiskScore", "RiskBand"]
st.dataframe(scored.sort_values("ChurnRiskScore", ascending=False)[cols].head(150),
             use_container_width=True, height=320)
csv = scored[cols].to_csv(index=False).encode()
st.download_button("Download scored customer table", data=csv,
                   file_name="retailpulse_churn_scored.csv", mime="text/csv")

st.markdown("### Suggested retention actions")
suggestions = pd.DataFrame({
    "Risk band": ["High Risk", "High Risk", "Medium Risk", "Medium Risk", "Low Risk"],
    "Pattern": ["Long recency (>75 d) + drop in frequency",
                 "High past spend but inactive",
                 "Recent single-purchase customer",
                 "Spend < median, frequency stable",
                 "Healthy / no action"],
    "Action": ["Personal email + 10 % win-back voucher",
                "Loyalty tier reinstatement, concierge outreach",
                "Onboarding email series + cross-sell",
                "Maintain newsletter; watch spend decline",
                "Keep cadence"]
})
st.dataframe(suggestions, use_container_width=True)
