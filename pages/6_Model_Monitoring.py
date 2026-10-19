"""Page 6 — Model Monitoring & MLOps."""
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px

MLRUNS = Path("mlruns")
ARTIFACT = Path("models")
DRIFT = Path("data/processed/drift_report.parquet")
REPORT = Path("reports/data_summary.json")

st.markdown("## 🛠️ Page 6 — Model Monitoring & MLOps")
st.caption("Lightweight transparent drift monitor (Population Stability Index on RFM). "
           "MLflow tracking runs live in the local `mlruns/` folder under URI `file:./mlruns`.")

@st.cache_data
def load_drift() -> pd.DataFrame:
    return pd.read_parquet(DRIFT) if DRIFT.exists() else pd.DataFrame()

drift = load_drift()
summary = json.loads(REPORT.read_text()) if REPORT.exists() else {}

c1, c2, c3, c4 = st.columns(4)
runs = []
if MLRUNS.exists():
    for p in MLRUNS.rglob("meta.yaml"):
        runs.append(p.parent.name)
n_runs = len(set(runs))
c1.metric("MLflow experiments logged", f"{n_runs}")
c2.metric("Models saved on disk", f"{len(list(ARTIFACT.glob('*.pkl')))}")
c3.metric("Parquet-derived artefacts", f"{len(list(Path('data/processed').glob('*.parquet')))}")
c4.metric("Process seconds", f"{summary.get('process_seconds','-')}")

st.markdown("### Drift (PSI, log-scaled features)")
if not drift.empty:
    fig = px.bar(drift, x="feature", y="psi", color="status",
                  title="Population Stability Index per RFM feature")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(drift, use_container_width=True, height=200)

st.markdown("### Pipeline heartbeat")
ph = [
    {"step": "01 Ingest", "tool": "openpyxl read-only streaming", "output": "raw.parquet"},
    {"step": "02 Clean", "tool": "pandas", "output": "sales_clean.parquet"},
    {"step": "03 RFM", "tool": "sklearn StandardScaler", "output": "rfm_segments.parquet"},
    {"step": "04 Forecast", "tool": "sklearn HistGradientBoostingRegressor",
     "output": "forecasts.parquet"},
    {"step": "05 Churn", "tool": "sklearn RandomForestClassifier", "output": "churn_scored.parquet"},
    {"step": "06 Inventory", "tool": "SciPy + configurable inputs",
     "output": "inventory_recommendations.parquet"},
    {"step": "07 Drift", "tool": "PSI (custom)", "output": "drift_report.parquet"},
]
st.dataframe(pd.DataFrame(ph), use_container_width=True)

st.markdown("### How to retrain")
st.code("""\
# 1. Drop new raw workbook into data/raw/online_retail_II.xlsx
# 2. Run the full pipeline
python -m src.build_processed
# 3. Re-launch the Streamlit app
streamlit run app.py
""", language="bash")
