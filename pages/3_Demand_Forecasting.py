"""Page 3 — Demand Forecasting Dashboard."""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DAILY = Path("data/processed/daily_aggregate.parquet")
FORECASTS = Path("data/processed/forecasts.parquet")
FCST_METRICS = Path("data/processed/forecast_metrics.parquet")
AGG_FORECAST = Path("data/processed/forecast_aggregate.parquet")

st.markdown("## 📈 Page 3 — Demand Forecasting")
st.caption("Training: HistGradientBoostingRegressor on calendar features + lag/rolling stats. "
           "Evaluated against a seasonal-naive baseline on a chronological hold-out (last 30 days).")

@st.cache_data(show_spinner="Loading daily aggregate …")
def load_daily() -> pd.DataFrame:
    return pd.read_parquet(DAILY)

@st.cache_data(show_spinner="Loading forecasts …")
def load_fcsts() -> pd.DataFrame:
    if not FORECASTS.exists():
        return pd.DataFrame(columns=["Date", "Forecast", "Lower", "Upper", "StockCode"])
    return pd.read_parquet(FORECASTS)

@st.cache_data(show_spinner="Loading metric table …")
def load_metrics() -> pd.DataFrame:
    if not FCST_METRICS.exists():
        return pd.DataFrame()
    return pd.read_parquet(FCST_METRICS)

daily = load_daily()
forecasts = load_fcsts()
metrics = load_metrics()

# KPI bar — aggregate metrics (fall back to a representative row if aggregate model is separate)
# Aggregate forecast metrics
agg_forecast_metrics = {}

if AGG_FORECAST.exists():
    aggregate_forecast = pd.read_parquet(AGG_FORECAST)
else:
    aggregate_forecast = pd.DataFrame()

with st.container():
    c1, c2, c3 = st.columns(3)

    n_test = (
        int(metrics["n_test_days"].iloc[0])
        if not metrics.empty and "n_test_days" in metrics.columns
        else 30
    )

    mae_reg = (
        float(metrics["mae_reg"].mean())
        if not metrics.empty and "mae_reg" in metrics.columns
        else 0.0
    )

    wape_reg = (
        float(metrics["wape_reg"].mean())
        if not metrics.empty and "wape_reg" in metrics.columns
        else 0.0
    )

    c1.metric("Hold-out length", f"{n_test} days")
    c2.metric("Avg product MAE", f"{mae_reg:,.2f}")
    c3.metric("Avg product WAPE", f"{wape_reg:.2%}")

st.caption(
    "Product metrics are averaged across the available evaluated products. "
    "They are not aggregate-demand accuracy metrics."
)

daily_fig = px.line(daily, x="Date", y="Quantity", title="Historical daily aggregate demand")
st.plotly_chart(daily_fig, use_container_width=True)

st.markdown("### Forecast vs actual — top products")
if not forecasts.empty and not metrics.empty:
    codes = sorted(metrics["StockCode"].astype(str).unique().tolist())
    code = st.selectbox("Choose product", codes)
    fc = forecasts[forecasts["StockCode"].astype(str) == code].sort_values("Date")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=daily["Date"], y=daily["Quantity"],
                            mode="lines", name="History", line=dict(color="#1D4ED8")))
    f2.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast"],
                            mode="lines+markers", name="Forecast",
                            line=dict(color="#16A34A")))
    f2.add_trace(go.Scatter(x=fc["Date"], y=fc["Upper"],
                            mode="lines", line=dict(width=0), showlegend=False))
    f2.add_trace(go.Scatter(x=fc["Date"], y=fc["Lower"],
                            mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(22,163,74,0.18)", name="Confidence band"))
    f2.update_layout(title=f"30-day forecast — StockCode {code}",
                       xaxis_title="Date", yaxis_title="Quantity (units)")
    st.plotly_chart(f2, use_container_width=True)

    row = metrics[metrics["StockCode"].astype(str) == code].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline MAE", f"{row.get('mae_naive', 0):,.0f}")
    k2.metric("Model MAE", f"{row.get('mae_reg', 0):,.0f}")
    k3.metric("Baseline WAPE", f"{row.get('wape_naive', 0):.2%}")
    k4.metric("Model WAPE", f"{row.get('wape_reg', 0):.2%}")

st.markdown("### Per-product metric table")
if not metrics.empty:
    cols = ["StockCode", "n_test_days", "mae_naive", "mae_reg", "rmse_naive", "rmse_reg",
            "wape_naive", "wape_reg", "mape_reg"]
    cols = [c for c in cols if c in metrics.columns]
    st.dataframe(metrics[cols], use_container_width=True, height=300)
    csv = metrics[cols].to_csv(index=False).encode()
    st.download_button("Download forecast metrics", data=csv,
                       file_name="retailpulse_forecast_metrics.csv", mime="text/csv")

if not forecasts.empty:
    csv = forecasts.to_csv(index=False).encode()
    st.download_button("Download forecast tables", data=csv,
                       file_name="retailpulse_forecasts.csv", mime="text/csv")

st.info("LSTM and Prophet+LSTM ensembles are intentionally kept out of the production path. CPU-only laptops "
        "and ~1 M-row transactional datasets favour well-engineered gradient boosters, which are the model "
        "evaluated above.")
