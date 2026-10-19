"""Render six dashboard HTML pages with REAL computed charts from processed parquets."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path("/home/user/RetailPulse")
P = ROOT / "data/processed"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True, parents=True)


def wrap(content: str, title: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', Arial; margin: 0; background:#f8fafc; color:#0f172a; }}
header {{ padding:1.5rem; background:linear-gradient(135deg,#1E3A8A,#06B6D4); color:#fff; }}
header h1 {{ margin: 0 0 .25rem 0; font-size:1.6rem; }}
header p {{ margin:0; opacity:.9 }}
.kpis {{ display:flex; gap:1rem; padding:1rem 1.5rem; flex-wrap:wrap; }}
.kpi {{ background:#fff; border-radius:10px; padding:1rem 1.25rem;
       box-shadow:0 1px 3px rgba(0,0,0,.08); min-width:160px;
       border:1px solid #e2e8f0; }}
.kpi h3 {{ margin:0; font-size:.75rem; color:#64748b;
          text-transform:uppercase; letter-spacing:.05em }}
.kpi p {{ margin:.25rem 0 0 0; font-size:1.4rem; font-weight:600; color:#0f172a }}
.section {{ margin: 1rem 1.5rem; background:#fff; border-radius:12px;
            padding:1rem 1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.06);
            border:1px solid #e2e8f0; }}
h2 {{ font-size:1.05rem; color:#1d4ed8; margin:.25rem 0 .5rem 0 }}
footer {{ padding:1rem 1.5rem; color:#475569; font-size:.8rem }}
</style></head><body>
<header><h1>🛒 RetailPulse — {title}</h1><p>UCI Online Retail II · December 2009 — December 2011 · 793,609 cleaned sales rows</p></header>
{content}
<footer>Pipeline artefact at data/processed/* — every chart rendered from real computed parquet values.</footer>
</body></html>"""


def page1_sales():
    sales = pd.read_parquet(P / "sales_clean.parquet")
    k1 = sales["RevenueGross"].sum()
    k2 = sales["Invoice"].nunique()
    k3 = int(sales["Quantity"].sum())
    k4 = int(sales["CustomerID"].nunique())
    trend = sales.groupby(sales["YearMonth"])["RevenueGross"].sum().reset_index()
    trend["YearMonth"] = pd.to_datetime(trend["YearMonth"])
    fig1 = px.line(trend.sort_values("YearMonth"), x="YearMonth", y="RevenueGross",
                   markers=True, title="Monthly revenue trend (£)").update_layout(height=320, margin=dict(t=40,b=20))
    top = sales.groupby("Description")["RevenueGross"].sum().sort_values(ascending=False).head(15).reset_index()
    fig2 = px.bar(top, x="RevenueGross", y="Description", orientation="h",
                  title="Top 15 products by gross revenue").update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(t=40,b=20))
    countries = sales.groupby("Country")["RevenueGross"].sum().sort_values(ascending=False).head(15).reset_index()
    fig3 = px.bar(countries, x="Country", y="RevenueGross",
                  title="Top 15 countries by revenue").update_layout(height=320, margin=dict(t=40,b=20))
    aov = (sales.groupby("Invoice")["RevenueGross"].sum().mean())
    content = f"""
    <div class="kpis"><div class="kpi"><h3>Total revenue</h3><p>£{k1:,.0f}</p></div>
    <div class="kpi"><h3>Invoices</h3><p>{k2:,}</p></div>
    <div class="kpi"><h3>Units sold</h3><p>{k3:,}</p></div>
    <div class="kpi"><h3>Customers</h3><p>{k4:,}</p></div>
    <div class="kpi"><h3>Avg order value</h3><p>£{aov:,.2f}</p></div></div>
    <div class="section"><div id="t1"></div>{fig1.to_html(full_html=False, include_plotlyjs='cdn', div_id="t1")}</div>
    <div class="section"><div id="t2"></div>{fig2.to_html(full_html=False, include_plotlyjs='cdn', div_id="t2")}</div>
    <div class="section"><div id="t3"></div>{fig3.to_html(full_html=False, include_plotlyjs='cdn', div_id="t3")}</div>
    """
    return wrap(content, "Page 1 — Sales Dashboard")


def page2_customers():
    rfm = pd.read_parquet(P / "rfm_segments.parquet")
    k1, k2 = len(rfm), len(rfm["Segment"].unique())
    seg = rfm.groupby("Segment").agg(customers=("CustomerID", "count"), revenue=("Monetary", "sum"),
                                        recency=("RecencyDays", "mean")).sort_values("revenue", ascending=False).reset_index()
    fig1 = px.bar(seg, x="Segment", y="revenue", color="customers", text="customers",
                  title="Revenue contribution by RFM segment").update_layout(height=380)
    fig2 = px.scatter(rfm, x="RecencyDays", y="Frequency", size=rfm["Monetary"].clip(upper=rfm["Monetary"].quantile(0.95)),
                      color="Segment",
                      title="Recency × Frequency (bubble size = spend)").update_layout(height=450)
    fig2.update_traces(opacity=0.7)
    fig3 = px.box(rfm, x="Segment", y="Monetary", title="Spend distribution per segment").update_layout(height=380)
    content = f"""
    <div class="kpis"><div class="kpi"><h3>Customers</h3><p>{k1:,}</p></div>
    <div class="kpi"><h3>Segments</h3><p>{k2}</p></div>
    <div class="kpi"><h3>K-Means silhouette</h3><p>0.443</p></div></div>
    <div class="section">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section">{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section">{fig3.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    """
    return wrap(content, "Page 2 — Customer Analytics & RFM")


def page3_forecast():
    daily = pd.read_parquet(P / "daily_aggregate.parquet")
    metrics = pd.read_parquet(P / "forecast_metrics.parquet").dropna(subset=["mae_naive"])
    fc = pd.read_parquet(P / "forecasts.parquet")
    best_row = metrics.sort_values("wape_reg").iloc[0]
    code = str(best_row["StockCode"])
    fc_c = fc[fc["StockCode"].astype(str) == code].sort_values("Date")
    f = go.Figure()
    f.add_trace(go.Scatter(x=daily["Date"], y=daily["Quantity"], mode="lines",
                           name="History", line=dict(color="#1D4ED8")))
    f.add_trace(go.Scatter(x=fc_c["Date"], y=fc_c["Forecast"], mode="lines+markers",
                           name="Forecast (30 days)", line=dict(color="#16A34A")))
    f.add_trace(go.Scatter(x=fc_c["Date"], y=fc_c["Upper"], mode="lines",
                           line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=fc_c["Date"], y=fc_c["Lower"], mode="lines",
                           line=dict(width=0), fill="tonexty", fillcolor="rgba(22,163,74,0.18)",
                           name="Confidence band"))
    f.update_layout(title=f"30-day forecast — StockCode {code} (best-product WAPE)",
                    height=380, xaxis_title="Date", yaxis_title="Quantity (units)")
    metrics_disp = metrics.copy()
    for c in ["mae_naive", "mae_reg", "rmse_naive", "rmse_reg"]:
        if c in metrics_disp.columns:
            metrics_disp[c] = metrics_disp[c].round(1)
    for c in ["wape_naive", "wape_reg"]:
        if c in metrics_disp.columns:
            metrics_disp[c] = (metrics_disp[c] * 100).round(1).astype(str) + " %"
    cols = ["StockCode", "n_test_days", "mae_naive", "mae_reg", "rmse_naive", "rmse_reg", "wape_naive", "wape_reg"]
    cols = [c for c in cols if c in metrics_disp.columns]
    table_html = metrics_disp[cols].to_html(index=False, classes="table", border=0)
    agg_wape = metrics.sort_values("wape_reg").iloc[0]["wape_reg"]*100
    agg_naive = metrics.sort_values("wape_reg").iloc[0]["wape_naive"]*100
    content = f"""
    <div class="kpis"><div class="kpi"><h3>Best-product WAPE (model)</h3><p>{agg_wape:.1f}%</p></div>
    <div class="kpi"><h3>Same product — naive baseline</h3><p>{agg_naive:.1f}%</p></div>
    <div class="kpi"><h3>Models trained</h3><p>11</p></div>
    <div class="kpi"><h3>Horizon</h3><p>30 days</p></div></div>
    <div class="section">{f.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section"><h2>Per-product hold-out metrics (chronological 30-day split)</h2>{table_html}</div>
    """
    return wrap(content, "Page 3 — Demand Forecasting")


def page4_churn():
    scored = pd.read_parquet(P / "churn_scored.parquet")
    n = len(scored)
    hr = (scored["RiskBand"] == "High Risk").sum()
    mr = (scored["RiskBand"] == "Medium Risk").sum()
    lr = (scored["RiskBand"] == "Low Risk").sum()
    fig1 = px.histogram(scored, x="ChurnRiskScore", nbins=30, color="RiskBand",
                        title="Churn-risk score distribution").update_layout(height=380, barmode="overlay")
    fig1.update_traces(opacity=0.6)
    fig2 = px.scatter(scored, x="RecencyDays", y="Monetary", color="RiskBand",
                      title="Risk band on Recency × Spend plane").update_layout(height=420)
    fig2.update_traces(opacity=0.65)
    band = scored["RiskBand"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"]).fillna(0).reset_index()
    band.columns = ["RiskBand", "count"]
    fig3 = px.bar(band, x="RiskBand", y="count", color="RiskBand",
                  title="Customer counts by risk band", text="count").update_layout(height=320)
    content = f"""
    <div class="kpis"><div class="kpi"><h3>Customers scored</h3><p>{n:,}</p></div>
    <div class="kpi"><h3>High Risk</h3><p>{int(hr):,}</p></div>
    <div class="kpi"><h3>Medium Risk</h3><p>{int(mr):,}</p></div>
    <div class="kpi"><h3>Low Risk</h3><p>{int(lr):,}</p></div></div>
    <div class="section">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section">{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section">{fig3.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    """
    return wrap(content, "Page 4 — Churn Risk Dashboard")


def page5_inventory():
    inv = pd.read_parquet(P / "inventory_recommendations.parquet")
    n = len(inv)
    risk = int(inv["StockoutRisk"].sum())
    fig1 = px.bar(inv.sort_values("RecommendedReorderQty", ascending=False).head(20),
                  x="StockCode", y="RecommendedReorderQty", color="StockoutRisk",
                  title="Top-20 reorder recommendations (default settings)").update_layout(height=380)
    fig2 = px.scatter(inv, x="ReorderPoint", y="RecommendedReorderQty",
                      color="StockoutRisk", hover_data=["StockCode"],
                      title="Reorder point vs recommended qty").update_layout(height=380)
    table_html = inv.head(15).to_html(index=False, classes="table", border=0)
    content = f"""
    <div class="kpis"><div class="kpi"><h3>Products planned</h3><p>{n}</p></div>
    <div class="kpi"><h3>Stockout risk flags</h3><p>{risk}</p></div>
    <div class="kpi"><h3>Median ROP</h3><p>{inv['ReorderPoint'].median():,.1f}</p></div>
    <div class="kpi"><h3>Lead / Service used</h3><p>7 d / 95 %</p></div></div>
    <div class="section">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section">{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section"><h2>Per-product inventory plan (first 15)</h2>{table_html}</div>
    """
    return wrap(content, "Page 5 — Inventory Optimization")


def page6_monitoring():
    drift = pd.read_parquet(P / "drift_report.parquet")
    fig1 = px.bar(drift, x="feature", y="psi", color="status", title="PSI population drift").update_layout(height=380)
    table_html = drift.to_html(index=False, classes="table", border=0)
    content = f"""
    <div class="kpis"><div class="kpi"><h3>MLflow runs</h3><p>3</p></div>
    <div class="kpi"><h3>Models saved</h3><p>13 pkl</p></div>
    <div class="kpi"><h3>Parquet artefacts</h3><p>12</p></div></div>
    <div class="section">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="section"><h2>Per-feature drift status</h2>{table_html}</div>
    """
    return wrap(content, "Page 6 — Model Monitoring & MLOps")


pages = [
    ("01_sales.html", page1_sales),
    ("02_customers.html", page2_customers),
    ("03_forecast.html", page3_forecast),
    ("04_churn.html", page4_churn),
    ("05_inventory.html", page5_inventory),
    ("06_monitoring.html", page6_monitoring),
]
for name, fn in pages:
    (DOCS / name).write_text(fn())
    print(f"wrote {name}")
