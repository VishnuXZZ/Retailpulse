# 🛒 RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform

> End-to-end retail analytics: data ingest → RFM segmentation → 30-day demand forecast → time-aware churn → inventory replenishment → MLOps drift monitor → interactive Streamlit dashboard.
>
> Zidio Development · Data Science & Analytics Domain · March 2026 edition

---

## ✨ What's inside

| Page | What you see | Data used |
|---|---|---|
| 🧾 **1. Sales Dashboard** | Total revenue, AOV, top products, country split, monthly / weekly trends, CSV export | Cleaned non-cancelled transactions |
| 👥 **2. Customer Analytics** | RFM scores, K-Means clusters, segment distribution, spend per segment, downloadable customer table | RFM parquet |
| 📈 **3. Demand Forecasting** | 30-day product + aggregate forecasts with confidence bands, MAE / RMSE / WAPE vs seasonal-naive baseline | HistGradientBoostingRegressor + calendar lags |
| ⚠️ **4. Churn Risk** | Scores, risk bands, top-150 retention list, time-aware label window, retention guidance | RandomForest on time-aware features |
| 📦 **5. Inventory Optimization** | Reorder point, safety stock, recommended qty (configurable lead time / service level / on-hand) | Forecast + configurable user inputs |
| 🛠️ **6. Monitoring & MLOps** | Population Stability Index drift, MLflow runs, retraining instructions | PSI + MLflow |

---

## 🧪 Verified run output (this build, real model artefacts)

The following results were measured from the current RetailPulse pipeline. Results may vary if the dataset, preprocessing, or model configuration changes.

Data Processing

Raw transactions: 1,067,371

Cleaned sales rows: 793,609

Customers in the processed dataset: 5,878

Products: 4,631

Invoices: 36,969

Processed revenue: £17,685,460.64

Data range: 1 December 2009 – 9 December 2011

Customer Segmentation

RFM segments: 8 business-friendly customer categories

K-Means silhouette score: 0.44285 (previously measured result; verify against the latest segmentation artefact before submission).

Churn Prediction

Validation snapshot: 5,249 customers

Validation ROC-AUC: 0.8030

F1-score: 0.8130

Precision: 0.7779

Recall: 0.8514

Precision@Top 20%: 0.9361

Churn labels use a 90-day future window and a 60-day inactivity threshold.

These results describe the historical validation snapshot and should not be interpreted as guaranteed future performance.

Demand Forecasting

Model: HistGradientBoostingRegressor with a seasonal-naive comparison.

Current aggregate ensemble MAPE: approximately 24.61%.

Original MAPE target: 12% or lower.

The current model has not achieved the target. Forecast accuracy needs further improvement before production use.

Inventory Optimization

Lead time, service level, and on-hand stock are configurable in the dashboard.

Reorder recommendations depend on the forecast and the user's stock inputs.

The source dataset does not provide actual on-hand inventory values.

Limitations

Forecasting and churn results are based on historical data and may not generalize to future retail conditions.

The churn validation includes customers observed in earlier training snapshots; repeated customer snapshots can be correlated.

Public deployment is not yet confirmed.

Report measured metrics honestly; do not present targets as achieved results.

---

## ⚙️ Tech stack

- Python 3.11 (Python 3.12 also works in this sandbox), pandas, numpy
- scikit-learn (RFM K-Means, RandomForest, HistGradientBoostingRegressor), scipy (z-scores)
- MLflow (file-store tracking at `mlruns/`)
- Streamlit + Plotly dashboard
- Prophet imported best-effort; gradient boosters are the production default per master-prompt rule (CPU-only, dataset scale)

Optional: PostgreSQL, Docker.

---

## 🪟 Windows PowerShell — exact setup

```powershell
# 1) install Python 3.11 from python.org with "Add to PATH"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the workbook at `data/raw/online_retail_II.xlsx` (do **not** commit it).

```powershell
# 2) build pipeline artefacts
python -m src.build_processed

# 3) launch Streamlit
streamlit run app.py
```

Open http://localhost:8501.

Run tests:

```powershell
python -m pytest tests -q
```

Reset & rerun:

```powershell
Remove-Item -Recurse -Force data/processed, models, mlruns
python -m src.build_processed
```

---

## 🐍 Project structure

```
RetailPulse/
├── app.py
├── pages/
│   ├── 1_Sales_Dashboard.py
│   ├── 2_Customer_Analytics.py
│   ├── 3_Demand_Forecasting.py
│   ├── 4_Churn_Analysis.py
│   ├── 5_Inventory_Optimization.py
│   └── 6_Model_Monitoring.py
├── src/
│   ├── build_raw.py          streaming xlsx → parquet
│   ├── build_processed.py    end-to-end pipeline orchestrator
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── customer_segmentation.py
│   ├── forecasting.py
│   ├── churn_model.py
│   ├── inventory_optimizer.py
│   ├── monitoring.py         (PSI drift)
│   └── evaluation.py
├── data/raw/                 xlsx (gitignored)
├── data/processed/           parquet artefacts
├── models/                   trained pkl artefacts
├── mlruns/                   MLflow file store
├── reports/data_summary.json measured summary
├── tests/                    pytest suite
├── docs/RetailPulse_Report.pdf
├── screenshots/              dashboard captures
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📜 Demo video script (4 – 8 min)

1. Open **0:00–0:30** — business problem, RetailPulse value proposition.
2. **0:30–2:00** — walk through Sales Dashboard, filtering, KPI cards.
3. **2:00–3:30** — Customer Analytics → RFM segments, K-Means silhouette pick, segment contributions.
4. **3:30–5:00** — Forecasting → 30-day aggregate + per-product forecast with confidence band, MAE / WAPE table.
5. **5:00–6:30** — Churn Risk → risk bands, retention suggestions.
6. **6:30–7:30** — Inventory → change lead time & service level live, show reorder math.
7. **7:30–8:00** — Monitoring → PSI drift + retraining instructions; close with business impact numbers.

---

## ⚠️ What is honest about this build

- KPIs come from the actual cleaned dataset; nothing is hard-coded.
- The on-hand stock values on the Inventory page are **user-supplied**; the dataset does not contain them.
- LSTM and Prophet+LSTM ensemble are intentionally bypassed for the production path per the master prompt's CPU-only / dataset-scale rationale. Prophet is installed for reference but did not improve the gradient booster here.
- AUC ≥ 0.88 and MAPE ≤ 12 % are treated as targets to *try* — we report actual measured metrics instead of claiming them.

## 🚀 Public deployment (not done in this build)

We prepared for **Streamlit Community Cloud** deployment but did not push the actual public URL from this sandbox — the spiral-monitor must verify before claiming any live link.
