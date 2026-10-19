"""End-to-end build: raw -> cleaned -> RFM & segmentation -> forecasting -> churn -> inventory -> drift.

Run with: python3 -m src.build_processed
"""
from __future__ import annotations
import sys, time, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import mlflow

from src.data_ingestion import load_all, save_parquet
from src.data_cleaning import basic_clean, splits_and_policy, feature_engineer
from src.feature_engineering import (rfm_table, rfm_scores, segment_label,
                                     product_demand, make_churn_labels, daily_demand)
from src.customer_segmentation import cluster_kmeans, sweep_k
from src.forecasting import forecast_product, forecast_aggregate, try_prophet
from src.churn_model import train_churn, score_all_customers
from src.inventory_optimizer import daily_std, recommend
from src.monitoring import drift_report

PROCESSED = Path("data/processed")
ARTIFACT = Path("models")
REPORTS = Path("reports")
MLRUNS = Path("mlruns")

MLRUNS.mkdir(exist_ok=True, parents=True)
PROCESSED.mkdir(exist_ok=True, parents=True)
ARTIFACT.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("retail-pulse")

def build_churn_snapshots(
    sales: pd.DataFrame,
    snapshot_dates: list[pd.Timestamp],
    future_window_days: int = 90,
    churn_threshold_days: int = 60,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build customer features and labels at multiple observation dates."""
    snapshots = []
    infos = []

    for snapshot_date in snapshot_dates:
        snapshot_date = pd.Timestamp(snapshot_date).normalize()

        if sales["InvoiceDate"].max() < (
            snapshot_date + pd.Timedelta(days=future_window_days)
        ):
            continue

        snapshot_df, info = make_churn_labels(
            sales,
            snapshot_date,
            future_window_days=future_window_days,
            churn_threshold_days=churn_threshold_days,
        )

        snapshot_df["ObsEnd"] = snapshot_date
        snapshots.append(snapshot_df)
        infos.append(info)

    if not snapshots:
        raise ValueError(
            "No valid churn snapshots. Check the dataset date range."
        )

    return pd.concat(snapshots, ignore_index=True), infos



def main():
    t0 = time.time()
    print("Loading workbook …")
    raw = load_all()
    save_parquet(raw, PROCESSED / "raw.parquet")
    print("Cleaning …")
    df, clean_summary = basic_clean(raw)
    sales, cancellations, policy_summary = splits_and_policy(df)
    sales_fe = feature_engineer(sales)
    save_parquet(sales_fe, PROCESSED / "sales_clean.parquet")

    obs_end = sales_fe["InvoiceDate"].max().normalize()
    print(f"Observation end: {obs_end.date()}  rows={len(sales_fe):,}")

    # === RFM & segmentation ===
    rfm = rfm_table(sales_fe, obs_end)
    rfm_scored = rfm_scores(rfm)
    rfm_seg = segment_label(rfm_scored)

    with mlflow.start_run(run_name="rfm_kmeans"):
        best_k = 5
        sweep = sweep_k(rfm_seg, k_range=range(3, 8))
        sweep_best = sweep.sort_values("silhouette", ascending=False).iloc[0]
        best_k = int(sweep_best["k"])
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("silhouette", float(sweep_best["silhouette"]))
        rfm_clust, info = cluster_kmeans(rfm_seg, k=best_k)
        for col in ("Churned",):
            pass
    rfm_seg["ClusterKMeans"] = rfm_clust["KMeansCluster"]
    rfm_seg.to_parquet(PROCESSED / "rfm_segments.parquet", index=False)

    # === Forecasting ===
    daily_agg = sales_fe.groupby(sales_fe["InvoiceDate"].dt.normalize())["Quantity"].sum().reset_index()
    daily_agg.columns = ["Date", "Quantity"]
    daily_agg.to_parquet(PROCESSED / "daily_aggregate.parquet", index=False)
    daily_agg["Revenue"] = sales_fe.groupby(sales_fe["InvoiceDate"].dt.normalize())["RevenueGross"].sum().values
    daily_agg[["Date", "Quantity"]].to_parquet(PROCESSED / "daily_aggregate.parquet", index=False)

    fcst_agg = forecast_aggregate(daily_agg, horizon=30)
    print("Aggregate forecast metrics:", fcst_agg.get("metrics"))
    if isinstance(fcst_agg, dict) and "metrics" in fcst_agg:
        with mlflow.start_run(run_name="forecast_aggregate"):
            for k, v in fcst_agg["metrics"].items():
                mlflow.log_metric(k, v)
            mlflow.log_param("model", "HistGradientBoostingRegressor (lags+calendar)")

    # Per-product (top 10 by revenue)
    prod_demand = product_demand(sales_fe)
    prod_demand.to_parquet(PROCESSED / "product_demand.parquet", index=False)
    fcst_metrics = forecast_product(
        sales_fe, horizon=30, top_k=10
    )
    per_prod_metrics_df = fcst_metrics[1] if isinstance(fcst_metrics, tuple) else fcst_metrics
    print("Per-product forecast metrics (head):")
    print(per_prod_metrics_df.head())

    # Prophet attempt (best-effort, may be unavailable; not the production fallback)
    if fcst_agg and "forecasts" in fcst_agg and not fcst_agg["forecasts"].empty:
        try:
            p = try_prophet(daily_agg[["Date", "Quantity"]], horizon=30)
            print("Prophet attempt:", {k: type(v).__name__ for k, v in (p or {}).items()})
        except Exception as e:
            print("Prophet attempt failed:", e)

    
    # === Churn ===
    sales_fe["SalesWeek"] = (
        sales_fe["InvoiceDate"].dt.to_period("W").astype(str)
    )

    # Build snapshots with separate future label windows.
    first_date = sales_fe["InvoiceDate"].min().normalize()
    last_date = sales_fe["InvoiceDate"].max().normalize()

    # Monthly observation dates, leaving enough time for future labels.
    snapshot_dates = list(
        pd.date_range(
            start=first_date + pd.Timedelta(days=180),
            end=last_date - pd.Timedelta(days=90),
            freq="MS",
        )
    )

    churn_df, churn_infos = build_churn_snapshots(
        sales_fe,
        snapshot_dates,
        future_window_days=90,
        churn_threshold_days=60,
    )

    churn_df.to_parquet(PROCESSED / "churn_labels.parquet", index=False)
    print("Churn snapshots:", len(churn_infos))
    print("Churn label summaries:", churn_infos)

    # Hold out the latest snapshot date for validation.
    available_dates = sorted(churn_df["ObsEnd"].dropna().unique())

    if len(available_dates) < 2:
        raise ValueError(
            "Need at least two valid snapshot dates for temporal validation."
        )

    val_date = pd.Timestamp(available_dates[-1])
    train_df = churn_df[churn_df["ObsEnd"] < val_date].copy()
    val_df = churn_df[churn_df["ObsEnd"] == val_date].copy()

    # Train only on earlier snapshots and evaluate on the latest snapshot.
    with mlflow.start_run(run_name="churn"):
        ch_results = train_churn(
            pd.concat([train_df, val_df], ignore_index=True),
            train_end=pd.Timestamp(available_dates[-2]),
            val_end=val_date,
        )

        for name, res in ch_results.items():
            mlflow.log_metrics({
                f"{name}_{k}": float(v)
                for k, v in res["metrics"].items()
            })

        print("Churn metrics:", {
            name: result["metrics"]
            for name, result in ch_results.items()
        })

    # Score the latest customer snapshot using the trained RF model.
    latest_snapshot = churn_df[churn_df["ObsEnd"] == val_date].copy()
    scored = score_all_customers(
        latest_snapshot,
        ch_results["random_forest"]["model"],
    )
    scored.to_parquet(PROCESSED / "churn_scored.parquet", index=False)


    # === Inventory ===
    daily = sales_fe.copy()
    daily["Date"] = daily["InvoiceDate"].dt.normalize()
    daily_std_pd = daily_std(prod_demand, daily.rename(columns={"Quantity": "Quantity"}).groupby(["StockCode", "Date"], as_index=False)["Quantity"].sum())
    daily_std_pd.to_parquet(PROCESSED / "product_demand_with_stats.parquet", index=False)
    fcst_df = pd.read_parquet(PROCESSED / "forecasts.parquet") if (PROCESSED / "forecasts.parquet").exists() else pd.DataFrame(columns=["Date","Forecast","Lower","Upper","StockCode"])
    inv = recommend(daily_std_pd, fcst_df, on_hand_lookup={}, lead_time_days=7, service_level=0.95)
    inv.to_parquet(PROCESSED / "inventory_recommendations.parquet", index=False)

    # === Drift snapshot: reference = earliest half, current = latest half ===
    cust_dates = sales_fe.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"]).reset_index()
    cutoff = cust_dates["max"].quantile(0.5)
    ref_cust = cust_dates[cust_dates["max"] <= cutoff]["CustomerID"]
    cur_cust = cust_dates[cust_dates["max"] > cutoff]["CustomerID"]
    ref_rfm = rfm_seg[rfm_seg["CustomerID"].isin(ref_cust)]
    cur_rfm = rfm_seg[rfm_seg["CustomerID"].isin(cur_cust)]
    rep = drift_report(ref_rfm, cur_rfm) if len(ref_rfm) and len(cur_rfm) else pd.DataFrame()
    rep.to_parquet(PROCESSED / "drift_report.parquet", index=False)

    # === Summary metrics ===
    summary = {
        "raw_rows": int(len(raw)),
        "clean_sales_rows": int(len(sales_fe)),
        "cancellations_rows": int(len(cancellations)),
        "unique_customers": int(sales_fe["CustomerID"].nunique()),
        "unique_products": int(sales_fe["StockCode"].nunique()),
        "total_revenue_gbp": round(float(sales_fe["RevenueGross"].sum()), 2),
        "unique_invoices": int(sales_fe["Invoice"].nunique()),
        "date_min": str(sales_fe["InvoiceDate"].min().date()),
        "date_max": str(sales_fe["InvoiceDate"].max().date()),
        "rfm_n_segments": int(rfm_seg["Segment"].nunique()),
        "kmeans_silhouette": float(info["silhouette"]),
        "churn": churn_infos,
        "forecast_aggregate": fcst_agg.get("metrics") if isinstance(fcst_agg, dict) else {},
        "process_seconds": round(time.time() - t0, 2),
    }
    (REPORTS / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
