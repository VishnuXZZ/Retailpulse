"""Tests for RetailPulse data + modeling helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pytest
from src.data_cleaning import basic_clean
from src.feature_engineering import rfm_table, rfm_scores, segment_label
from src.forecasting import seasonal_naive, wape
from src.churn_model import train_churn
from src.inventory_optimizer import recommend
from src.monitoring import psi


def _toy_df():
    return pd.DataFrame({
        "Invoice": ["A1","A1","A2","A2","C1","A3","A3"],
        "StockCode": ["S1","S1","S2","S2","S1","S3","S3"],
        "Description": ["Pen","Pen","Book","Book","Pen","Cup","Cup"],
        "Quantity": [2, 1, 1, 1, -2, 3, 1],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01","2011-01-15","2011-02-01","2011-02-15",
            "2011-02-20","2011-03-01","2011-04-01"]),
        "Price": [1.0, 1.0, 5.0, 5.0, 1.0, 2.0, 2.0],
        "Customer ID": [10, 10, 11, 11, 10, 12, 12],
        "Country": ["UK","UK","UK","UK","UK","FR","FR"],
    })


def test_basic_clean_segments():
    raw = _toy_df()
    df, summary = basic_clean(raw)
    assert "InvoiceNoCancel" in df.columns
    assert summary["duplicates_dropped"] >= 0
    n_cancelled = df["InvoiceNoCancel"].sum()
    assert n_cancelled < len(df)


def test_rfm_pipeline():
    raw = _toy_df()
    df, _ = basic_clean(raw)
    sales = df[(df["InvoiceNoCancel"]) & (df["Quantity"] > 0) & (df["Price"] > 0)].dropna(subset=["CustomerID"])
    rfm = rfm_table(sales, sales["InvoiceDate"].max())
    assert set(["RecencyDays", "Frequency", "Monetary"]) <= set(rfm.columns)
    rfm_scored = rfm_scores(rfm)
    assert {"R_Score", "F_Score", "M_Score"}.issubset(set(rfm_scored.columns))
    rfm_seg = segment_label(rfm_scored)
    assert "Segment" in rfm_seg.columns
    assert rfm_seg["Segment"].nunique() >= 1


def test_seasonal_naive_repeats():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7])
    pred = seasonal_naive(s, season=7, horizon=14)
    assert len(pred) == 14
    # cyclic: first prediction is the last in-window value
    assert float(pred[0]) == float(s.iloc[-1])


def test_wape_zero_when_perfect():
    y = np.array([1, 2, 3])
    assert wape(y, y) == 0.0


def test_psi_stable_when_identical():
    a = np.random.RandomState(0).randn(10000)
    score = psi(a, a.copy(), bins=10)
    assert score < 1e-6


def test_psi_drift_when_distributions_shift():
    a = np.random.RandomState(0).randn(10000)
    b = np.random.RandomState(1).randn(10000) + 2
    score = psi(a, b, bins=10)
    assert score > 0.05


def test_recommend_returns_columns():
    out = recommend(pd.DataFrame({"StockCode": ["X"], "Description": ["Pen"], "AvgPrice": [1.0],
                                    "AvgDailyDemand": [5.0], "__daily_std__": [1.0]}),
                     pd.DataFrame({"StockCode": ["X"], "Date": pd.to_datetime(["2025-01-01"]),
                                   "Forecast": [5]}), {}, lead_time_days=7, service_level=0.95)
    assert {"ReorderPoint","RecommendedReorderQty","OnHand","SafetyStock"}.issubset(set(out.columns))
