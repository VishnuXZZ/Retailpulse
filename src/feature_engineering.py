"""Feature engineering helpers: RFM, time-series aggregates, churn label bands."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rfm_table(sales: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer."""

    revenue_col = (
        "RevenueGross"
        if "RevenueGross" in sales.columns
        else "Revenue"
    )

    grp = sales.groupby("CustomerID").agg(
        RecencyDays=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=(revenue_col, "sum"),
    ).reset_index()

    return grp


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Score 1-5 using qcut; ties share the lower score to avoid same-bin issues."""
    rf = rfm.copy()
    rf["R_Score"] = pd.qcut(rf["RecencyDays"].rank(method="first"), 5,
                            labels=[5, 4, 3, 2, 1]).astype(int)
    rf["F_Score"] = pd.qcut(rf["Frequency"].rank(method="first"), 5,
                            labels=[1, 2, 3, 4, 5]).astype(int)
    rf["M_Score"] = pd.qcut(rf["Monetary"].rank(method="first"), 5,
                            labels=[1, 2, 3, 4, 5]).astype(int)
    rf["RFM_Sum"] = rf[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return rf


def segment_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """Map RFM sums to business-friendly segment names."""
    rfm = rfm.copy()
    def label(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 4:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "Recent Customers"
        if r <= 2 and f >= 4 and m >= 4:
            return "At-Risk VIPs"
        if r <= 2 and f >= 3 and m >= 3:
            return "At-Risk Customers"
        if r >= 3 and f <= 2 and m <= 2:
            return "Low-Value Recent"
        if r <= 2 and f <= 2:
            return "Hibernating"
        return "Potential Loyalists"
    rfm["Segment"] = rfm.apply(label, axis=1)
    return rfm


def daily_demand(sales: pd.DataFrame, stock_code: str | None = None) -> pd.DataFrame:
    df = sales
    if stock_code is not None:
        df = df[df["StockCode"] == stock_code]
    out = df.groupby(df["InvoiceDate"].dt.normalize()).agg(
        Quantity=("Quantity", "sum"),
        Revenue=("RevenueGross", "sum"),
        Invoices=("Invoice", "nunique"),
    ).reset_index().rename(columns={"InvoiceDate": "Date"}).sort_values("Date")
    return out


def weekly_demand(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["WeekStart"] = s["InvoiceDate"].dt.to_period("W").apply(lambda p: p.start_time)
    out = s.groupby("WeekStart").agg(
        Quantity=("Quantity", "sum"),
        Revenue=("RevenueGross", "sum"),
        Invoices=("Invoice", "nunique"),
    ).reset_index().sort_values("WeekStart")
    return out


def product_demand(sales: pd.DataFrame) -> pd.DataFrame:
    out = sales.groupby("StockCode").agg(
        Description=("Description", "first"),
        Quantity=("Quantity", "sum"),
        Revenue=("RevenueGross", "sum"),
        Invoices=("Invoice", "nunique"),
        AvgPrice=("Price", "mean"),
    ).reset_index().sort_values("Revenue", ascending=False)
    return out



def make_churn_labels(
    sales: pd.DataFrame,
    obs_end: pd.Timestamp,
    future_window_days: int = 90,
    churn_threshold_days: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Build customer features at obs_end and label future inactivity.

    Features use purchases up to obs_end.
    Labels use purchases in the following future window.
    """
    obs_end = pd.Timestamp(obs_end).normalize()
    future_end = obs_end + pd.Timedelta(days=future_window_days)

    obs = sales[sales["InvoiceDate"] <= obs_end].copy()
    fut = sales[
        (sales["InvoiceDate"] > obs_end)
        & (sales["InvoiceDate"] <= future_end)
    ].copy()

    base = obs.groupby("CustomerID").agg(
        RecencyDays=("InvoiceDate", lambda x: (obs_end - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("RevenueGross", "sum"),
        TenureDays=("InvoiceDate", lambda x: (obs_end - x.min()).days),
        AvgBasket=("RevenueGross", "mean"),
        UniqueProducts=("StockCode", "nunique"),
        ActiveWeeks=(
            "SalesWeek",
            "nunique",
        ) if "SalesWeek" in obs.columns else (
            "InvoiceDate",
            lambda x: x.dt.to_period("W").nunique(),
        ),
    ).reset_index()

    # Latest purchase in the future window for each customer.
    fut_last = (
        fut.groupby("CustomerID")["InvoiceDate"]
        .max()
        .rename("FutureLastPurchase")
        .reset_index()
    )
    df = base.merge(fut_last, on="CustomerID", how="left")

    # A customer is churned if their latest future purchase is absent,
    # or they remain inactive for at least the threshold by future_end.
    df["DaysSinceFuturePurchase"] = (
        future_end - df["FutureLastPurchase"]
    ).dt.days

    df["Churned"] = (
        df["FutureLastPurchase"].isna()
        | (df["DaysSinceFuturePurchase"] >= churn_threshold_days)
    ).astype(int)

    info = {
        "obs_end": str(obs_end.date()),
        "future_end": str(future_end.date()),
        "future_window_days": future_window_days,
        "churn_threshold_days": churn_threshold_days,
        "n_observed_customers": int(len(df)),
        "n_churned": int(df["Churned"].sum()),
        "churn_rate": float(df["Churned"].mean()),
    }

    return df, info

