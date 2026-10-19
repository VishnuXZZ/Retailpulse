"""Retain raw cancellation semantics; produce clean + cancelled subsets."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["Invoice", "StockCode", "Quantity", "InvoiceDate",
            "Price", "Customer ID", "Country"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})
    return df


def basic_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(raw)
    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    df["Invoice"] = df["Invoice"].astype(str)
    df["InvoiceNoCancel"] = ~df["Invoice"].str.startswith("C")
    df["StockCode"] = df["StockCode"].astype(str)
    df["Country"] = df["Country"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Description"] = df["Description"].astype(str).fillna("UNKNOWN")
    df["Revenue"] = df["Quantity"] * df["Price"]
    summary = {
        "raw_rows": n0,
        "after_dedup_rows": n1,
        "duplicates_dropped": n0 - n1,
    }
    return df, summary


def splits_and_policy(df: pd.DataFrame) -> dict:
    sales = df[df["InvoiceNoCancel"] & (df["Quantity"] > 0) & (df["Price"] > 0)]
    sales = sales.dropna(subset=["CustomerID", "InvoiceDate", "Price", "Quantity"])
    cancellations = df[df["Invoice"].str.startswith("C")]
    returns_or_neg = df[~df["InvoiceNoCancel"] & (df["Quantity"] > 0)]
    invalid_qty = df[(df["Quantity"] <= 0) & ~df["InvoiceNoCancel"]]
    no_customer = df[df["CustomerID"].isna() & df["InvoiceNoCancel"] & (df["Quantity"] > 0)]
    summary = {
        "sales_rows": int(len(sales)),
        "cancellations_rows": int(len(cancellations)),
        "negative_qty_rows": int(len(invalid_qty)),
        "missing_customer_rows": int(len(no_customer)),
    }
    return sales, cancellations, summary


def feature_engineer(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["Year"] = s["InvoiceDate"].dt.year
    s["Month"] = s["InvoiceDate"].dt.month
    s["YearMonth"] = s["InvoiceDate"].dt.to_period("M").astype(str)
    s["Week"] = s["InvoiceDate"].dt.isocalendar().week.astype(int)
    s["Date"] = s["InvoiceDate"].dt.normalize()
    s["DayOfWeek"] = s["InvoiceDate"].dt.day_name()
    s["RevenueGross"] = s["Revenue"]  # gross (already positive qty/price)
    return s
