"""Inventory recommendation logic.
The provided UCI Online Retail II workbook does NOT include on-hand stock,
supplier lead times, or purchase orders. We therefore:

* Estimate expected daily demand from the *forecasting* model (or a rolling
  mean fallback when no per-product forecast is saved).
* Take user-configurable Stock-on-hand, Lead time, Safety stock service level
  as inputs from the dashboard.
* Recommend replenishment quantities using a documented formula:
        ROP  = E[D_L] + z * sigma_D_L
        Q    = max(0, ROP + expected_demand_during_lead_time - on_hand)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ARTIFACT_DIR = Path("models")


def _z_for_service_level(sl: float) -> float:
    """z-score for a given service level (0..1)."""
    if sl >= 0.999: return 3.09
    if sl >= 0.99: return 2.33
    if sl >= 0.97: return 1.88
    if sl >= 0.95: return 1.65
    if sl >= 0.90: return 1.28
    return 0.84


def recommend(product_demand: pd.DataFrame,
              forecasts: pd.DataFrame,
              on_hand_lookup: dict[str, float] | None = None,
              lead_time_days: int = 7,
              service_level: float = 0.95) -> pd.DataFrame:
    sl = _z_for_service_level(service_level)
    daily_std_lookup = product_demand.set_index("StockCode")["__daily_std__"].to_dict() if "__daily_std__" in product_demand.columns else {}
    out_rows = []
    for code in product_demand["StockCode"].astype(str).tolist():
        f_sub = forecasts[forecasts["StockCode"].astype(str) == code]
        if f_sub.empty:
            mean_d = float(product_demand.set_index("StockCode").loc[code, "AvgPrice"]) if "AvgPrice" in product_demand.columns else 0.0
            daily_qty = mean_d
        else:
            daily_qty = float(f_sub["Forecast"].mean())
        # sigma daily demand
        sigma_day = float(daily_std_lookup.get(code, 0.0)) if daily_std_lookup else 0.0
        # expected demand during lead time
        e_dl = daily_qty * lead_time_days
        # safety stock ≈ z * sigma (assume sigma independent per day)
        sigma_dl = sigma_day * np.sqrt(lead_time_days)
        rop = e_dl + sl * sigma_dl
        on_hand = float((on_hand_lookup or {}).get(code, 0.0))
        rec_qty = max(0.0, rop - on_hand)
        stockout_flag = on_hand < rop
        out_rows.append({
            "StockCode": code,
            "Description": product_demand.set_index("StockCode").loc[code, "Description"]
                if "Description" in product_demand.columns else "",
            "AvgDailyDemand": round(daily_qty, 3),
            "StdDailyDemand": round(sigma_day, 3),
            "LeadTimeDays": int(lead_time_days),
            "ServiceLevel": float(service_level),
            "ExpectedDemandLeadTime": round(e_dl, 2),
            "SafetyStock": round(sl * sigma_dl, 2),
            "ReorderPoint": round(rop, 2),
            "OnHand": round(on_hand, 2),
            "RecommendedReorderQty": int(round(rec_qty)),
            "StockoutRisk": bool(stockout_flag),
        })
    out = pd.DataFrame(out_rows)
    return out


def daily_std(product_demand: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    s = daily.copy()
    s["StockCode"] = s["StockCode"].astype(str)
    grp = s.groupby("StockCode")["Quantity"].agg(["std", "mean"]).reset_index()
    grp.columns = ["StockCode", "__daily_std__", "__daily_mean__"]
    out = product_demand.copy()
    out["StockCode"] = out["StockCode"].astype(str)
    out = out.merge(grp, on="StockCode", how="left")
    return out
