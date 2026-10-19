"""Lightweight drift monitor: Population Stability Index (PSI) on RFM features.
Evidently AI provides a richer dashboard but ships heavy deps. For CPU-only
laptop builds we expose a transparent, single-purpose drift score."""
from __future__ import annotations
import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10,
        eps: float = 1e-6) -> float:
    """Quantile-binned PSI."""
    qs = np.quantile(reference, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    ref_counts, _ = np.histogram(reference, bins=qs)
    cur_counts, _ = np.histogram(current, bins=qs)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference_rfm: pd.DataFrame, current_rfm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("RecencyDays", "Frequency", "Monetary"):
        ref = np.log1p(reference_rfm[col].clip(lower=0))
        cur = np.log1p(current_rfm[col].clip(lower=0))
        score = psi(ref.values, cur.values, bins=10)
        flag = "stable" if score < 0.1 else ("moderate drift" if score < 0.25 else "high drift")
        rows.append({"feature": col, "psi": round(score, 4), "status": flag})
    return pd.DataFrame(rows)
