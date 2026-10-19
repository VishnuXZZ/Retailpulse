"""RFM clustering with K-Means + silhouette score and optional DBSCAN sanity."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging

log = logging.getLogger(__name__)


def cluster_kmeans(rfm: pd.DataFrame, k: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    work = rfm.copy()
    feats = work[["RecencyDays", "Frequency", "Monetary"]].copy()
    feats["LogMonetary"] = np.log1p(feats["Monetary"])
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[["RecencyDays", "Frequency", "LogMonetary"]].values)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    work["KMeansCluster"] = labels
    sil = float(silhouette_score(X, labels)) if k > 1 else float("nan")
    info = {"k": int(k), "silhouette": sil, "inertia": float(km.inertia_)}
    return work, info


def sweep_k(rfm: pd.DataFrame, k_range=range(3, 8)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        _, info = cluster_kmeans(rfm, k)
        rows.append({"k": k, **info})
    return pd.DataFrame(rows)


def dbscan_compare(rfm: pd.DataFrame, eps: float = 0.8, min_samples: int = 50) -> tuple[pd.DataFrame, dict]:
    feats = rfm[["RecencyDays", "Frequency", "Monetary"]].copy()
    feats["LogMonetary"] = np.log1p(feats["Monetary"])
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[["RecencyDays", "Frequency", "LogMonetary"]].values)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    out = rfm.copy()
    out["DBSCANLabel"] = db.labels_
    n_clusters = len(set(db.labels_) - {-1})
    info = {"n_clusters": int(n_clusters), "n_noise": int((db.labels_ == -1).sum()),
            "eps": float(eps), "min_samples": int(min_samples)}
    return out, info
