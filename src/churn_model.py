"""Time-aware churn classifier. Baseline: LogisticRegression. Add: RandomForest."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle, logging
from pathlib import Path

log = logging.getLogger(__name__)
ARTIFACT_DIR = Path("models")


def temporal_split(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp):
    train = df[(df["ObsEnd"] <= train_end)]
    val = df[(df["ObsEnd"] > train_end) & (df["ObsEnd"] <= val_end)]
    return train, val


def hash_split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.30):
    """Deterministic hash-based split (random fallback) when no time info is present."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - val_frac))
    train = df.iloc[idx[:cut]]
    val = df.iloc[idx[cut:]]
    return train, val


def train_churn(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> dict:
    feats = ["RecencyDays", "Frequency", "Monetary", "TenureDays",
             "AvgBasket", "UniqueProducts", "ActiveWeeks"]
    df = df.dropna(subset=feats + ["Churned"]).copy().reset_index(drop=True)
    
    train, val = temporal_split(df, train_end, val_end)

    print("Temporal training rows:", len(train))
    print("Temporal validation rows:", len(val))

    if len(train) < 50 or len(val) < 50:
        raise ValueError(
            "Insufficient temporal training/validation data. "
            "Check the snapshot dates before training."
        )

    if train["ObsEnd"].max() >= val["ObsEnd"].min():
        raise ValueError(
            "Temporal leakage detected: training dates must be "
            "strictly earlier than validation dates."
        )

    if train["Churned"].nunique() < 2 or val["Churned"].nunique() < 2:
        raise ValueError(
            "Both training and validation data must contain "
            "churned and non-churned customers."
        )
        
    print("Training date range:",
          train["ObsEnd"].min(), "to", train["ObsEnd"].max())
    print("Validation date range:",
          val["ObsEnd"].min(), "to", val["ObsEnd"].max())
    print("Training unique customers:",
          train["CustomerID"].nunique())
    print("Validation unique customers:",
          val["CustomerID"].nunique())
    print("Validation churn distribution:")
    print(val["Churned"].value_counts().sort_index().to_dict())


    X_tr, y_tr = train[feats].values, train["Churned"].values.astype(int)
    X_te, y_te = val[feats].values, val["Churned"].values.astype(int)
    out = {}
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=400, class_weight="balanced",
                                                random_state=42))])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "auc_roc": float(roc_auc_score(y_te, proba)),
        "f1": float(f1_score(y_te, pipe.predict(X_te))),
        "precision": float(precision_score(y_te, pipe.predict(X_te))),
        "recall": float(recall_score(y_te, pipe.predict(X_te))),
    }
    out["logistic"] = {"metrics": metrics, "model": pipe}
    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    proba_rf = rf.predict_proba(X_te)[:, 1]
    pred_rf = rf.predict(X_te)
    metrics_rf = {
        "auc_roc": float(roc_auc_score(y_te, proba_rf)),
        "f1": float(f1_score(y_te, pred_rf)),
        "precision": float(precision_score(y_te, pred_rf)),
        "recall": float(recall_score(y_te, pred_rf)),
    }
    feat_imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_})\
        .sort_values("importance", ascending=False).reset_index(drop=True)
    out["random_forest"] = {"metrics": metrics_rf, "model": rf, "feat_imp": feat_imp}
    rank = pd.DataFrame({"p": proba_rf, "y": y_te}).sort_values("p", ascending=False).reset_index(drop=True)
    top_n = max(int(len(rank) * 0.20), 50)
    out["random_forest"]["metrics"]["precision_at_top_20pct"] = float(rank.head(top_n)["y"].mean())
    with open(ARTIFACT_DIR / "churn_baseline.pkl", "wb") as f:
        pickle.dump(pipe, f)
    with open(ARTIFACT_DIR / "churn_rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    return out


def score_all_customers(df: pd.DataFrame, model) -> pd.DataFrame:
    feats = ["RecencyDays", "Frequency", "Monetary", "TenureDays",
             "AvgBasket", "UniqueProducts", "ActiveWeeks"]
    X = df[feats].fillna(0).values
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["ChurnRiskScore"] = p
    out["RiskBand"] = pd.cut(p, bins=[-0.001, 0.30, 0.65, 1.001],
                              labels=["Low Risk", "Medium Risk", "High Risk"])
    return out.sort_values("ChurnRiskScore", ascending=False)
