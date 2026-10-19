"""Forecasting module: seasonal-naive baseline + HistGradientBoostingRegressor.
LSTM is intentionally skipped per project master-prompt rule (CPU-only, dataset scale)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging, pickle
from pathlib import Path

log = logging.getLogger(__name__)
ARTIFACT_DIR = Path("models")
FEAT_COLS = ["DOW", "Month", "Day", "WeekOfYear", "Year",
             "Lag1", "Lag7", "Lag14", "Lag28",
             "RollingMean7", "RollingMean28", "RollingStd7"]


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def seasonal_naive(
    series: pd.Series,
    season: int = 7,
    horizon: int = 30
) -> np.ndarray:
    """Repeat the seasonal pattern, starting with the last observation."""

    last = series.dropna().to_numpy(dtype=float)

    if len(last) == 0:
        return np.zeros(horizon)

    if season < 1:
        raise ValueError("season must be at least 1")

    base = last[-season:] if len(last) >= season else last
    base = np.roll(base, 1)

    return np.resize(base, horizon)


def _add_calendar_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy().sort_values("Date").reset_index(drop=True)
    s["Quantity"] = s["Quantity"].astype(float)
    s["DOW"] = s["Date"].dt.dayofweek
    s["Month"] = s["Date"].dt.month
    s["Day"] = s["Date"].dt.day
    s["WeekOfYear"] = s["Date"].dt.isocalendar().week.astype(int)
    s["Year"] = s["Date"].dt.year
    s["Lag1"] = s["Quantity"].shift(1)
    s["Lag7"] = s["Quantity"].shift(7)
    s["Lag14"] = s["Quantity"].shift(14)
    s["Lag28"] = s["Quantity"].shift(28)
    s["RollingMean7"]  = s["Quantity"].shift(1).rolling(7).mean()
    s["RollingMean28"] = s["Quantity"].shift(1).rolling(28).mean()
    s["RollingStd7"]   = s["Quantity"].shift(1).rolling(7).std()
    return s


def _train_test_split(full: pd.DataFrame, horizon: int = 30):
    feat = full.dropna(subset=FEAT_COLS).reset_index(drop=True)
    if len(feat) < horizon + 30:
        return None, None, {"reason": "valid_rows_too_short", "n_valid": int(len(feat))}
    train = feat.iloc[:-horizon].copy()
    test = feat.iloc[-horizon:].copy()
    return train, test, {"n_valid": int(len(feat)), "n_train": int(len(train)), "n_test": int(len(test))}


def train_regressor(full_daily: pd.DataFrame, horizon: int = 30) -> tuple[object | None, dict]:
    full = _add_calendar_and_lags(full_daily[["Date", "Quantity"]].drop_duplicates(subset=["Date"]))
    train, test, split_info = _train_test_split(full, horizon=horizon)
    if train is None:
        return None, split_info
    X_tr = train[FEAT_COLS].values; y_tr = train["Quantity"].values.astype(float)
    X_te = test[FEAT_COLS].values;  y_te = test["Quantity"].values.astype(float)
    regressor = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=42)
    regressor.fit(X_tr, y_tr)
    pred_reg = regressor.predict(X_te)
    pred_base = seasonal_naive(
    train["Quantity"],
    season=7,
    horizon=len(test)
)
    pred_avg = 0.5 * pred_reg + 0.5 * pred_base
    metrics = {
        "n_test_days": int(len(test)),
        "mae_naive": float(mean_absolute_error(y_te, pred_base)),
        "rmse_naive": float(np.sqrt(mean_squared_error(y_te, pred_base))),
        "wape_naive": wape(y_te, pred_base),
        "mae_reg": float(mean_absolute_error(y_te, pred_reg)),
        "rmse_reg": float(np.sqrt(mean_squared_error(y_te, pred_reg))),
        "wape_reg": wape(y_te, pred_reg),
        "mape_reg": mape_safe(y_te, pred_reg),
        "mape_avg": mape_safe(y_te, pred_avg),
        "mae_avg": float(mean_absolute_error(y_te, pred_avg)),
        "rmse_avg": float(np.sqrt(mean_squared_error(y_te, pred_avg))),
        "wape_avg": wape(y_te, pred_avg),
    }
    return regressor, {"model": regressor, "metrics": metrics, "feat_cols": FEAT_COLS,
                       "split": split_info, "last_train_date": str(test["Date"].min().date())}


def _recursive_forecast(model, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast H future days, feeding predictions back as Lag1 to keep feature continuity."""
    cur = history_df.copy()
    future = pd.date_range(cur["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    rows = []
    for d in future:
        probe = cur.copy()
        new_row = pd.DataFrame({"Date": [d], "Quantity": [np.nan]})
        probe = pd.concat([probe, new_row], ignore_index=True)
        feat = _add_calendar_and_lags(probe).iloc[[-1]]
        yhat = max(float(model.predict(feat[FEAT_COLS].values)[0]), 0.0)
        rows.append({"Date": d, "Forecast": yhat, "Lower": yhat * 0.85, "Upper": yhat * 1.15})
        cur = pd.concat([cur, pd.DataFrame({"Date": [d], "Quantity": [yhat]})], ignore_index=True)
    return pd.DataFrame(rows)


def forecast_product(sales: pd.DataFrame, horizon: int = 30, top_k: int = 10,
                     stock_codes: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    metrics = []
    forecasts = []
    if stock_codes is None:
        revenue_rank = salary = sales.groupby(sales["StockCode"].astype(str))["RevenueGross"].sum() \
            .sort_values(ascending=False)
        stock_codes = revenue_rank.head(top_k).index.astype(str).tolist()

    for code in stock_codes:
        sub = sales[sales["StockCode"].astype(str) == code].copy()
        sub["Date"] = sub["InvoiceDate"].dt.normalize()
        daily = sub.groupby("Date", as_index=False)["Quantity"].sum().sort_values("Date")
        if len(daily) < 30:
            metrics.append({"StockCode": code, "reason": "insufficient_history", "n_days": int(len(daily))})
            continue
        model, info = train_regressor(daily, horizon=horizon)
        if model is None:
            metrics.append({"StockCode": code, **info})
            continue
        fc = _recursive_forecast(model, daily.rename(columns={"Date": "Date", "Quantity": "Quantity"}), horizon=horizon)
        fc["StockCode"] = code
        forecasts.append(fc)
        m = {"StockCode": code,
             **{k: info["metrics"][k] for k in info["metrics"]}}
        metrics.append(m)
        with open(ARTIFACT_DIR / f"forecast_{code}.pkl", "wb") as f:
            pickle.dump({"model": model, "feat_cols": FEAT_COLS, "horizon": horizon}, f)

    metrics_df = pd.DataFrame(metrics)
    fcst_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    if not fcst_df.empty:
        fcst_df.to_parquet("data/processed/forecasts.parquet", index=False)
    metrics_df.to_parquet("data/processed/forecast_metrics.parquet", index=False)
    return {"metrics": metrics_df, "forecasts": fcst_df}, metrics_df


def forecast_aggregate(daily_agg: pd.DataFrame, horizon: int = 30) -> dict:
    s = daily_agg[["Date", "Quantity"]].copy().sort_values("Date").drop_duplicates(subset=["Date"])
    if len(s) < 30:
        return {"reason": "short_history", "n_days": int(len(s))}
    model, info = train_regressor(s, horizon=horizon)
    if model is None:
        return {"reason": "model_not_fit", **info}
    fc = _recursive_forecast(model, s, horizon=horizon)
    fc.to_parquet("data/processed/forecast_aggregate.parquet", index=False)
    with open(ARTIFACT_DIR / "forecast_aggregate.pkl", "wb") as f:
        pickle.dump({"model": model, "feat_cols": FEAT_COLS, "horizon": horizon}, f)
    return {"metrics": info["metrics"], "forecasts": fc}


def try_prophet(daily_agg: pd.DataFrame, horizon: int = 30) -> dict | None:
    try:
        from prophet import Prophet  # type: ignore
    except Exception as e:
        return {"reason": "prophet_not_available", "error": str(e)}
    df = daily_agg.rename(columns={"Date": "ds", "Quantity": "y"})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    fut = m.make_future_dataframe(periods=horizon)
    fc = m.predict(fut)
    return {"fitted": True,
            "tail": fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).rename(
                columns={"ds": "Date", "yhat": "Forecast",
                         "yhat_lower": "Lower", "yhat_upper": "Upper"})}
