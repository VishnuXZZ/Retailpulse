"""Memory-efficient xlsx ingestion using openpyxl read-only streaming."""
from __future__ import annotations
import openpyxl
import pandas as pd
from pathlib import Path
import time

WORKBOOK = Path("data/raw/online_retail_II.xlsx")

EXPECTED_COLS = ["Invoice", "StockCode", "Description", "Quantity",
                 "InvoiceDate", "Price", "Customer ID", "Country"]


def stream_sheet(sheet_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]
    str_cols = {"Invoice", "StockCode", "Description", "Country"}
    chunk = []
    chunks = []
    for row in rows:
        safe = []
        for h, v in zip(header, row):
            if h in str_cols:
                v = "" if v is None else str(v)
            safe.append(v)
        chunk.append(safe)
        if len(chunk) >= 25_000:
            chunks.append(pd.DataFrame(chunk, columns=header))
            chunk = []
    if chunk:
        chunks.append(pd.DataFrame(chunk, columns=header))
    wb.close()
    df = pd.concat(chunks, ignore_index=True)
    df["__sheet__"] = sheet_name
    return df


def load_all() -> pd.DataFrame:
    t0 = time.time()
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    sheets = wb.sheetnames
    wb.close()
    parts = [stream_sheet(s) for s in sheets]
    df = pd.concat(parts, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {sheets} in {time.time()-t0:.1f}s")
    return df


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("Invoice", "StockCode", "Description", "Country"):
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_for_parquet(df).to_parquet(path, index=False)
    print(f"Saved {path} ({path.stat().st_size/1e6:.1f} MB)")


def inspect(df: pd.DataFrame) -> dict:
    info = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_pct": {c: round(float(df[c].isna().mean()*100), 3) for c in df.columns},
        "sheets": sorted(df["__sheet__"].unique().tolist()) if "__sheet__" in df.columns else None,
    }
    if "InvoiceDate" in df.columns:
        d = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        info["date_min"] = str(d.min())
        info["date_max"] = str(d.max())
        info["date_range_days"] = int((d.max() - d.min()).days)
    if "Invoice" in df.columns:
        s = df["Invoice"].astype(str)
        info["n_cancelled_invoices"] = int(s.str.startswith("C").sum())
        info="skip"
    return info


if __name__ == "__main__":
    pass
