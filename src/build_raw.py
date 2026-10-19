"""Memory-efficient ingestion: build raw parquet from the xlsx workbook."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_ingestion import load_all, save_parquet

if __name__ == "__main__":
    t0 = time.time()
    df = load_all()
    save_parquet(df, Path("data/processed/raw.parquet"))
    print(f"Total time: {time.time()-t0:.2f}s")
