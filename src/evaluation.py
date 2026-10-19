"""Tiny evaluation helpers – keep pipeline honest, hide nothing."""
from __future__ import annotations
import json
from pathlib import Path


def dump_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()},
                  f, indent=2, default=str)
