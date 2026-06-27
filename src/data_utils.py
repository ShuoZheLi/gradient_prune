from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


def is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, type(True))) else False


def load_table(path: str | Path, split: str | None = None) -> pd.DataFrame:
    path = Path(path).expanduser()
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            preferred = [p for p in parquet_files if "correct" in p.name or "trajectory" in p.name]
            return pd.read_parquet(preferred[0] if preferred else parquet_files[0])
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            return pd.DataFrame(json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip())
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            return pd.DataFrame(json.loads(line) for line in handle if line.strip())
    dataset = load_dataset(str(path), split=split or "train")
    return dataset.to_pandas()


def first_present(row, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and not is_missing(row[key]):
            return row[key]
    return default
