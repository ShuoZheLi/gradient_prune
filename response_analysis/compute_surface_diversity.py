from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from response_analysis.io_utils import read_jsonl
from response_analysis.metrics import answer_diversity, group_records, surface_diversity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute surface-form and final-answer diversity per prompt.")
    parser.add_argument("--input", default="outputs/generations.jsonl")
    parser.add_argument("--output", default="outputs/response_metrics.parquet")
    parser.add_argument("--workers", type=int, default=1, help="Number of prompt groups to process in parallel.")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable the prompt-group progress bar.")
    return parser.parse_args()


def surface_metrics_for_group(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> dict[str, Any]:
    key, group = item
    model_id, prompt_id = key
    group = sorted(group, key=lambda row: row.get("sample_id", 0))
    texts = [str(row.get("generated_text", "")) for row in group]
    answers = [row.get("parsed_final_answer") for row in group]
    correctness = [row.get("correctness", False) for row in group]
    base = {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "pruning_sparsity": group[0].get("pruning_sparsity"),
        "mean_response_length": sum(float(row.get("response_length", 0)) for row in group) / len(group),
    }
    return {**base, **surface_diversity(texts), **answer_diversity(answers, correctness)}


def compute_surface_metrics(input_path: str | Path, output_path: str | Path, *, workers: int = 1, disable_tqdm: bool = False) -> pd.DataFrame:
    records = read_jsonl(input_path)
    groups = group_records(records, ["model_id", "prompt_id"])
    rows = []
    items = list(groups.items())
    workers = max(1, int(workers))
    if workers == 1 or len(items) <= 1:
        iterator = tqdm(items, total=len(items), desc="surface_diversity", unit="prompt", disable=disable_tqdm)
        for item in iterator:
            rows.append(surface_metrics_for_group(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(surface_metrics_for_group, item) for item in items]
            iterator = tqdm(as_completed(futures), total=len(futures), desc="surface_diversity", unit="prompt", disable=disable_tqdm)
            for future in iterator:
                rows.append(future.result())
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model_id", "prompt_id"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return df


def main() -> None:
    args = parse_args()
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    compute_surface_metrics(args.input, args.output, workers=workers, disable_tqdm=args.disable_tqdm)


if __name__ == "__main__":
    main()
