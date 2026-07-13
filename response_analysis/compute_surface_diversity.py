from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from response_analysis.io_utils import read_jsonl
from response_analysis.metrics import answer_diversity, group_records, surface_diversity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute surface-form and final-answer diversity per prompt.")
    parser.add_argument("--input", default="outputs/generations.jsonl")
    parser.add_argument("--output", default="outputs/response_metrics.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    rows = []
    for key, group in group_records(records, ["model_id", "prompt_id"]).items():
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
        rows.append({**base, **surface_diversity(texts), **answer_diversity(answers, correctness)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


if __name__ == "__main__":
    main()
