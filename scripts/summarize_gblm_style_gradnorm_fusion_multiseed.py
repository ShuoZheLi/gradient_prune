from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DEFAULT_BASELINE_METHOD = "wanda"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GBLM-style gradnorm fusion validation across seeds.")
    parser.add_argument("--result-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", default=DEFAULT_BASELINE_METHOD)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    paired_rows = []
    for root in args.result_roots:
        seed = _seed_from_root(root)
        eval_rows = _read_csv(root / "tables" / "main_results.csv")
        ft_rows = _read_csv(root / "first_token_diagnostics" / "summary.csv")
        ft_map = {(row["method"], row["sparsity"], row["lambda_value"]): row for row in ft_rows}
        for row in eval_rows:
            method = row["method"]
            sparsity = row["sparsity"]
            lambda_value = row["lambda_value"] or "none"
            ft = ft_map.get((method, sparsity, lambda_value), {})
            per_seed_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "sparsity": sparsity,
                    "lambda_value": lambda_value,
                    "calibration_response_ce": _float_or_nan(row.get("calibration_ce")),
                    "heldout_response_ce": _float_or_nan(row.get("heldout_ce")),
                    "wikitext_ppl": _float_or_nan(row.get("wikitext_ppl")),
                    "task_accuracy": _float_or_nan(row.get("task_accuracy")),
                    "empty_generation_fraction": _float_or_nan(ft.get("fraction_empty_generations")),
                    "first_token_eos_probability": _float_or_nan(ft.get("mean_first_token_eos_probability")),
                    "average_generated_length": _float_or_nan(ft.get("avg_generated_length_tokens")),
                    "first_token_kl_vs_dense": _float_or_nan(ft.get("mean_first_token_kl_vs_dense")),
                }
            )
        paired_rows.extend(_paired_rows(root, seed, args.baseline_method))

    aggregate_rows = _aggregate(per_seed_rows)
    paired_aggregate_rows = _aggregate_paired(paired_rows)
    _write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    _write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(args.output_dir / "paired_correctness_by_seed.csv", paired_rows)
    _write_csv(args.output_dir / "paired_correctness_aggregate.csv", paired_aggregate_rows)
    (args.output_dir / "README.md").write_text(_readme(), encoding="utf-8")
    print(f"Wrote summary to {args.output_dir}")


def _paired_rows(root: Path, seed: int, baseline_method: str) -> list[dict[str, Any]]:
    rows = []
    eval_rows = _read_csv(root / "tables" / "main_results.csv")
    conditions = [(row["method"], row["sparsity"], row["lambda_value"] or "none") for row in eval_rows]
    baseline_by_sparsity = {sparsity: (root / "accuracy" / f"method={baseline_method}" / f"sparsity={sparsity}" / "lambda=none" / "responses.jsonl") for method, sparsity, lambda_value in conditions if method == baseline_method}
    for method, sparsity, lambda_value in conditions:
        if method == baseline_method:
            continue
        baseline_path = baseline_by_sparsity.get(sparsity)
        candidate_path = root / "accuracy" / f"method={method}" / f"sparsity={sparsity}" / f"lambda={lambda_value}" / "responses.jsonl"
        if baseline_path is None or not baseline_path.is_file() or not candidate_path.is_file():
            continue
        baseline = _read_jsonl_by_id(baseline_path)
        candidate = _read_jsonl_by_id(candidate_path)
        common = sorted(set(baseline) & set(candidate))
        both_correct = baseline_correct_candidate_wrong = baseline_wrong_candidate_correct = both_wrong = 0
        for example_id in common:
            b = bool(baseline[example_id].get("is_correct"))
            c = bool(candidate[example_id].get("is_correct"))
            if b and c:
                both_correct += 1
            elif b and not c:
                baseline_correct_candidate_wrong += 1
            elif (not b) and c:
                baseline_wrong_candidate_correct += 1
            else:
                both_wrong += 1
        rows.append(
            {
                "seed": seed,
                "baseline_method": baseline_method,
                "candidate_method": method,
                "sparsity": sparsity,
                "lambda_value": lambda_value,
                "num_examples": len(common),
                "baseline_correct_candidate_correct": both_correct,
                "baseline_correct_candidate_wrong": baseline_correct_candidate_wrong,
                "baseline_wrong_candidate_correct": baseline_wrong_candidate_correct,
                "baseline_wrong_candidate_wrong": both_wrong,
                "candidate_wins": baseline_wrong_candidate_correct,
                "candidate_losses": baseline_correct_candidate_wrong,
                "net_wins": baseline_wrong_candidate_correct - baseline_correct_candidate_wrong,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["sparsity"], row["lambda_value"])].append(row)
    metrics = ["calibration_response_ce", "heldout_response_ce", "wikitext_ppl", "task_accuracy", "empty_generation_fraction", "first_token_eos_probability", "average_generated_length", "first_token_kl_vs_dense"]
    out = []
    for (method, sparsity, lambda_value), group in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]), _lambda_sort_key(x[0][2]))):
        row = {"method": method, "sparsity": sparsity, "lambda_value": lambda_value, "num_seeds": len(group)}
        for metric in metrics:
            vals = [float(item[metric]) for item in group if not math.isnan(float(item[metric]))]
            row[f"{metric}_mean"] = mean(vals) if vals else math.nan
            row[f"{metric}_std"] = stdev(vals) if len(vals) >= 2 else 0.0 if vals else math.nan
        out.append(row)
    return out


def _aggregate_paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate_method"], row["sparsity"], row["lambda_value"])].append(row)
    out = []
    count_cols = ["baseline_correct_candidate_correct", "baseline_correct_candidate_wrong", "baseline_wrong_candidate_correct", "baseline_wrong_candidate_wrong", "candidate_wins", "candidate_losses", "net_wins"]
    for (candidate_method, sparsity, lambda_value), group in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]), _lambda_sort_key(x[0][2]))):
        row = {"candidate_method": candidate_method, "sparsity": sparsity, "lambda_value": lambda_value, "num_seeds": len(group), "num_examples_total": sum(int(item["num_examples"]) for item in group)}
        for col in count_cols:
            vals = [int(item[col]) for item in group]
            row[f"{col}_sum"] = sum(vals)
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = stdev(vals) if len(vals) >= 2 else 0.0
        out.append(row)
    return out


def _sparsities_from_accuracy(root: Path) -> list[str]:
    method_dir = root / "accuracy" / "method=wanda"
    return [path.name.split("=", 1)[1] for path in method_dir.glob("sparsity=*") if path.is_dir()]


def _read_jsonl_by_id(path: Path) -> dict[int, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[int(row["example_id"])] = row
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float_or_nan(value) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _seed_from_root(root: Path) -> int:
    text = root.name
    marker = "seed"
    if marker in text:
        return int(text.rsplit(marker, 1)[1])
    return -1


def _lambda_sort_key(value: str):
    try:
        return float(value)
    except ValueError:
        return -1.0


def _readme() -> str:
    return """# GBLM-Style GradNorm Fusion Multi-Seed Summary

`aggregate_metrics.csv` reports mean/std over seeds for CE, PPL, accuracy, empty generation fraction, first-token EOS probability, average generated length, and first-token KL.

`paired_correctness_aggregate.csv` sums paired baseline-vs-candidate correctness counts over seeds. The key columns are `candidate_wins_sum`, `candidate_losses_sum`, and `net_wins_sum`.
"""


if __name__ == "__main__":
    main()
