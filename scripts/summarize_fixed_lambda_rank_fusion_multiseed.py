from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

METHOD = "wanda_gradnorm_rank_fusion"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-lambda WANDA+gradnorm rank-fusion validation across seeds.")
    parser.add_argument("--result-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        paired_rows.extend(_paired_rows(root, seed))

    aggregate_rows = _aggregate(per_seed_rows)
    paired_aggregate_rows = _aggregate_paired(paired_rows)
    _write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    _write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(args.output_dir / "paired_correctness_by_seed.csv", paired_rows)
    _write_csv(args.output_dir / "paired_correctness_aggregate.csv", paired_aggregate_rows)
    (args.output_dir / "README.md").write_text(_readme(), encoding="utf-8")
    print(f"Wrote summary to {args.output_dir}")


def _paired_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    sparsities = sorted(_sparsities_from_accuracy(root), key=float)
    for sparsity in sparsities:
        wanda_path = root / "accuracy" / "method=wanda" / f"sparsity={sparsity}" / "lambda=none" / "responses.jsonl"
        fusion_path = root / "accuracy" / f"method={METHOD}" / f"sparsity={sparsity}" / "lambda=0.03" / "responses.jsonl"
        wanda = _read_jsonl_by_id(wanda_path)
        fusion = _read_jsonl_by_id(fusion_path)
        common = sorted(set(wanda) & set(fusion))
        both_correct = wanda_correct_fusion_wrong = wanda_wrong_fusion_correct = both_wrong = 0
        for example_id in common:
            w = bool(wanda[example_id].get("is_correct"))
            f = bool(fusion[example_id].get("is_correct"))
            if w and f:
                both_correct += 1
            elif w and not f:
                wanda_correct_fusion_wrong += 1
            elif (not w) and f:
                wanda_wrong_fusion_correct += 1
            else:
                both_wrong += 1
        rows.append(
            {
                "seed": seed,
                "sparsity": sparsity,
                "num_examples": len(common),
                "wanda_correct_fusion_correct": both_correct,
                "wanda_correct_fusion_wrong": wanda_correct_fusion_wrong,
                "wanda_wrong_fusion_correct": wanda_wrong_fusion_correct,
                "wanda_wrong_fusion_wrong": both_wrong,
                "fusion_wins": wanda_wrong_fusion_correct,
                "fusion_losses": wanda_correct_fusion_wrong,
                "net_wins": wanda_wrong_fusion_correct - wanda_correct_fusion_wrong,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["sparsity"], row["lambda_value"])].append(row)
    metrics = ["calibration_response_ce", "heldout_response_ce", "wikitext_ppl", "task_accuracy", "empty_generation_fraction", "first_token_eos_probability", "average_generated_length", "first_token_kl_vs_dense"]
    out = []
    for (method, sparsity, lambda_value), group in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]), x[0][2])):
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
        grouped[row["sparsity"]].append(row)
    out = []
    count_cols = ["wanda_correct_fusion_correct", "wanda_correct_fusion_wrong", "wanda_wrong_fusion_correct", "wanda_wrong_fusion_wrong", "fusion_wins", "fusion_losses", "net_wins"]
    for sparsity, group in sorted(grouped.items(), key=lambda x: float(x[0])):
        row = {"sparsity": sparsity, "num_seeds": len(group), "num_examples_total": sum(int(item["num_examples"]) for item in group)}
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


def _readme() -> str:
    return """# Fixed-Lambda Rank-Fusion Multi-Seed Summary

`aggregate_metrics.csv` reports mean/std over seeds for CE, PPL, accuracy, empty generation fraction, first-token EOS probability, average generated length, and first-token KL.

`paired_correctness_aggregate.csv` sums paired WANDA-vs-fusion correctness counts over seeds. The key columns are `fusion_wins_sum`, `fusion_losses_sum`, and `net_wins_sum`.
"""


if __name__ == "__main__":
    main()
