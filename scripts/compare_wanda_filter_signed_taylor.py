from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from layer_utils import iter_prunable_modules  # noqa: E402
from model_utils import load_model_and_tokenizer  # noqa: E402
from pruning_scores import wanda_score  # noqa: E402

METHOD = "wanda_filter_signed_taylor"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WANDA-filter signed-Taylor rerank against WANDA baseline.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--wanda-root", type=Path, required=True)
    parser.add_argument("--signed-taylor-score-dir", type=Path, required=True)
    parser.add_argument("--wanda-activation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    config = load_config(args.result_root / "config.yaml")
    config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype

    rows = _read_csv(args.result_root / "tables" / "main_results.csv")
    eval_by_method_sparsity = {(row["method"], float(row["sparsity"])): row for row in rows}
    sparsities = sorted({float(row["sparsity"]) for row in rows if float(row["sparsity"]) > 0.0})

    activation_stats = _load_tensor_index(args.wanda_activation_dir)
    print(f"Loading model from {config.model.model_name_or_path} on {config.model.device}...", flush=True)
    model, _tokenizer = load_model_and_tokenizer(
        config.model.model_name_or_path,
        config.model.dtype,
        config.model.device,
        config.model.trust_remote_code,
    )
    modules = list(iter_prunable_modules(model, config.pruning.prune_ops))
    module_names = [name for name, _ in modules]
    _validate_module_sets(module_names, activation_stats, "WANDA activation stats")

    report_rows = []
    per_layer_rows = []
    for sparsity in sparsities:
        wanda_masks = _load_masks(args.result_root / "masks" / "method=wanda" / f"sparsity={sparsity}" / "lambda=none")
        rerank_masks = _load_masks(args.result_root / "masks" / f"method={METHOD}" / f"sparsity={sparsity}" / "lambda=none")
        _validate_module_sets(module_names, wanda_masks, f"WANDA masks sparsity={sparsity}")
        _validate_module_sets(module_names, rerank_masks, f"rerank masks sparsity={sparsity}")

        per_layer = []
        for name, module in modules:
            weight = module.weight.detach().cpu().float()
            wanda = wanda_score(weight, activation_stats[name]).float()
            wanda_pruned = ~wanda_masks[name].bool()
            rerank_pruned = ~rerank_masks[name].bool()
            layer = _layer_compare(name, sparsity, weight, wanda, wanda_pruned, rerank_pruned)
            per_layer.append(layer)
            per_layer_rows.append(layer)

        diag = _aggregate(sparsity, per_layer)
        wanda_eval = eval_by_method_sparsity.get(("wanda", sparsity), {})
        rerank_eval = eval_by_method_sparsity.get((METHOD, sparsity), {})
        report_rows.append(
            {
                "sparsity": sparsity,
                "wanda_wikitext_ppl": _float_or_nan(wanda_eval.get("wikitext_ppl")),
                "rerank_wikitext_ppl": _float_or_nan(rerank_eval.get("wikitext_ppl")),
                "ppl_delta_rerank_minus_wanda": _float_or_nan(rerank_eval.get("wikitext_ppl")) - _float_or_nan(wanda_eval.get("wikitext_ppl")),
                "wanda_task_accuracy": _float_or_nan(wanda_eval.get("task_accuracy")),
                "rerank_task_accuracy": _float_or_nan(rerank_eval.get("task_accuracy")),
                "accuracy_delta_rerank_minus_wanda": _float_or_nan(rerank_eval.get("task_accuracy")) - _float_or_nan(wanda_eval.get("task_accuracy")),
                **diag,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "comparison_summary.csv", report_rows)
    _write_csv(args.output_dir / "comparison_per_layer.csv", per_layer_rows)
    _write_json(args.output_dir / "comparison_summary.json", report_rows)
    print(f"Wrote comparison report to {args.output_dir / 'comparison_summary.csv'}", flush=True)


def _layer_compare(name: str, sparsity: float, weight: torch.Tensor, wanda: torch.Tensor, wanda_pruned: torch.Tensor, rerank_pruned: torch.Tensor) -> dict[str, Any]:
    overlap = int((wanda_pruned & rerank_pruned).sum().item())
    rerank_count = int(rerank_pruned.sum().item())
    wanda_count = int(wanda_pruned.sum().item())
    union = int((wanda_pruned | rerank_pruned).sum().item())
    rerank_mean_abs = _masked_mean(weight.abs(), rerank_pruned)
    wanda_mean_abs = _masked_mean(weight.abs(), wanda_pruned)
    rerank_mean_wanda = _masked_mean(wanda, rerank_pruned)
    wanda_mean_wanda = _masked_mean(wanda, wanda_pruned)
    return {
        "sparsity": sparsity,
        "module": name,
        "num_weights": int(weight.numel()),
        "rerank_num_pruned": rerank_count,
        "wanda_num_pruned": wanda_count,
        "rerank_mean_abs_w_pruned": rerank_mean_abs,
        "wanda_mean_abs_w_pruned": wanda_mean_abs,
        "mean_abs_w_pruned_over_wanda": _safe_div(rerank_mean_abs, wanda_mean_abs),
        "rerank_mean_wanda_pruned": rerank_mean_wanda,
        "wanda_mean_wanda_pruned": wanda_mean_wanda,
        "mean_wanda_pruned_over_wanda": _safe_div(rerank_mean_wanda, wanda_mean_wanda),
        "overlap_with_wanda_count": overlap,
        "overlap_with_wanda_frac_of_rerank": _safe_div(overlap, rerank_count),
        "overlap_with_wanda_jaccard": _safe_div(overlap, union),
    }


def _aggregate(sparsity: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rerank_total = sum(int(row["rerank_num_pruned"]) for row in rows)
    wanda_total = sum(int(row["wanda_num_pruned"]) for row in rows)
    overlap_total = sum(int(row["overlap_with_wanda_count"]) for row in rows)
    union_total = sum(int(row["rerank_num_pruned"]) + int(row["wanda_num_pruned"]) - int(row["overlap_with_wanda_count"]) for row in rows)
    rerank_mean_abs = _weighted_mean(rows, "rerank_mean_abs_w_pruned", "rerank_num_pruned")
    wanda_mean_abs = _weighted_mean(rows, "wanda_mean_abs_w_pruned", "wanda_num_pruned")
    rerank_mean_wanda = _weighted_mean(rows, "rerank_mean_wanda_pruned", "rerank_num_pruned")
    wanda_mean_wanda = _weighted_mean(rows, "wanda_mean_wanda_pruned", "wanda_num_pruned")
    return {
        "rerank_num_pruned": rerank_total,
        "wanda_num_pruned": wanda_total,
        "rerank_mean_abs_w_pruned": rerank_mean_abs,
        "wanda_mean_abs_w_pruned": wanda_mean_abs,
        "mean_abs_w_pruned_over_wanda": _safe_div(rerank_mean_abs, wanda_mean_abs),
        "rerank_mean_wanda_pruned": rerank_mean_wanda,
        "wanda_mean_wanda_pruned": wanda_mean_wanda,
        "mean_wanda_pruned_over_wanda": _safe_div(rerank_mean_wanda, wanda_mean_wanda),
        "overlap_with_wanda_count": overlap_total,
        "overlap_with_wanda_frac_of_rerank": _safe_div(overlap_total, rerank_total),
        "overlap_with_wanda_jaccard": _safe_div(overlap_total, union_total),
    }


def _load_tensor_index(stats_dir: Path) -> dict[str, torch.Tensor]:
    metadata_path = stats_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in metadata["modules"].items()}


def _load_masks(mask_dir: Path) -> dict[str, torch.Tensor]:
    return _load_tensor_index(mask_dir)


def _validate_module_sets(expected_names: list[str], actual: dict[str, torch.Tensor], label: str) -> None:
    expected = set(expected_names)
    observed = set(actual)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Module mismatch for {label}: missing={missing[:5]} extra={extra[:5]}")


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return math.nan
    return float(values[mask].float().mean().item())


def _weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = float(row[value_key])
        weight = float(row[weight_key])
        if math.isnan(value) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return _safe_div(numerator, denominator)


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator == 0.0 or math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def _float_or_nan(value) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


if __name__ == "__main__":
    main()
