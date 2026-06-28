from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
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

DEFAULT_RESULTS = {
    "magnitude": REPO_ROOT / "results/qwen25_1p5b_magnitude_math7500",
    "signed_first_order": REPO_ROOT / "results/qwen25_1p5b_signed_first_order_math7500",
    "signed_taylor": REPO_ROOT / "results/qwen25_1p5b_signed_taylor_math7500",
    "wanda": REPO_ROOT / "results/qwen25_1p5b_wanda_math7500",
}
TARGET_METHODS = ("signed_first_order", "signed_taylor")
REFERENCE_METHODS = ("wanda", "magnitude")
DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose signed pruning masks against |w|, WANDA scores, score quantiles, and reference masks."
    )
    parser.add_argument("--signed-first-order-root", type=Path, default=DEFAULT_RESULTS["signed_first_order"])
    parser.add_argument("--signed-taylor-root", type=Path, default=DEFAULT_RESULTS["signed_taylor"])
    parser.add_argument("--wanda-root", type=Path, default=DEFAULT_RESULTS["wanda"])
    parser.add_argument("--magnitude-root", type=Path, default=DEFAULT_RESULTS["magnitude"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/pruning_diagnostics_math7500")
    parser.add_argument("--device", default="cpu", help="Device for temporary model load. Use cuda:0 if CPU memory is tight.")
    parser.add_argument("--dtype", default=None, help="Override model dtype. Default: use signed-first-order config dtype.")
    parser.add_argument("--quantiles", nargs="*", type=float, default=list(DEFAULT_QUANTILES))
    args = parser.parse_args()

    roots = {
        "signed_first_order": args.signed_first_order_root,
        "signed_taylor": args.signed_taylor_root,
        "wanda": args.wanda_root,
        "magnitude": args.magnitude_root,
    }
    for method, root in roots.items():
        _require_dir(root, f"{method} result root")

    config = load_config(roots["signed_first_order"] / "config.yaml")
    config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype

    print(f"Loading model from {config.model.model_name_or_path} on {config.model.device}...", flush=True)
    model, _tokenizer = load_model_and_tokenizer(
        config.model.model_name_or_path,
        dtype=config.model.dtype,
        device=config.model.device,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()
    module_weights = {name: module.weight.detach().cpu().float() for name, module in iter_prunable_modules(model, config.pruning.prune_ops)}
    del model

    activation_stats = _load_tensor_index(roots["wanda"] / "stats" / "activations")
    scores_by_method = {method: _load_scores(roots[method] / "scores", method) for method in TARGET_METHODS}
    masks_by_method = {
        method: _load_masks_for_method(roots[method] / "masks" / f"method={method}")
        for method in (*TARGET_METHODS, *REFERENCE_METHODS)
    }

    module_names = sorted(module_weights)
    _validate_module_sets(module_names, activation_stats, "WANDA activation stats")
    for method, scores in scores_by_method.items():
        _validate_module_sets(module_names, scores, f"{method} scores")
    for method, masks_by_sparsity in masks_by_method.items():
        for sparsity, masks in masks_by_sparsity.items():
            _validate_module_sets(module_names, masks, f"{method} masks at sparsity {sparsity}")

    sparsities = _common_nonzero_sparsities(masks_by_method)
    if not sparsities:
        raise ValueError("No common nonzero sparsities found across signed, WANDA, and magnitude masks.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []

    for method in TARGET_METHODS:
        print(f"Computing exact score quantiles for {method}...", flush=True)
        for module_name in module_names:
            quantile_rows.extend(_score_quantile_rows(method, module_name, scores_by_method[method][module_name], args.quantiles))

    wanda_scores_by_module = {}
    print("Computing WANDA scores from weights and activation stats...", flush=True)
    for module_name in module_names:
        wanda_scores_by_module[module_name] = wanda_score(module_weights[module_name], activation_stats[module_name]).float()

    for method in TARGET_METHODS:
        for sparsity in sparsities:
            print(f"Computing diagnostics for method={method} sparsity={sparsity}...", flush=True)
            per_layer_stats = []
            for module_name in module_names:
                weight = module_weights[module_name]
                signed_score = scores_by_method[method][module_name].float()
                wanda = wanda_scores_by_module[module_name]

                signed_pruned = ~masks_by_method[method][sparsity][module_name].bool()
                signed_kept = ~signed_pruned
                wanda_pruned = ~masks_by_method["wanda"][sparsity][module_name].bool()
                magnitude_pruned = ~masks_by_method["magnitude"][sparsity][module_name].bool()

                layer_stat = _compute_stats(
                    method=method,
                    sparsity=sparsity,
                    module_name=module_name,
                    weight=weight,
                    wanda_score_tensor=wanda,
                    signed_score=signed_score,
                    signed_pruned=signed_pruned,
                    signed_kept=signed_kept,
                    wanda_pruned=wanda_pruned,
                    magnitude_pruned=magnitude_pruned,
                )
                per_layer_stats.append(layer_stat)
                layer_rows.append(layer_stat)
            summary_rows.append(_aggregate_layer_stats(method, sparsity, per_layer_stats))

    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "per_layer.csv", layer_rows)
    _write_csv(args.output_dir / "score_quantiles_per_layer.csv", quantile_rows)
    _write_json(args.output_dir / "diagnostics.json", {"summary": summary_rows, "per_layer": layer_rows, "score_quantiles_per_layer": quantile_rows})
    _write_readme(args.output_dir, roots, config.pruning.granularity, sparsities)

    print(f"Wrote diagnostics to {args.output_dir}", flush=True)
    print(f"Rows: summary={len(summary_rows)}, per_layer={len(layer_rows)}, score_quantiles_per_layer={len(quantile_rows)}", flush=True)
    print("Inspect summary.csv for the key ratios comparing signed-pruned weights to WANDA-pruned weights.", flush=True)


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _load_tensor_index(stats_dir: Path) -> dict[str, torch.Tensor]:
    metadata_path = stats_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in metadata["modules"].items()}


def _load_scores(scores_dir: Path, method: str) -> dict[str, torch.Tensor]:
    _require_dir(scores_dir, f"{method} scores dir")
    scores = {}
    for path in sorted(scores_dir.glob("*.pt")):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or method not in obj:
            raise KeyError(f"Score file {path} does not contain key {method!r}")
        scores[_module_name_from_pt(path)] = obj[method]
    if not scores:
        raise FileNotFoundError(f"No .pt score files found in {scores_dir}")
    return scores


def _load_masks_for_method(method_masks_root: Path) -> dict[float, dict[str, torch.Tensor]]:
    _require_dir(method_masks_root, "method masks root")
    by_sparsity = {}
    for sparsity_dir in sorted(method_masks_root.glob("sparsity=*")):
        if not sparsity_dir.is_dir():
            continue
        sparsity = float(sparsity_dir.name.split("=", 1)[1])
        mask_dir = sparsity_dir / "lambda=none"
        if (mask_dir / "metadata.json").is_file():
            by_sparsity[sparsity] = _load_tensor_index(mask_dir)
    if not by_sparsity:
        raise FileNotFoundError(f"No masks found under {method_masks_root}")
    return by_sparsity


def _module_name_from_pt(path: Path) -> str:
    return path.stem.replace("__", ".")


def _validate_module_sets(expected: Iterable[str], actual: dict[str, Any], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            details.append(f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}")
        raise ValueError(f"Module mismatch for {label}: {'; '.join(details)}")


def _common_nonzero_sparsities(masks_by_method: dict[str, dict[float, dict[str, torch.Tensor]]]) -> list[float]:
    common = None
    for masks_by_sparsity in masks_by_method.values():
        sparsities = {float(s) for s in masks_by_sparsity if float(s) > 0.0}
        common = sparsities if common is None else common & sparsities
    return sorted(common or [])


def _compute_stats(
    *,
    method: str,
    sparsity: float,
    module_name: str,
    weight: torch.Tensor,
    wanda_score_tensor: torch.Tensor,
    signed_score: torch.Tensor,
    signed_pruned: torch.Tensor,
    signed_kept: torch.Tensor,
    wanda_pruned: torch.Tensor,
    magnitude_pruned: torch.Tensor,
) -> dict[str, Any]:
    if weight.shape != signed_score.shape or weight.shape != wanda_score_tensor.shape:
        raise ValueError(f"Shape mismatch for {module_name}")
    for label, mask in (("signed", signed_pruned), ("wanda", wanda_pruned), ("magnitude", magnitude_pruned)):
        if mask.shape != weight.shape:
            raise ValueError(f"{label} mask shape mismatch for {module_name}: {tuple(mask.shape)} vs {tuple(weight.shape)}")

    abs_w = weight.abs()
    signed_n = int(signed_pruned.sum().item())
    wanda_n = int(wanda_pruned.sum().item())
    magnitude_n = int(magnitude_pruned.sum().item())
    total_n = int(weight.numel())
    wanda_overlap = int((signed_pruned & wanda_pruned).sum().item())
    magnitude_overlap = int((signed_pruned & magnitude_pruned).sum().item())

    signed_abs_w_pruned = _masked_mean(abs_w, signed_pruned)
    signed_wanda_pruned = _masked_mean(wanda_score_tensor, signed_pruned)
    wanda_abs_w_pruned = _masked_mean(abs_w, wanda_pruned)
    wanda_wanda_pruned = _masked_mean(wanda_score_tensor, wanda_pruned)

    return {
        "method": method,
        "sparsity": sparsity,
        "module": module_name,
        "num_weights": total_n,
        "num_pruned": signed_n,
        "actual_sparsity": _safe_div(signed_n, total_n),
        "mean_abs_w_pruned": signed_abs_w_pruned,
        "mean_abs_w_kept": _masked_mean(abs_w, signed_kept),
        "mean_wanda_pruned": signed_wanda_pruned,
        "mean_wanda_kept": _masked_mean(wanda_score_tensor, signed_kept),
        "fraction_pruned_negative_score": _masked_mean((signed_score < 0).float(), signed_pruned),
        "mean_signed_score_pruned": _masked_mean(signed_score, signed_pruned),
        "mean_signed_score_kept": _masked_mean(signed_score, signed_kept),
        "wanda_num_pruned": wanda_n,
        "magnitude_num_pruned": magnitude_n,
        "wanda_mean_abs_w_pruned": wanda_abs_w_pruned,
        "wanda_mean_wanda_pruned": wanda_wanda_pruned,
        "magnitude_mean_abs_w_pruned": _masked_mean(abs_w, magnitude_pruned),
        "magnitude_mean_wanda_pruned": _masked_mean(wanda_score_tensor, magnitude_pruned),
        "signed_pruned_mean_abs_w_over_wanda_pruned": _safe_div(signed_abs_w_pruned, wanda_abs_w_pruned),
        "signed_pruned_mean_wanda_over_wanda_pruned": _safe_div(signed_wanda_pruned, wanda_wanda_pruned),
        "pruned_overlap_with_wanda_count": wanda_overlap,
        "pruned_overlap_with_magnitude_count": magnitude_overlap,
        "pruned_overlap_with_wanda_frac_of_signed": _safe_div(wanda_overlap, signed_n),
        "pruned_overlap_with_magnitude_frac_of_signed": _safe_div(magnitude_overlap, signed_n),
        "pruned_overlap_with_wanda_jaccard": _jaccard(signed_pruned, wanda_pruned),
        "pruned_overlap_with_magnitude_jaccard": _jaccard(signed_pruned, magnitude_pruned),
    }


def _score_quantile_rows(method: str, module_name: str, score: torch.Tensor, quantiles: Iterable[float]) -> list[dict[str, Any]]:
    quantile_list = [float(q) for q in quantiles]
    for quantile in quantile_list:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {quantile}")
    flat = score.detach().float().reshape(-1)
    q_tensor = torch.tensor(quantile_list, dtype=torch.float32)
    values = torch.quantile(flat, q_tensor).tolist()
    return [
        {"method": method, "module": module_name, "quantile": quantile, "score_value": float(value)}
        for quantile, value in zip(quantile_list, values, strict=True)
    ]


def _aggregate_layer_stats(method: str, sparsity: float, layer_stats: list[dict[str, Any]]) -> dict[str, Any]:
    total_weights = sum(int(row["num_weights"]) for row in layer_stats)
    total_pruned = sum(int(row["num_pruned"]) for row in layer_stats)
    total_wanda_pruned = sum(int(row["wanda_num_pruned"]) for row in layer_stats)
    total_magnitude_pruned = sum(int(row["magnitude_num_pruned"]) for row in layer_stats)
    total_wanda_overlap = sum(int(row["pruned_overlap_with_wanda_count"]) for row in layer_stats)
    total_magnitude_overlap = sum(int(row["pruned_overlap_with_magnitude_count"]) for row in layer_stats)
    wanda_union = sum(int(row["num_pruned"]) + int(row["wanda_num_pruned"]) - int(row["pruned_overlap_with_wanda_count"]) for row in layer_stats)
    magnitude_union = sum(int(row["num_pruned"]) + int(row["magnitude_num_pruned"]) - int(row["pruned_overlap_with_magnitude_count"]) for row in layer_stats)

    signed_mean_abs = _weighted_mean(layer_stats, "mean_abs_w_pruned", "num_pruned")
    signed_mean_wanda = _weighted_mean(layer_stats, "mean_wanda_pruned", "num_pruned")
    wanda_mean_abs = _weighted_mean(layer_stats, "wanda_mean_abs_w_pruned", "wanda_num_pruned")
    wanda_mean_wanda = _weighted_mean(layer_stats, "wanda_mean_wanda_pruned", "wanda_num_pruned")

    return {
        "method": method,
        "sparsity": sparsity,
        "num_modules": len(layer_stats),
        "num_weights": total_weights,
        "num_pruned": total_pruned,
        "actual_sparsity": _safe_div(total_pruned, total_weights),
        "mean_abs_w_pruned": signed_mean_abs,
        "mean_abs_w_kept": _weighted_mean_kept(layer_stats, "mean_abs_w_kept"),
        "mean_wanda_pruned": signed_mean_wanda,
        "mean_wanda_kept": _weighted_mean_kept(layer_stats, "mean_wanda_kept"),
        "fraction_pruned_negative_score": _weighted_mean(layer_stats, "fraction_pruned_negative_score", "num_pruned"),
        "mean_signed_score_pruned": _weighted_mean(layer_stats, "mean_signed_score_pruned", "num_pruned"),
        "mean_signed_score_kept": _weighted_mean_kept(layer_stats, "mean_signed_score_kept"),
        "wanda_num_pruned": total_wanda_pruned,
        "magnitude_num_pruned": total_magnitude_pruned,
        "wanda_mean_abs_w_pruned": wanda_mean_abs,
        "wanda_mean_wanda_pruned": wanda_mean_wanda,
        "magnitude_mean_abs_w_pruned": _weighted_mean(layer_stats, "magnitude_mean_abs_w_pruned", "magnitude_num_pruned"),
        "magnitude_mean_wanda_pruned": _weighted_mean(layer_stats, "magnitude_mean_wanda_pruned", "magnitude_num_pruned"),
        "signed_pruned_mean_abs_w_over_wanda_pruned": _safe_div(signed_mean_abs, wanda_mean_abs),
        "signed_pruned_mean_wanda_over_wanda_pruned": _safe_div(signed_mean_wanda, wanda_mean_wanda),
        "pruned_overlap_with_wanda_count": total_wanda_overlap,
        "pruned_overlap_with_magnitude_count": total_magnitude_overlap,
        "pruned_overlap_with_wanda_frac_of_signed": _safe_div(total_wanda_overlap, total_pruned),
        "pruned_overlap_with_magnitude_frac_of_signed": _safe_div(total_magnitude_overlap, total_pruned),
        "pruned_overlap_with_wanda_jaccard": _safe_div(total_wanda_overlap, wanda_union),
        "pruned_overlap_with_magnitude_jaccard": _safe_div(total_magnitude_overlap, magnitude_union),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return math.nan
    return float(values[mask].float().mean().item())


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator == 0.0 or math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return _safe_div(intersection, union)


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


def _weighted_mean_kept(rows: list[dict[str, Any]], value_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        kept = int(row["num_weights"]) - int(row["num_pruned"])
        value = float(row[value_key])
        if math.isnan(value) or kept <= 0:
            continue
        numerator += value * kept
        denominator += kept
    return _safe_div(numerator, denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _write_readme(output_dir: Path, roots: dict[str, Path], granularity: str, sparsities: list[float]) -> None:
    text = f"""# Pruning Diagnostics

Generated by `scripts/pruning_diagnostics.py`.

Compared target methods: `signed_first_order`, `signed_taylor`.
Reference masks: `wanda`, `magnitude`.
Mask granularity from config: `{granularity}`.
Common nonzero sparsities analyzed: `{sparsities}`.

Definitions:
- `mean_abs_w_pruned`: mean `abs(weight)` over weights pruned by the target signed method.
- `mean_abs_w_kept`: mean `abs(weight)` over weights kept by the target signed method.
- `mean_wanda_pruned`: mean WANDA score `abs(weight) * activation_norm` over weights pruned by the target signed method.
- `mean_wanda_kept`: mean WANDA score over weights kept by the target signed method.
- `fraction_pruned_negative_score`: fraction of target-pruned weights whose signed score is `< 0`.
- `signed_pruned_mean_abs_w_over_wanda_pruned`: target-pruned mean `abs(weight)` divided by WANDA-pruned mean `abs(weight)` at the same sparsity.
- `signed_pruned_mean_wanda_over_wanda_pruned`: target-pruned mean WANDA score divided by WANDA-pruned mean WANDA score at the same sparsity.
- `pruned_overlap_*_frac_of_signed`: `|target_pruned ∩ reference_pruned| / |target_pruned|`.
- `pruned_overlap_*_jaccard`: `|target_pruned ∩ reference_pruned| / |target_pruned ∪ reference_pruned|`.

Input roots:
{json.dumps({key: str(value) for key, value in roots.items()}, indent=2)}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
