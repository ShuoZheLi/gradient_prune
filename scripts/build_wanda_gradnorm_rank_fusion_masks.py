from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apply_pruning import save_masks  # noqa: E402
from config import load_config  # noqa: E402
from layer_utils import iter_prunable_modules  # noqa: E402
from model_utils import load_model_and_tokenizer  # noqa: E402
from pruning_scores import gradient_norm_score, wanda_score  # noqa: E402

METHOD = "wanda_gradnorm_rank_fusion"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WANDA + gradient-norm rowwise rank-fusion masks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gradient-stats-dir", type=Path, required=True)
    parser.add_argument("--wanda-activation-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to config output.root_dir/masks.")
    parser.add_argument("--sparsities", nargs="*", type=float, default=None)
    parser.add_argument("--lambdas", nargs="*", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if config.pruning.granularity != "rowwise":
        raise ValueError(f"Rank fusion is implemented for rowwise pruning, got {config.pruning.granularity!r}")

    sparsities = args.sparsities if args.sparsities is not None else [float(s) for s in config.pruning.sparsities if float(s) > 0]
    lambdas = args.lambdas if args.lambdas is not None else [float(x) for x in config.hybrid.lambda_values]
    masks_root = args.output_root if args.output_root is not None else Path(config.output.root_dir) / "masks"

    activation_stats = _load_tensor_index(args.wanda_activation_dir)
    gradient_stats = _load_tensor_index(args.gradient_stats_dir)

    print(f"Loading model from {config.model.model_name_or_path} on {config.model.device}...", flush=True)
    model, _tokenizer = load_model_and_tokenizer(
        config.model.model_name_or_path,
        config.model.dtype,
        config.model.device,
        config.model.trust_remote_code,
    )
    modules = list(iter_prunable_modules(model, config.pruning.prune_ops))
    module_names = [name for name, _module in modules]
    _validate_module_sets(module_names, activation_stats, "WANDA activation stats")
    _validate_module_sets(module_names, gradient_stats, "gradient stats")

    layer_rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        for sparsity in sparsities:
            print(f"Building {METHOD} masks for lambda={lambda_value} sparsity={sparsity}...", flush=True)
            masks = {}
            for name, module in modules:
                weight = module.weight.detach().cpu().float()
                grad_entry = gradient_stats[name]
                if not isinstance(grad_entry, dict) or "h" not in grad_entry:
                    raise KeyError(f"Gradient stats for {name} must contain key 'h'")
                wanda = wanda_score(weight, activation_stats[name]).float()
                gradnorm = gradient_norm_score(weight, grad_entry["h"]).float()
                fused = rowwise_rank_fusion_score(wanda, gradnorm, float(lambda_value))
                mask = make_rowwise_low_score_mask(fused, float(sparsity))
                masks[name] = mask.cpu()
                layer_rows.append(_layer_metrics(name, float(lambda_value), float(sparsity), weight, wanda, gradnorm, fused, mask))
            out_dir = masks_root / f"method={METHOD}" / f"sparsity={sparsity}" / f"lambda={lambda_value}"
            save_masks(
                masks,
                out_dir,
                {
                    "method": METHOD,
                    "sparsity": sparsity,
                    "lambda_value": lambda_value,
                    "rule": "rowwise rank(wanda_score) + lambda * rank(gradient_norm_score), prune lowest fused rank",
                    "wanda_score": "abs(weight) * sqrt(mean x_j^2)",
                    "gradient_norm_score": "abs(weight) * sqrt(E[g_i^2])",
                    "gradient_stats_dir": str(args.gradient_stats_dir),
                    "wanda_activation_dir": str(args.wanda_activation_dir),
                },
            )

    diag_dir = Path(config.output.root_dir) / "rank_fusion_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(diag_dir / "mask_build_per_layer.csv", layer_rows)
    print(f"Wrote masks under {masks_root / f'method={METHOD}'}", flush=True)
    print(f"Wrote mask-build diagnostics under {diag_dir}", flush=True)


def rowwise_rank_fusion_score(wanda: torch.Tensor, gradnorm: torch.Tensor, lambda_value: float) -> torch.Tensor:
    if wanda.shape != gradnorm.shape:
        raise ValueError(f"Shape mismatch: wanda={tuple(wanda.shape)} gradnorm={tuple(gradnorm.shape)}")
    if wanda.dim() != 2:
        raise ValueError(f"Expected 2D score tensors, got {tuple(wanda.shape)}")
    wanda_rank = _ascending_row_ranks(wanda.float())
    grad_rank = _ascending_row_ranks(gradnorm.float())
    return wanda_rank + float(lambda_value) * grad_rank


def _ascending_row_ranks(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score, dim=1, stable=True)
    ranks = torch.empty_like(score, dtype=torch.float32)
    rank_values = torch.arange(score.shape[1], dtype=torch.float32).view(1, -1).expand_as(score)
    ranks.scatter_(1, order, rank_values)
    return ranks


def make_rowwise_low_score_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    rows, cols = score.shape
    prune_count = int(cols * float(sparsity))
    if prune_count <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    if prune_count >= cols:
        return torch.zeros_like(score, dtype=torch.bool)
    indices = torch.topk(score.float(), k=prune_count, dim=1, largest=False, sorted=False).indices
    mask = torch.ones_like(score, dtype=torch.bool)
    mask.scatter_(1, indices, False)
    return mask


def _load_tensor_index(stats_dir: Path) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    metadata_path = stats_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in metadata["modules"].items()}


def _validate_module_sets(expected_names: list[str], actual: dict[str, Any], label: str) -> None:
    expected = set(expected_names)
    observed = set(actual)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Module mismatch for {label}: missing={missing[:5]} extra={extra[:5]}")


def _layer_metrics(name: str, lambda_value: float, sparsity: float, weight: torch.Tensor, wanda: torch.Tensor, gradnorm: torch.Tensor, fused: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    pruned = ~mask.bool()
    return {
        "method": METHOD,
        "lambda_value": lambda_value,
        "sparsity": sparsity,
        "module": name,
        "num_weights": int(mask.numel()),
        "num_pruned": int(pruned.sum().item()),
        "actual_sparsity": _safe_div(int(pruned.sum().item()), int(mask.numel())),
        "mean_abs_w_pruned": _masked_mean(weight.abs(), pruned),
        "mean_wanda_pruned": _masked_mean(wanda, pruned),
        "mean_gradnorm_pruned": _masked_mean(gradnorm, pruned),
        "mean_fused_rank_pruned": _masked_mean(fused, pruned),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return float("nan")
    return float(values[mask].float().mean().item())


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
