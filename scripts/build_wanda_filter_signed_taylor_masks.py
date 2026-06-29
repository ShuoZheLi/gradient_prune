from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from layer_utils import iter_prunable_modules  # noqa: E402
from model_utils import load_model_and_tokenizer  # noqa: E402
from pruning_scores import wanda_score  # noqa: E402
from apply_pruning import save_masks  # noqa: E402

METHOD = "wanda_filter_signed_taylor"
DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rowwise WANDA-candidate + signed-Taylor rerank masks.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config whose model/prune_ops/output root will be used.")
    parser.add_argument("--signed-taylor-score-dir", type=Path, required=True)
    parser.add_argument("--wanda-activation-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to config output.root_dir/masks.")
    parser.add_argument("--sparsities", nargs="*", type=float, default=None, help="Defaults to config pruning.sparsities excluding 0.")
    parser.add_argument("--device", default=None, help="Override model device for loading weights.")
    parser.add_argument("--dtype", default=None, help="Override model dtype for loading weights.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if config.pruning.granularity != "rowwise":
        raise ValueError(f"This experiment is defined rowwise, got granularity={config.pruning.granularity!r}")

    sparsities = args.sparsities if args.sparsities is not None else [float(s) for s in config.pruning.sparsities if float(s) > 0]
    for sparsity in sparsities:
        if not 0.0 < float(sparsity) <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")

    masks_root = args.output_root if args.output_root is not None else Path(config.output.root_dir) / "masks"
    activation_stats = _load_tensor_index(args.wanda_activation_dir)
    signed_taylor_scores = _load_scores(args.signed_taylor_score_dir, "signed_taylor")

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
    _validate_module_sets(module_names, signed_taylor_scores, "signed Taylor scores")

    metrics_rows = []
    quantile_rows = []
    for sparsity in sparsities:
        print(f"Building {METHOD} masks for sparsity={sparsity}...", flush=True)
        masks = {}
        for name, module in modules:
            weight = module.weight.detach().cpu().float()
            wanda = wanda_score(weight, activation_stats[name]).float()
            signed_taylor = signed_taylor_scores[name].float()
            mask = make_wanda_filter_signed_taylor_rowwise_mask(wanda, signed_taylor, float(sparsity))
            masks[name] = mask.cpu()
            metrics_rows.append(_layer_metrics(name, float(sparsity), weight, wanda, signed_taylor, mask))
            if float(sparsity) == float(sparsities[0]):
                quantile_rows.extend(_score_quantile_rows(name, signed_taylor, DEFAULT_QUANTILES))

        out_dir = masks_root / f"method={METHOD}" / f"sparsity={sparsity}" / "lambda=none"
        save_masks(
            masks,
            out_dir,
            {
                "method": METHOD,
                "sparsity": sparsity,
                "lambda_value": None,
                "candidate_rule": "bottom min(0.6, 2 * sparsity) weights by WANDA score in each row",
                "rerank_rule": "within candidate, prune bottom sparsity weights by signed_taylor score in each row",
                "signed_taylor_score_dir": str(args.signed_taylor_score_dir),
                "wanda_activation_dir": str(args.wanda_activation_dir),
            },
        )

    diag_dir = Path(config.output.root_dir) / "rerank_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(diag_dir / "mask_build_per_layer.csv", metrics_rows)
    _write_csv(diag_dir / "signed_taylor_score_quantiles_per_layer.csv", quantile_rows)
    print(f"Wrote masks under {masks_root / f'method={METHOD}'}", flush=True)
    print(f"Wrote mask-build diagnostics under {diag_dir}", flush=True)


def make_wanda_filter_signed_taylor_rowwise_mask(wanda_score_tensor: torch.Tensor, signed_taylor_score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if wanda_score_tensor.shape != signed_taylor_score.shape:
        raise ValueError(f"Shape mismatch: wanda={tuple(wanda_score_tensor.shape)} signed_taylor={tuple(signed_taylor_score.shape)}")
    if wanda_score_tensor.dim() != 2:
        raise ValueError(f"Expected 2D linear weight scores, got shape={tuple(wanda_score_tensor.shape)}")
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    rows, cols = wanda_score_tensor.shape
    prune_count = int(cols * float(sparsity))
    if prune_count <= 0:
        return torch.ones_like(wanda_score_tensor, dtype=torch.bool)
    if prune_count >= cols:
        return torch.zeros_like(wanda_score_tensor, dtype=torch.bool)

    candidate_fraction = min(0.6, 2.0 * float(sparsity))
    candidate_count = int(cols * candidate_fraction)
    candidate_count = max(prune_count, min(candidate_count, cols))

    candidate_idx = torch.topk(wanda_score_tensor.float(), k=candidate_count, dim=1, largest=False, sorted=False).indices
    candidate_signed = signed_taylor_score.float().gather(1, candidate_idx)
    rerank_local_idx = torch.topk(candidate_signed, k=prune_count, dim=1, largest=False, sorted=False).indices
    prune_idx = candidate_idx.gather(1, rerank_local_idx)

    mask = torch.ones_like(wanda_score_tensor, dtype=torch.bool)
    mask.scatter_(1, prune_idx, False)
    return mask


def _load_tensor_index(stats_dir: Path) -> dict[str, torch.Tensor]:
    metadata_path = stats_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in metadata["modules"].items()}


def _load_scores(scores_dir: Path, method: str) -> dict[str, torch.Tensor]:
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"Missing score dir: {scores_dir}")
    scores = {}
    for path in sorted(scores_dir.glob("*.pt")):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or method not in obj:
            raise KeyError(f"Score file {path} does not contain key {method!r}")
        scores[path.stem.replace("__", ".")] = obj[method]
    if not scores:
        raise FileNotFoundError(f"No .pt score files found in {scores_dir}")
    return scores


def _validate_module_sets(expected_names: list[str], actual: dict[str, torch.Tensor], label: str) -> None:
    expected = set(expected_names)
    observed = set(actual)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Module mismatch for {label}: missing={missing[:5]} extra={extra[:5]}")


def _layer_metrics(name: str, sparsity: float, weight: torch.Tensor, wanda: torch.Tensor, signed_taylor: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | str]:
    pruned = ~mask.bool()
    kept = mask.bool()
    return {
        "method": METHOD,
        "sparsity": sparsity,
        "module": name,
        "candidate_fraction": min(0.6, 2.0 * sparsity),
        "num_weights": int(mask.numel()),
        "num_pruned": int(pruned.sum().item()),
        "actual_sparsity": _safe_div(int(pruned.sum().item()), int(mask.numel())),
        "mean_abs_w_pruned": _masked_mean(weight.abs(), pruned),
        "mean_abs_w_kept": _masked_mean(weight.abs(), kept),
        "mean_wanda_pruned": _masked_mean(wanda, pruned),
        "mean_wanda_kept": _masked_mean(wanda, kept),
        "fraction_pruned_negative_score": _masked_mean((signed_taylor < 0).float(), pruned),
        "mean_signed_taylor_pruned": _masked_mean(signed_taylor, pruned),
        "mean_signed_taylor_kept": _masked_mean(signed_taylor, kept),
    }


def _score_quantile_rows(module_name: str, score: torch.Tensor, quantiles: tuple[float, ...]) -> list[dict[str, float | str]]:
    flat = score.detach().float().reshape(-1)
    values = torch.quantile(flat, torch.tensor(list(quantiles), dtype=torch.float32)).tolist()
    return [{"method": METHOD, "module": module_name, "quantile": q, "score_value": float(v)} for q, v in zip(quantiles, values, strict=True)]


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
