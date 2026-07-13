from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apply_pruning import apply_masks  # noqa: E402
from layer_utils import iter_prunable_modules  # noqa: E402
from masks import compute_actual_sparsity, compute_mask_sparsity, make_layerwise_mask, make_rowwise_mask  # noqa: E402


def score_lookup_keys(method: str, lambda_value: float | None = None) -> tuple[str, ...]:
    if lambda_value is None:
        return (method,)
    return (f"{method}__lambda_{lambda_value:g}", f"{method}__lambda_{lambda_value}", method)


def load_score_metadata(score_dir: str | Path) -> dict[str, Any]:
    score_dir = Path(score_dir)
    metadata_path = score_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Score metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def infer_score_key(metadata: dict[str, Any], method: str | None) -> str:
    if method:
        return method
    score_key = metadata.get("score_key")
    if isinstance(score_key, str) and score_key:
        return score_key
    score_keys = metadata.get("score_keys")
    if isinstance(score_keys, list) and len(score_keys) == 1 and isinstance(score_keys[0], str):
        return score_keys[0]
    raise ValueError("Could not infer score key from metadata; pass --prune_score_key explicitly")


def load_score_tensor(score_path: Path, score_keys: tuple[str, ...]) -> torch.Tensor:
    entry = torch.load(score_path, map_location="cpu")
    if isinstance(entry, torch.Tensor):
        return entry
    if not isinstance(entry, dict):
        raise TypeError(f"Unsupported score file payload in {score_path}: {type(entry).__name__}")
    for key in score_keys:
        if key in entry:
            score = entry[key]
            if not isinstance(score, torch.Tensor):
                raise TypeError(f"Score key {key!r} in {score_path} is not a tensor")
            return score
    expected = " or ".join(repr(key) for key in score_keys)
    raise KeyError(f"Score file {score_path} does not contain key {expected}; available keys: {sorted(entry)}")


def build_masks_from_score_dir(
    model,
    *,
    score_dir: str | Path,
    sparsity: float,
    score_key: str | None = None,
    prune_ops=None,
    granularity: str = "rowwise",
    lambda_value: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    score_dir = Path(score_dir)
    metadata = load_score_metadata(score_dir)
    method = infer_score_key(metadata, score_key)
    modules_index = metadata.get("modules") if isinstance(metadata.get("modules"), dict) else None
    masks: dict[str, torch.Tensor] = {}
    score_keys = score_lookup_keys(method, lambda_value)

    for name, module in iter_prunable_modules(model, prune_ops):
        file_name = modules_index.get(name) if modules_index else f"{name.replace('.', '__')}.pt"
        if not file_name:
            raise FileNotFoundError(f"Score metadata does not list module {name!r}")
        score_path = score_dir / file_name
        if not score_path.is_file():
            raise FileNotFoundError(f"Score file not found for {name}: {score_path}")
        score = load_score_tensor(score_path, score_keys)
        if tuple(score.shape) != tuple(module.weight.shape):
            raise ValueError(f"Score shape {tuple(score.shape)} for {name} does not match weight shape {tuple(module.weight.shape)}")
        if granularity == "rowwise":
            mask = make_rowwise_mask(score, sparsity)
        elif granularity == "layerwise":
            mask = make_layerwise_mask(score, sparsity)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        masks[name] = mask.cpu()

    mask_stats = compute_mask_sparsity(masks)
    info = {
        "score_dir": str(score_dir),
        "score_key": method,
        "requested_sparsity": float(sparsity),
        "granularity": granularity,
        "lambda_value": lambda_value,
        **mask_stats,
    }
    return masks, info


def apply_score_pruning(
    model,
    *,
    score_dir: str | Path | None,
    sparsity: float = 0.0,
    score_key: str | None = None,
    prune_ops=None,
    granularity: str = "rowwise",
    lambda_value: float | None = None,
) -> dict[str, Any]:
    if not score_dir or float(sparsity) <= 0.0:
        return {
            "enabled": False,
            "requested_sparsity": float(sparsity),
            "actual_sparsity": 0.0,
            "num_pruned_weights": 0,
            "num_total_prunable_weights": 0,
        }
    masks, info = build_masks_from_score_dir(
        model,
        score_dir=score_dir,
        sparsity=sparsity,
        score_key=score_key,
        prune_ops=prune_ops,
        granularity=granularity,
        lambda_value=lambda_value,
    )
    apply_masks(model, masks, prune_ops=prune_ops)
    actual = compute_actual_sparsity(model, prune_ops=prune_ops)
    return {"enabled": True, **info, **{f"model_{key}": value for key, value in actual.items()}}
