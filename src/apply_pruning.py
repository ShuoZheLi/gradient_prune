from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from layer_utils import iter_prunable_modules
from masks import apply_mask_to_module, make_layerwise_mask, make_rowwise_mask
from pruning_scores import compute_score


def build_masks_for_model(model, *, method: str, sparsity: float, prune_ops=None, gradient_stats=None, activation_stats=None, lambda_value: float | None = None, granularity: str = "rowwise") -> dict[str, torch.Tensor]:
    masks = {}
    for name, module in iter_prunable_modules(model, prune_ops):
        if method == "dense" or sparsity <= 0:
            masks[name] = torch.ones_like(module.weight, dtype=torch.bool, device="cpu")
            continue
        grad_entry = gradient_stats.get(name) if gradient_stats else None
        score = compute_score(
            method,
            module.weight.detach().cpu(),
            g=grad_entry.get("g") if grad_entry else None,
            h=grad_entry.get("h") if grad_entry else None,
            abs_g=grad_entry.get("abs_g") if grad_entry else None,
            activation_norm=activation_stats.get(name) if activation_stats else None,
            lambda_value=lambda_value,
        )
        if granularity == "rowwise":
            mask = make_rowwise_mask(score, sparsity)
        elif granularity == "layerwise":
            mask = make_layerwise_mask(score, sparsity)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        masks[name] = mask.cpu()
    return masks


def apply_masks(model, masks: dict[str, torch.Tensor], prune_ops=None) -> None:
    modules = dict(iter_prunable_modules(model, prune_ops))
    for name, mask in masks.items():
        apply_mask_to_module(modules[name], mask)


def save_masks(masks: dict[str, torch.Tensor], output_dir: str | Path, metadata: dict | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, mask in masks.items():
        file_name = name.replace(".", "__") + ".pt"
        torch.save(mask.cpu(), output_dir / file_name)
        index[name] = file_name
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"modules": index, **(metadata or {})}, handle, indent=2, default=str)


def save_pruned_model(model, tokenizer, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_use_cache = getattr(model.config, "use_cache", None)
    old_torch_dtype = getattr(model.config, "torch_dtype", None)
    old_dtype = getattr(model.config, "dtype", None)
    if old_use_cache is not None:
        model.config.use_cache = True
    if old_torch_dtype is None and old_dtype is not None:
        model.config.torch_dtype = old_dtype
    try:
        model.save_pretrained(output_dir)
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if old_torch_dtype is None and hasattr(model.config, "torch_dtype"):
            model.config.torch_dtype = old_torch_dtype
    _copy_original_config_if_available(model, output_dir)
    tokenizer.save_pretrained(output_dir)


def _copy_original_config_if_available(model, output_dir: Path) -> None:
    source = getattr(model.config, "_name_or_path", None)
    if not source:
        return
    source_config = Path(source) / "config.json"
    if source_config.is_file():
        shutil.copy2(source_config, output_dir / "config.json")


def load_masks(mask_dir: str | Path) -> dict[str, torch.Tensor]:
    mask_dir = Path(mask_dir)
    metadata_path = mask_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Mask metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {name: torch.load(mask_dir / file_name, map_location="cpu") for name, file_name in metadata["modules"].items()}
