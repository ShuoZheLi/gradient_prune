from __future__ import annotations

import torch
import torch.nn as nn

from layer_utils import iter_prunable_modules


def make_rowwise_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    score = score.detach().float()
    if sparsity <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    rows, cols = score.shape
    prune_count = int(cols * float(sparsity))
    if prune_count <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    if prune_count >= cols:
        return torch.zeros_like(score, dtype=torch.bool)
    indices = torch.topk(score, k=prune_count, dim=1, largest=False, sorted=False).indices
    mask = torch.ones_like(score, dtype=torch.bool)
    mask.scatter_(1, indices, False)
    return mask


def make_layerwise_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    prune_count = int(score.numel() * float(sparsity))
    if prune_count <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    flat = score.detach().float().view(-1)
    indices = torch.topk(flat, k=min(prune_count, flat.numel()), largest=False, sorted=False).indices
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=score.device)
    mask[indices] = False
    return mask.view_as(score)


def apply_mask_to_module(module: nn.Linear, mask: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))


def compute_actual_sparsity(model, prune_ops=None) -> dict[str, float | int]:
    total = 0
    zeros = 0
    for _, module in iter_prunable_modules(model, prune_ops):
        weight = module.weight.detach()
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    return {
        "num_pruned_weights": zeros,
        "num_total_prunable_weights": total,
        "actual_sparsity": float(zeros / total) if total else 0.0,
    }


def compute_mask_sparsity(masks: dict[str, torch.Tensor]) -> dict[str, float | int]:
    total = sum(mask.numel() for mask in masks.values())
    pruned = sum(int((~mask.bool()).sum().item()) for mask in masks.values())
    return {
        "num_pruned_weights": pruned,
        "num_total_prunable_weights": total,
        "actual_sparsity": float(pruned / total) if total else 0.0,
    }
