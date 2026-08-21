from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import torch


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int | list[int]]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    valid = value[finite]
    summary: dict[str, float | int | list[int]] = {
        "shape": list(value.shape),
        "numel": value.numel(),
        "num_finite": int(finite.sum().item()),
    }
    if valid.numel() == 0:
        summary.update({"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan})
    else:
        summary.update(
            {
                "mean": float(valid.mean().item()),
                "std": float(valid.std(unbiased=False).item()),
                "min": float(valid.min().item()),
                "max": float(valid.max().item()),
            }
        )
    return summary


def mapping_summary(values: Mapping[str, torch.Tensor], *, scale: float = 1.0) -> dict[str, float | int]:
    total = 0
    total_sum = 0.0
    total_square_sum = 0.0
    minimum = math.inf
    maximum = -math.inf
    for tensor in values.values():
        value = tensor.detach().double()
        if scale != 1.0:
            value = value * scale
        total += value.numel()
        total_sum += float(value.sum().item())
        total_square_sum += float(value.square().sum().item())
        minimum = min(minimum, float(value.min().item()))
        maximum = max(maximum, float(value.max().item()))
    mean = total_sum / total
    variance = max(0.0, total_square_sum / total - mean * mean)
    return {
        "numel": total,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
    }


def parameter_layer_name(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "h"} and index + 1 < len(parts):
            return ".".join(parts[: index + 2])
    return parameter_name.rsplit(".", 2)[0]


def _grouped_mapping_summaries(values: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    moments: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "numel": 0,
            "sum": 0.0,
            "square_sum": 0.0,
            "min": math.inf,
            "max": -math.inf,
        }
    )
    for name, tensor in values.items():
        layer = parameter_layer_name(name)
        value = tensor.detach().double()
        entry = moments[layer]
        entry["numel"] += value.numel()
        entry["sum"] += float(value.sum().item())
        entry["square_sum"] += float(value.square().sum().item())
        entry["min"] = min(float(entry["min"]), float(value.min().item()))
        entry["max"] = max(float(entry["max"]), float(value.max().item()))
    summaries = {}
    for layer, entry in moments.items():
        count = int(entry["numel"])
        mean = float(entry["sum"]) / count
        variance = max(0.0, float(entry["square_sum"]) / count - mean * mean)
        summaries[layer] = {
            "numel": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": float(entry["min"]),
            "max": float(entry["max"]),
        }
    return summaries


def final_diagnostics(
    h_ref: Mapping[str, torch.Tensor],
    rho: Mapping[str, torch.Tensor],
    damage: Mapping[str, torch.Tensor],
    recovery: Mapping[str, torch.Tensor],
    scores: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    per_tensor = {}
    per_layer_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in scores:
        rho_value = rho[name]
        score_value = scores[name]
        per_tensor[name] = {
            "h_ref": tensor_summary(h_ref[name]),
            "rho": tensor_summary(rho_value),
            "damage": tensor_summary(damage[name]),
            "recovery": tensor_summary(recovery[name]),
            "final_score": tensor_summary(score_value),
            "fraction_rho_positive": float((rho_value > 0).float().mean().item()),
            "fraction_rho_negative": float((rho_value < 0).float().mean().item()),
            "fraction_score_negative": float((score_value < 0).float().mean().item()),
        }
        layer = parameter_layer_name(name)
        per_layer_counts[layer]["numel"] += score_value.numel()
        per_layer_counts[layer]["rho_positive"] += int((rho_value > 0).sum().item())
        per_layer_counts[layer]["rho_negative"] += int((rho_value < 0).sum().item())
        per_layer_counts[layer]["score_negative"] += int((score_value < 0).sum().item())
    component_summaries = {
        "h_ref": _grouped_mapping_summaries(h_ref),
        "rho": _grouped_mapping_summaries(rho),
        "damage": _grouped_mapping_summaries(damage),
        "recovery": _grouped_mapping_summaries(recovery),
        "final_score": _grouped_mapping_summaries(scores),
    }
    per_layer = {
        layer: {
            "numel": counts["numel"],
            "fraction_rho_positive": counts["rho_positive"] / counts["numel"],
            "fraction_rho_negative": counts["rho_negative"] / counts["numel"],
            "fraction_score_negative": counts["score_negative"] / counts["numel"],
            **{component: summaries[layer] for component, summaries in component_summaries.items()},
        }
        for layer, counts in per_layer_counts.items()
    }
    return {"per_tensor": per_tensor, "per_layer": per_layer}


def gpu_memory_summary() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        f"cuda:{index}_allocated_gib": torch.cuda.memory_allocated(index) / 2**30
        for index in range(torch.cuda.device_count())
    } | {
        f"cuda:{index}_reserved_gib": torch.cuda.memory_reserved(index) / 2**30
        for index in range(torch.cuda.device_count())
    } | {
        f"cuda:{index}_peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30
        for index in range(torch.cuda.device_count())
    }
