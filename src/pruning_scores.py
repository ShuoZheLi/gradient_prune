from __future__ import annotations

import torch


def magnitude_score(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs()


def gradient_norm_score(weight: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs() * h.float().clamp_min(0).sqrt()


def signed_first_order_score(weight: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return -g.float() * weight.detach().float()


def signed_taylor_score(weight: torch.Tensor, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float()
    return -g.float() * w + 0.5 * h.float() * w.pow(2)


def wanda_score(weight: torch.Tensor, activation_norm: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs() * activation_norm.float().view(1, -1)


def hybrid_wanda_signed_taylor_score(weight: torch.Tensor, activation_norm: torch.Tensor, g: torch.Tensor, h: torch.Tensor, lambda_value: float) -> torch.Tensor:
    return wanda_score(weight, activation_norm) + float(lambda_value) * signed_taylor_score(weight, g, h)


def compute_score(method: str, weight: torch.Tensor, *, g=None, h=None, activation_norm=None, lambda_value: float | None = None) -> torch.Tensor | None:
    if method == "dense":
        return None
    if method == "magnitude":
        return magnitude_score(weight)
    if method == "gradient_norm":
        return gradient_norm_score(weight, _require(h, "h"))
    if method == "signed_first_order":
        return signed_first_order_score(weight, _require(g, "g"))
    if method == "signed_taylor":
        return signed_taylor_score(weight, _require(g, "g"), _require(h, "h"))
    if method == "wanda":
        return wanda_score(weight, _require(activation_norm, "activation_norm"))
    if method == "hybrid_wanda_signed_taylor":
        return hybrid_wanda_signed_taylor_score(weight, _require(activation_norm, "activation_norm"), _require(g, "g"), _require(h, "h"), 1.0 if lambda_value is None else lambda_value)
    raise ValueError(f"Unknown pruning method: {method}")


def _require(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required for this score")
    return value
