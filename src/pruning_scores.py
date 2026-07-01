from __future__ import annotations

import torch


def magnitude_score(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs()


def gradient_norm_score(weight: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs() * h.float().clamp_min(0).sqrt()


def gradient_l1_score(weight: torch.Tensor, abs_g: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs() * abs_g.float()


def signed_first_order_score(weight: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return -g.float() * weight.detach().float()


def signed_taylor_score(weight: torch.Tensor, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float()
    return -g.float() * w + 0.5 * h.float() * w.pow(2)


def wanda_score(weight: torch.Tensor, activation_norm: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().abs() * activation_norm.float().view(1, -1)


def hybrid_wanda_signed_taylor_score(weight: torch.Tensor, activation_norm: torch.Tensor, g: torch.Tensor, h: torch.Tensor, lambda_value: float) -> torch.Tensor:
    return wanda_score(weight, activation_norm) + float(lambda_value) * signed_taylor_score(weight, g, h)


def wanda_gradnorm_rank_fusion_score(weight: torch.Tensor, activation_norm: torch.Tensor, h: torch.Tensor, lambda_value: float) -> torch.Tensor:
    wanda = wanda_score(weight, activation_norm)
    gradnorm = gradient_norm_score(weight, h)
    return _rowwise_ascending_ranks(wanda) + float(lambda_value) * _rowwise_ascending_ranks(gradnorm)


def wanda_gradnorm_add_l2_score(weight: torch.Tensor, activation_norm: torch.Tensor, h: torch.Tensor, alpha: float) -> torch.Tensor:
    return wanda_score(weight, activation_norm) + float(alpha) * gradient_norm_score(weight, h)


def wanda_gradnorm_add_l1_score(weight: torch.Tensor, activation_norm: torch.Tensor, abs_g: torch.Tensor, alpha: float) -> torch.Tensor:
    return wanda_score(weight, activation_norm) + float(alpha) * gradient_l1_score(weight, abs_g)


def wanda_gradnorm_z_fusion_score(weight: torch.Tensor, activation_norm: torch.Tensor, h: torch.Tensor, alpha: float) -> torch.Tensor:
    wanda = wanda_score(weight, activation_norm)
    gradnorm = gradient_norm_score(weight, h)
    return _rowwise_zscore(wanda) + float(alpha) * _rowwise_zscore(gradnorm)


def _rowwise_ascending_ranks(score: torch.Tensor) -> torch.Tensor:
    score = score.detach().float()
    if score.dim() != 2:
        raise ValueError(f"rank fusion expects a 2D score tensor, got shape={tuple(score.shape)}")
    order = torch.argsort(score, dim=1, stable=True)
    ranks = torch.empty_like(score, dtype=torch.float32)
    rank_values = torch.arange(score.shape[1], dtype=torch.float32).view(1, -1).expand_as(score)
    ranks.scatter_(1, order, rank_values)
    return ranks


def _rowwise_zscore(score: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    score = score.detach().float()
    if score.dim() != 2:
        raise ValueError(f"z fusion expects a 2D score tensor, got shape={tuple(score.shape)}")
    mean = score.mean(dim=1, keepdim=True)
    std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (score - mean) / std


def compute_score(method: str, weight: torch.Tensor, *, g=None, h=None, abs_g=None, activation_norm=None, lambda_value: float | None = None) -> torch.Tensor | None:
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
    if method == "wanda_gradnorm_rank_fusion":
        return wanda_gradnorm_rank_fusion_score(weight, _require(activation_norm, "activation_norm"), _require(h, "h"), 1.0 if lambda_value is None else lambda_value)
    if method == "wanda_gradnorm_add_l2":
        return wanda_gradnorm_add_l2_score(weight, _require(activation_norm, "activation_norm"), _require(h, "h"), 1.0 if lambda_value is None else lambda_value)
    if method == "wanda_gradnorm_add_l1":
        return wanda_gradnorm_add_l1_score(weight, _require(activation_norm, "activation_norm"), _require(abs_g, "abs_g"), 1.0 if lambda_value is None else lambda_value)
    if method == "wanda_gradnorm_z_fusion":
        return wanda_gradnorm_z_fusion_score(weight, _require(activation_norm, "activation_norm"), _require(h, "h"), 1.0 if lambda_value is None else lambda_value)
    raise ValueError(f"Unknown pruning method: {method}")


def _require(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required for this score")
    return value
