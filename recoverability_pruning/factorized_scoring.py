from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from .diagnostics import tensor_summary
from .factorized_factors import FactorPair


def factorized_score_tensors(
    weight: torch.Tensor,
    ref: FactorPair,
    kd: FactorPair,
    *,
    eta: float,
) -> dict[str, torch.Tensor]:
    if ref.g_structure != kd.g_structure:
        raise ValueError(
            f"Reference/KD g_structure mismatch: {ref.g_structure!r} != {kd.g_structure!r}"
        )
    activation_ref = ref.activation.float()
    activation_kd = kd.activation.float()
    gradient_ref = ref.output_gradient.float()
    gradient_kd = kd.output_gradient.float()

    diag_activation_ref = torch.diagonal(activation_ref)
    diag_activation_kd = torch.diagonal(activation_kd)
    cross_activation = (activation_ref * activation_kd.transpose(0, 1)).sum(dim=1)
    input_recoverability = cross_activation - diag_activation_ref * diag_activation_kd

    if ref.g_structure == "full":
        if gradient_ref.ndim != 2 or gradient_kd.ndim != 2:
            raise ValueError("Full-G factors must be matrices")
        diag_gradient_ref = torch.diagonal(gradient_ref)
        diag_gradient_kd = torch.diagonal(gradient_kd)
        cross_gradient = (gradient_ref * gradient_kd.transpose(0, 1)).sum(dim=1)
        h_ref = diag_gradient_ref[:, None] * diag_activation_ref[None, :]
        rho = (
            cross_gradient[:, None] * cross_activation[None, :]
            - (diag_gradient_ref * diag_gradient_kd)[:, None]
            * (diag_activation_ref * diag_activation_kd)[None, :]
        )
    elif ref.g_structure == "diagonal":
        if gradient_ref.ndim != 1 or gradient_kd.ndim != 1:
            raise ValueError("Diagonal-G factors must be vectors")
        diag_gradient_ref = gradient_ref
        diag_gradient_kd = gradient_kd
        h_ref = diag_gradient_ref[:, None] * diag_activation_ref[None, :]
        rho = (
            (diag_gradient_ref * diag_gradient_kd)[:, None]
            * input_recoverability[None, :]
        )
        cross_gradient = None
    else:
        raise ValueError(f"Unsupported g_structure={ref.g_structure!r}")

    weight_square = weight.detach().float().cpu().square()
    h_ref = h_ref.cpu()
    rho = rho.cpu()
    if weight_square.shape != h_ref.shape or weight_square.shape != rho.shape:
        raise ValueError(
            f"Score shape mismatch: weight={tuple(weight_square.shape)} "
            f"h_ref={tuple(h_ref.shape)} rho={tuple(rho.shape)}"
        )
    damage = 0.5 * weight_square * h_ref
    recovery = float(eta) * weight_square * rho
    score = damage - recovery
    for name, tensor in {"score": score, "damage": damage, "recovery": recovery}.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values in {name}")
    tensors = {
        "score": score.contiguous(),
        "damage": damage.contiguous(),
        "recovery": recovery.contiguous(),
        "h_ref": h_ref.contiguous(),
        "rho": rho.contiguous(),
        "diag_A_ref": diag_activation_ref.cpu().contiguous(),
        "diag_A_kd": diag_activation_kd.cpu().contiguous(),
        "cross_A": cross_activation.cpu().contiguous(),
        "input_recoverability": input_recoverability.cpu().contiguous(),
    }
    if ref.g_structure == "diagonal":
        tensors["gamma_ref"] = diag_gradient_ref.cpu().contiguous()
        tensors["gamma_kd"] = diag_gradient_kd.cpu().contiguous()
    else:
        tensors["diag_G_ref"] = diag_gradient_ref.cpu().contiguous()
        tensors["diag_G_kd"] = diag_gradient_kd.cpu().contiguous()
        tensors["cross_G"] = cross_gradient.cpu().contiguous()
    return tensors


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float((left.double() * right.double()).sum().item())
    denominator = math.sqrt(float(left.double().square().sum().item())) * math.sqrt(
        float(right.double().square().sum().item())
    )
    return numerator / denominator if denominator else math.nan


def module_diagnostics(
    tensors: dict[str, torch.Tensor],
    ref: FactorPair,
    kd: FactorPair,
) -> dict[str, object]:
    recovery = tensors["recovery"]
    diagnostics = {
        "g_structure": ref.g_structure,
        "diag_A_ref": tensor_summary(tensors["diag_A_ref"]),
        "diag_A_kd": tensor_summary(tensors["diag_A_kd"]),
        "cross_A": tensor_summary(tensors["cross_A"]),
        "input_recoverability": tensor_summary(tensors["input_recoverability"]),
        "damage": tensor_summary(tensors["damage"]),
        "recovery": tensor_summary(recovery),
        "score": tensor_summary(tensors["score"]),
        "fraction_recovery_positive": float(recovery.gt(0).float().mean().item()),
        "fraction_recovery_negative": float(recovery.lt(0).float().mean().item()),
        "activation_ref_frobenius_norm": float(torch.linalg.vector_norm(ref.activation.float()).item()),
        "activation_kd_frobenius_norm": float(torch.linalg.vector_norm(kd.activation.float()).item()),
        "gradient_ref_frobenius_norm": float(torch.linalg.vector_norm(ref.output_gradient.float()).item()),
        "gradient_kd_frobenius_norm": float(torch.linalg.vector_norm(kd.output_gradient.float()).item()),
        "activation_factor_cosine": _cosine(ref.activation, kd.activation),
        "gradient_factor_cosine": _cosine(ref.output_gradient, kd.output_gradient),
        "activation_count_ref": ref.activation_count,
        "activation_count_kd": kd.activation_count,
        "output_gradient_count_ref": ref.output_gradient_count,
        "output_gradient_count_kd": kd.output_gradient_count,
    }
    if ref.g_structure == "full":
        diagnostics["diag_G_ref"] = tensor_summary(tensors["diag_G_ref"])
        diagnostics["diag_G_kd"] = tensor_summary(tensors["diag_G_kd"])
        diagnostics["cross_G"] = tensor_summary(tensors["cross_G"])
    else:
        diagnostics["gamma_ref"] = tensor_summary(tensors["gamma_ref"])
        diagnostics["gamma_kd"] = tensor_summary(tensors["gamma_kd"])
    return diagnostics


def save_score_shard(
    output_dir: Path,
    parameter_name: str,
    tensors: dict[str, torch.Tensor],
    *,
    save_factor_diagnostics: bool,
    shard_format: str = "safetensors",
) -> Path:
    module_name = parameter_name.removesuffix(".weight")
    keys = ["score", "damage", "recovery"]
    if save_factor_diagnostics:
        keys.extend(["h_ref", "rho", "diag_A_ref", "diag_A_kd", "cross_A", "input_recoverability"])
        if "gamma_ref" in tensors:
            keys.extend(["gamma_ref", "gamma_kd"])
        else:
            keys.extend(["diag_G_ref", "diag_G_kd", "cross_G"])
    payload = {key: tensors[key].cpu().contiguous() for key in keys}
    if shard_format == "safetensors":
        shard_path = output_dir / f"{module_name}.safetensors"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(payload, str(shard_path))
    elif shard_format == "pt":
        shard_path = output_dir / f"{module_name.replace('.', '__')}.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, shard_path)
    else:
        raise ValueError(f"Unsupported score shard format: {shard_format!r}")
    return shard_path
