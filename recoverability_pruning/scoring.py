from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch

from .diagnostics import final_diagnostics, gpu_memory_summary, mapping_summary
from .hvp import compute_dataset_hvp
from .losses import causal_sft_nll
from .online_stats import OnlineMeanCovariance
from .params import ParameterSpace
from .probes import make_rademacher_probe


LOGGER = logging.getLogger(__name__)


def compute_score_components(
    parameter_space: ParameterSpace,
    stats: OnlineMeanCovariance,
    *,
    eta: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    h_ref = stats.mean_x
    rho = stats.sample_covariance()
    damage = {}
    recovery = {}
    scores = {}
    for name, parameter in zip(
        parameter_space.candidate_names,
        parameter_space.candidate_parameters,
        strict=True,
    ):
        weight_square = parameter.detach().to(device="cpu", dtype=torch.float32).square()
        damage[name] = 0.5 * weight_square * h_ref[name]
        recovery[name] = eta * weight_square * rho[name]
        scores[name] = damage[name] - recovery[name]
    return scores, damage, recovery, h_ref, rho


def _compute_scores_only(
    parameter_space: ParameterSpace,
    stats: OnlineMeanCovariance,
    *,
    eta: float,
) -> dict[str, torch.Tensor]:
    if stats.count < 2:
        raise ValueError("At least two probes are required to compute scores")
    scores = {}
    for name, parameter in zip(
        parameter_space.candidate_names,
        parameter_space.candidate_parameters,
        strict=True,
    ):
        weight_square = parameter.detach().to(device="cpu", dtype=torch.float32).square()
        rho = stats.cov_m2[name] / (stats.count - 1)
        scores[name] = weight_square * (0.5 * stats.mean_x[name] - eta * rho)
    return scores


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_snapshot(
    output_path: Path,
    probe_count: int,
    parameter_space: ParameterSpace,
    stats: OnlineMeanCovariance,
    eta: float,
) -> Path:
    scores = _compute_scores_only(parameter_space, stats, eta=eta)
    snapshot_dir = output_path.parent / f"{output_path.stem}_convergence"
    snapshot_path = snapshot_dir / f"scores_probe_{probe_count:04d}.pt"
    _atomic_torch_save({"num_probes": probe_count, "scores": scores}, snapshot_path)
    return snapshot_path


def compute_dense_pruning_scores(
    model,
    ref_loader,
    kd_loader,
    parameter_space: ParameterSpace,
    *,
    num_probes: int,
    probe_seed: int,
    eta: float,
    output_path: str | Path,
    metadata: dict[str, object],
    save_intermediate_stats: bool,
    convergence_checkpoints: tuple[int, ...] = (),
) -> dict[str, object]:
    if num_probes < 2:
        raise ValueError("num_probes must be at least 2 for unbiased sample covariance; use --smoke_test for M=1")
    candidate_templates = {
        name: parameter
        for name, parameter in zip(
            parameter_space.candidate_names,
            parameter_space.candidate_parameters,
            strict=True,
        )
    }
    stats = OnlineMeanCovariance(candidate_templates)
    output_path = Path(output_path)
    probe_diagnostics = []
    snapshot_paths = []

    for probe_index in range(num_probes):
        started = time.perf_counter()
        seed = probe_seed + probe_index
        LOGGER.info("Starting shared Rademacher probe %d/%d seed=%d", probe_index + 1, num_probes, seed)
        probe = make_rademacher_probe(
            parameter_space.hvp_names,
            parameter_space.hvp_parameters,
            seed=seed,
        )
        reference_hvp, reference_info = compute_dataset_hvp(
            model,
            ref_loader,
            probe,
            parameter_space,
            causal_sft_nll,
            objective_name="reference",
        )
        x = {
            name: reference_hvp[name] * probe[name].to(dtype=torch.float32)
            for name in parameter_space.candidate_names
        }
        del reference_hvp
        kd_hvp, kd_info = compute_dataset_hvp(
            model,
            kd_loader,
            probe,
            parameter_space,
            causal_sft_nll,
            objective_name="kd",
        )
        y = {
            name: kd_hvp[name] * probe[name].to(dtype=torch.float32)
            for name in parameter_space.candidate_names
        }
        del kd_hvp, probe
        stats.update(x, y)
        diagnostics = {
            "probe_index": probe_index + 1,
            "seed": seed,
            "reference": reference_info,
            "kd": kd_info,
            "x": mapping_summary(x),
            "y": mapping_summary(y),
            "running_h_ref": mapping_summary(stats.mean_x),
            "runtime_seconds": time.perf_counter() - started,
            "gpu_memory": gpu_memory_summary(),
        }
        if stats.count >= 2:
            diagnostics["running_rho"] = mapping_summary(stats.sample_covariance())
        probe_diagnostics.append(diagnostics)
        LOGGER.info("Probe diagnostics: %s", json.dumps(diagnostics, sort_keys=True))
        del x, y

        if stats.count in convergence_checkpoints:
            snapshot_path = _save_snapshot(output_path, stats.count, parameter_space, stats, eta)
            snapshot_paths.append(str(snapshot_path))
            LOGGER.info("Saved cumulative score snapshot: %s", snapshot_path)

    scores, damage, recovery, h_ref, rho = compute_score_components(parameter_space, stats, eta=eta)
    diagnostics = final_diagnostics(h_ref, rho, damage, recovery, scores)
    payload: dict[str, object] = {
        "metadata": {
            **metadata,
            "num_probes": num_probes,
            "probe_seed": probe_seed,
            "eta": eta,
            "loss": "shifted_token_normalized_causal_nll",
            "convergence_snapshots": snapshot_paths,
        },
        "scores": scores,
        "damage": damage,
        "recovery": recovery,
    }
    if save_intermediate_stats:
        payload["h_ref"] = h_ref
        payload["rho"] = rho
    _atomic_torch_save(payload, output_path)

    diagnostics_payload = {
        "metadata": payload["metadata"],
        "probe_diagnostics": probe_diagnostics,
        "final": diagnostics,
    }
    diagnostics_path = output_path.with_suffix(output_path.suffix + ".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved score file to %s", output_path)
    LOGGER.info("Saved diagnostics to %s", diagnostics_path)
    return payload


def run_single_probe_smoke_test(
    model,
    ref_loader,
    kd_loader,
    parameter_space: ParameterSpace,
    *,
    probe_seed: int,
) -> dict[str, object]:
    probe = make_rademacher_probe(
        parameter_space.hvp_names,
        parameter_space.hvp_parameters,
        seed=probe_seed,
    )
    reference_hvp, reference_info = compute_dataset_hvp(
        model,
        ref_loader,
        probe,
        parameter_space,
        causal_sft_nll,
        objective_name="reference",
    )
    kd_hvp, kd_info = compute_dataset_hvp(
        model,
        kd_loader,
        probe,
        parameter_space,
        causal_sft_nll,
        objective_name="kd",
    )
    x = {name: reference_hvp[name] * probe[name].float() for name in parameter_space.candidate_names}
    y = {name: kd_hvp[name] * probe[name].float() for name in parameter_space.candidate_names}
    return {
        "probe_seed": probe_seed,
        "reference": reference_info,
        "kd": kd_info,
        "x": mapping_summary(x),
        "y": mapping_summary(y),
        "gpu_memory": gpu_memory_summary(),
    }
