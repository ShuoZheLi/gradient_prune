from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .diagnostics import (
    final_diagnostics,
    gpu_memory_summary,
    host_memory_summary,
    mapping_product_summary,
    mapping_summary,
)
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


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


def _candidate_templates(parameter_space: ParameterSpace) -> dict[str, torch.Tensor]:
    return {
        name: parameter
        for name, parameter in zip(
            parameter_space.candidate_names,
            parameter_space.candidate_parameters,
            strict=True,
        )
    }


def _compute_probe(
    model,
    ref_loader,
    kd_loader,
    parameter_space: ParameterSpace,
    stats: OnlineMeanCovariance,
    *,
    probe_index: int,
    total_probes: int,
    probe_seed: int,
    rank: int = 0,
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    seed = probe_seed + probe_index
    LOGGER.info(
        "Rank %d starting shared Rademacher probe %d/%d seed=%d",
        rank,
        probe_index + 1,
        total_probes,
        seed,
    )
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
        objective_name=f"reference/rank{rank}",
        activation_offload=activation_offload,
        activation_offload_pin_memory=activation_offload_pin_memory,
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
        objective_name=f"kd/rank{rank}",
        activation_offload=activation_offload,
        activation_offload_pin_memory=activation_offload_pin_memory,
    )
    y = {
        name: kd_hvp[name] * probe[name].to(dtype=torch.float32)
        for name in parameter_space.candidate_names
    }
    del kd_hvp, probe
    stats.update(x, y)
    diagnostics = {
        "probe_index": probe_index + 1,
        "rank": rank,
        "seed": seed,
        "reference": reference_info,
        "kd": kd_info,
        "x": mapping_summary(x),
        "y": mapping_summary(y),
        "running_h_ref_local": mapping_summary(stats.mean_x),
        "local_probe_count": stats.count,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_memory": gpu_memory_summary(),
    }
    if stats.count >= 2:
        diagnostics["running_rho_local"] = mapping_summary(
            stats.cov_m2,
            scale=1.0 / (stats.count - 1),
        )
    LOGGER.info("Probe diagnostics: %s", json.dumps(diagnostics, sort_keys=True))
    del x, y
    return diagnostics


def _finalize_scores(
    parameter_space: ParameterSpace,
    stats: OnlineMeanCovariance,
    *,
    num_probes: int,
    probe_seed: int,
    eta: float,
    output_path: Path,
    metadata: dict[str, object],
    save_intermediate_stats: bool,
    snapshot_paths: list[str],
    probe_diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    if stats.count != num_probes:
        raise ValueError(f"Merged probe count {stats.count} does not match requested num_probes={num_probes}")
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
        "probe_diagnostics": sorted(probe_diagnostics, key=lambda item: int(item["probe_index"])),
        "final": diagnostics,
    }
    diagnostics_path = output_path.with_suffix(output_path.suffix + ".diagnostics.json")
    _atomic_json_save(diagnostics_payload, diagnostics_path)
    LOGGER.info("Saved score file to %s", output_path)
    LOGGER.info("Saved diagnostics to %s", diagnostics_path)
    return payload


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
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
) -> dict[str, object]:
    if num_probes < 2:
        raise ValueError("num_probes must be at least 2 for unbiased sample covariance; use --smoke_test for M=1")
    stats = OnlineMeanCovariance(_candidate_templates(parameter_space))
    output_path = Path(output_path)
    probe_diagnostics = []
    snapshot_paths = []

    for probe_index in range(num_probes):
        probe_diagnostics.append(
            _compute_probe(
                model,
                ref_loader,
                kd_loader,
                parameter_space,
                stats,
                probe_index=probe_index,
                total_probes=num_probes,
                probe_seed=probe_seed,
                activation_offload=activation_offload,
                activation_offload_pin_memory=activation_offload_pin_memory,
            )
        )
        if stats.count in convergence_checkpoints:
            snapshot_path = _save_snapshot(output_path, stats.count, parameter_space, stats, eta)
            snapshot_paths.append(str(snapshot_path))
            LOGGER.info("Saved cumulative score snapshot: %s", snapshot_path)

    return _finalize_scores(
        parameter_space,
        stats,
        num_probes=num_probes,
        probe_seed=probe_seed,
        eta=eta,
        output_path=output_path,
        metadata=metadata,
        save_intermediate_stats=save_intermediate_stats,
        snapshot_paths=snapshot_paths,
        probe_diagnostics=probe_diagnostics,
    )


def contiguous_probe_range(num_probes: int, world_size: int, rank: int) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    base, remainder = divmod(num_probes, world_size)
    start = rank * base + min(rank, remainder)
    count = base + int(rank < remainder)
    return start, start + count


def validate_distributed_checkpoints(
    num_probes: int,
    world_size: int,
    convergence_checkpoints: tuple[int, ...],
) -> None:
    if world_size > num_probes:
        raise ValueError(
            f"Probe-parallel world_size={world_size} exceeds num_probes={num_probes}; some nodes would be idle"
        )
    first_start, first_end = contiguous_probe_range(num_probes, world_size, 0)
    if first_start != 0:
        raise RuntimeError("Rank 0 probe block must start at zero")
    supported = set(range(2, first_end + 1))
    supported.update(contiguous_probe_range(num_probes, world_size, rank)[1] for rank in range(world_size))
    unsupported = sorted(set(convergence_checkpoints) - supported)
    if unsupported:
        raise ValueError(
            "Distributed convergence checkpoints must fall within rank 0's initial probe block "
            f"or at a rank-block boundary. Unsupported={unsupported}; supported={sorted(supported)}"
        )


def _stats_file_name(index: int, parameter_name: str) -> str:
    return f"{index:04d}_{parameter_name.replace('.', '__')}.pt"


def save_stats_directory(stats: OnlineMeanCovariance, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for index, name in enumerate(stats.mean_x):
        file_name = _stats_file_name(index, name)
        _atomic_torch_save(
            {
                "mean_x": stats.mean_x[name],
                "mean_y": stats.mean_y[name],
                "cov_m2": stats.cov_m2[name],
            },
            directory / file_name,
        )
        files[name] = file_name
    _atomic_json_save({"count": stats.count, "files": files}, directory / "manifest.json")
    return directory


def merge_stats_directory(stats: OnlineMeanCovariance, directory: str | Path, *, delete_after: bool = False) -> None:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    other_count = int(manifest["count"])
    if other_count == 0:
        return
    if list(manifest["files"]) != list(stats.mean_x):
        raise ValueError(f"Distributed statistic parameter order mismatch in {directory}")
    old_count = stats.count
    new_count = old_count + other_count
    cross_factor = old_count * other_count / new_count if old_count else 0.0
    for name, file_name in manifest["files"].items():
        state = torch.load(directory / file_name, map_location="cpu", weights_only=False)
        if old_count == 0:
            stats.mean_x[name].copy_(state["mean_x"])
            stats.mean_y[name].copy_(state["mean_y"])
            stats.cov_m2[name].copy_(state["cov_m2"])
        else:
            delta_x = state["mean_x"] - stats.mean_x[name]
            delta_y = state["mean_y"] - stats.mean_y[name]
            stats.cov_m2[name].add_(state["cov_m2"])
            stats.cov_m2[name].addcmul_(delta_x, delta_y, value=cross_factor)
            stats.mean_x[name].add_(delta_x, alpha=other_count / new_count)
            stats.mean_y[name].add_(delta_y, alpha=other_count / new_count)
        del state
        if delete_after:
            (directory / file_name).unlink()
    stats.count = new_count
    if delete_after:
        (directory / "manifest.json").unlink()
        directory.rmdir()


def compute_distributed_dense_pruning_scores(
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
    convergence_checkpoints: tuple[int, ...],
    rank: int,
    world_size: int,
    distributed_state_dir: str | Path | None = None,
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
) -> dict[str, object] | None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for distributed probe scoring")
    validate_distributed_checkpoints(num_probes, world_size, convergence_checkpoints)
    output_path = Path(output_path)
    state_root = (
        Path(distributed_state_dir)
        if distributed_state_dir is not None
        else output_path.parent / f".{output_path.stem}_distributed_state"
    )
    if rank == 0:
        shutil.rmtree(state_root, ignore_errors=True)
        state_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    start, end = contiguous_probe_range(num_probes, world_size, rank)
    LOGGER.info("Rank %d/%d assigned global probes [%d, %d)", rank, world_size, start, end)
    stats = OnlineMeanCovariance(_candidate_templates(parameter_space))
    local_diagnostics = []
    snapshot_paths: list[str] = []
    for probe_index in range(start, end):
        local_diagnostics.append(
            _compute_probe(
                model,
                ref_loader,
                kd_loader,
                parameter_space,
                stats,
                probe_index=probe_index,
                total_probes=num_probes,
                probe_seed=probe_seed,
                rank=rank,
                activation_offload=activation_offload,
                activation_offload_pin_memory=activation_offload_pin_memory,
            )
        )
        global_prefix_count = probe_index + 1
        if rank == 0 and global_prefix_count in convergence_checkpoints:
            snapshot_path = _save_snapshot(output_path, global_prefix_count, parameter_space, stats, eta)
            snapshot_paths.append(str(snapshot_path))
            LOGGER.info("Saved distributed cumulative score snapshot: %s", snapshot_path)

    _atomic_json_save(local_diagnostics, state_root / f"probe_diagnostics_rank_{rank:04d}.json")
    if rank != 0:
        save_stats_directory(stats, state_root / f"rank_{rank:04d}")
    dist.barrier()

    if rank != 0:
        return None

    probe_diagnostics = list(local_diagnostics)
    for source_rank in range(1, world_size):
        source_diagnostics_path = state_root / f"probe_diagnostics_rank_{source_rank:04d}.json"
        probe_diagnostics.extend(json.loads(source_diagnostics_path.read_text(encoding="utf-8")))
        merge_stats_directory(stats, state_root / f"rank_{source_rank:04d}", delete_after=True)
        LOGGER.info("Merged rank %d statistics; global probe count=%d", source_rank, stats.count)
        if stats.count in convergence_checkpoints:
            snapshot_path = _save_snapshot(output_path, stats.count, parameter_space, stats, eta)
            snapshot_paths.append(str(snapshot_path))
            LOGGER.info("Saved distributed cumulative score snapshot: %s", snapshot_path)

    metadata = {
        **metadata,
        "distributed_probe_parallel": True,
        "distributed_world_size": world_size,
        "distributed_probe_ranges": {
            str(source_rank): list(contiguous_probe_range(num_probes, world_size, source_rank))
            for source_rank in range(world_size)
        },
    }
    payload = _finalize_scores(
        parameter_space,
        stats,
        num_probes=num_probes,
        probe_seed=probe_seed,
        eta=eta,
        output_path=output_path,
        metadata=metadata,
        save_intermediate_stats=save_intermediate_stats,
        snapshot_paths=sorted(set(snapshot_paths)),
        probe_diagnostics=probe_diagnostics,
    )
    for diagnostics_path in state_root.glob("probe_diagnostics_rank_*.json"):
        diagnostics_path.unlink()
    state_root.rmdir()
    return payload


def run_single_probe_smoke_test(
    model,
    ref_loader,
    kd_loader,
    parameter_space: ParameterSpace,
    *,
    probe_seed: int,
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
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
        activation_offload=activation_offload,
        activation_offload_pin_memory=activation_offload_pin_memory,
    )
    LOGGER.info("Smoke reference HVP host memory: %s", json.dumps(host_memory_summary(), sort_keys=True))
    x_summary = mapping_product_summary(
        reference_hvp,
        {name: probe[name] for name in parameter_space.candidate_names},
    )
    del reference_hvp
    kd_hvp, kd_info = compute_dataset_hvp(
        model,
        kd_loader,
        probe,
        parameter_space,
        causal_sft_nll,
        objective_name="kd",
        activation_offload=activation_offload,
        activation_offload_pin_memory=activation_offload_pin_memory,
    )
    LOGGER.info("Smoke KD HVP host memory: %s", json.dumps(host_memory_summary(), sort_keys=True))
    y_summary = mapping_product_summary(
        kd_hvp,
        {name: probe[name] for name in parameter_space.candidate_names},
    )
    del kd_hvp, probe
    return {
        "probe_seed": probe_seed,
        "reference": reference_info,
        "kd": kd_info,
        "x": x_summary,
        "y": y_summary,
        "gpu_memory": gpu_memory_summary(),
        "host_memory": host_memory_summary(),
    }
