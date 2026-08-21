from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .diagnostics import parameter_layer_name


def _load_scores(path: Path) -> tuple[int | None, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "scores" not in payload:
        raise ValueError(f"Score file does not contain a 'scores' dictionary: {path}")
    probe_count = payload.get("num_probes")
    if probe_count is None and isinstance(payload.get("metadata"), dict):
        probe_count = payload["metadata"].get("num_probes")
    return int(probe_count) if probe_count is not None else None, payload["scores"]


def _sample_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    max_elements: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    if first_flat.numel() != second_flat.numel():
        raise ValueError(f"Score tensor sizes differ: {first_flat.numel()} vs {second_flat.numel()}")
    exact = first_flat.numel() <= max_elements
    if exact:
        return first_flat.numpy(), second_flat.numpy(), True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(0, first_flat.numel(), (max_elements,), generator=generator)
    return first_flat[indices].numpy(), second_flat[indices].numpy(), False


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    result = spearmanr(first, second).statistic
    return float(result)


def _sample_mapping_pair(
    first_scores: dict[str, torch.Tensor],
    second_scores: dict[str, torch.Tensor],
    names: list[str],
    *,
    max_elements: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    sizes = [first_scores[name].numel() for name in names]
    if any(size != second_scores[name].numel() for name, size in zip(names, sizes, strict=True)):
        raise ValueError("Score tensor sizes differ between convergence snapshots")
    total = sum(sizes)
    if total <= max_elements:
        first = torch.cat([first_scores[name].detach().float().reshape(-1) for name in names])
        second = torch.cat([second_scores[name].detach().float().reshape(-1) for name in names])
        return first.numpy(), second.numpy(), True

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    global_indices = torch.randint(0, total, (max_elements,), generator=generator)
    sorted_indices, _ = torch.sort(global_indices)
    first_sample = torch.empty(max_elements, dtype=torch.float32)
    second_sample = torch.empty(max_elements, dtype=torch.float32)
    offset = 0
    output_offset = 0
    for name, size in zip(names, sizes, strict=True):
        start = int(torch.searchsorted(sorted_indices, offset, right=False).item())
        end = int(torch.searchsorted(sorted_indices, offset + size, right=False).item())
        if end > start:
            local_indices = sorted_indices[start:end] - offset
            count = end - start
            first_sample[output_offset : output_offset + count] = first_scores[name].detach().float().reshape(-1)[local_indices]
            second_sample[output_offset : output_offset + count] = second_scores[name].detach().float().reshape(-1)[local_indices]
            output_offset += count
        offset += size
    if output_offset != max_elements:
        raise RuntimeError(f"Expected {max_elements} sampled scores, populated {output_offset}")
    return first_sample.numpy(), second_sample.numpy(), False


def compare_score_files(
    first_path: str | Path,
    second_path: str | Path,
    *,
    max_elements_per_matrix: int = 100_000,
    max_aggregate_elements: int = 2_000_000,
    seed: int = 42,
) -> dict[str, object]:
    first_path = Path(first_path)
    second_path = Path(second_path)
    first_probes, first_scores = _load_scores(first_path)
    second_probes, second_scores = _load_scores(second_path)
    if first_scores.keys() != second_scores.keys():
        missing_first = sorted(second_scores.keys() - first_scores.keys())
        missing_second = sorted(first_scores.keys() - second_scores.keys())
        raise ValueError(
            f"Score parameter sets differ; missing from first={missing_first[:10]}, missing from second={missing_second[:10]}"
        )

    per_matrix = {}
    layer_names: dict[str, list[str]] = defaultdict(list)
    for matrix_index, name in enumerate(first_scores):
        first_values, second_values, exact = _sample_pair(
            first_scores[name],
            second_scores[name],
            max_elements=max_elements_per_matrix,
            seed=seed + matrix_index,
        )
        per_matrix[name] = {
            "spearman": _correlation(first_values, second_values),
            "num_compared": int(first_values.size),
            "exact": exact,
        }
        layer = parameter_layer_name(name)
        layer_names[layer].append(name)

    def aggregate(names: list[str], aggregate_seed: int):
        first_values, second_values, exact = _sample_mapping_pair(
            first_scores,
            second_scores,
            names,
            max_elements=max_aggregate_elements,
            seed=aggregate_seed,
        )
        return {
            "spearman": _correlation(first_values, second_values),
            "num_compared": int(first_values.size),
            "exact": exact,
        }

    per_layer = {
        layer: aggregate(layer_names[layer], seed + 10_000 + index)
        for index, layer in enumerate(sorted(layer_names))
    }
    global_result = aggregate(list(first_scores), seed + 20_000)
    return {
        "first_path": str(first_path),
        "second_path": str(second_path),
        "first_num_probes": first_probes,
        "second_num_probes": second_probes,
        "sampling": {
            "max_elements_per_matrix": max_elements_per_matrix,
            "max_aggregate_elements": max_aggregate_elements,
            "seed": seed,
            "aggregate_sampling": "uniform_over_parameter_coordinates_with_replacement",
        },
        "global": global_result,
        "per_layer": per_layer,
        "per_matrix": per_matrix,
    }


def analyze_snapshots(
    paths: list[str | Path],
    *,
    max_elements_per_matrix: int = 100_000,
    max_aggregate_elements: int = 2_000_000,
    seed: int = 42,
) -> dict[str, object]:
    sorted_paths = sorted(Path(path) for path in paths)
    if len(sorted_paths) < 2:
        raise ValueError("At least two score snapshots are required for convergence analysis")
    comparisons = []
    for first_index, first_path in enumerate(sorted_paths[:-1]):
        for second_path in sorted_paths[first_index + 1 :]:
            comparisons.append(
                compare_score_files(
                    first_path,
                    second_path,
                    max_elements_per_matrix=max_elements_per_matrix,
                    max_aggregate_elements=max_aggregate_elements,
                    seed=seed,
                )
            )
    return {"snapshots": [str(path) for path in sorted_paths], "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure pruning-score ranking convergence with Spearman correlation")
    parser.add_argument("score_paths", nargs="+")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_elements_per_matrix", type=int, default=100_000)
    parser.add_argument("--max_aggregate_elements", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze_snapshots(
        args.score_paths,
        max_elements_per_matrix=args.max_elements_per_matrix,
        max_aggregate_elements=args.max_aggregate_elements,
        seed=args.seed,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
