from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from calibration_loaders import load_calibration_examples, make_calibration_dataloader
from layer_utils import iter_prunable_modules
from model_utils import temporarily_disable_cache

LOGGER = logging.getLogger(__name__)


def collect_gradient_stats(model, tokenizer, *, calibration_path: str, output_dir: str | Path | None = None, calibration_type: str = "prompt_response", only_correct: bool = False, max_calibration_samples: int | None = None, microbatch_size: int = 1, gradient_accumulation_steps: int = 1, loss_on: str = "response_only", max_length: int = 4096, device: str | None = None, prune_ops=None, dtype: str = "bf16", seed: int = 42, model_name: str = "", fisher_estimator: str = "microbatch", shuffle: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    fisher_estimator = str(fisher_estimator).lower()
    if fisher_estimator not in {"microbatch", "per_example"}:
        raise ValueError(f"Unsupported fisher_estimator={fisher_estimator!r}; use 'microbatch' or 'per_example'")
    if fisher_estimator == "microbatch":
        LOGGER.warning("Collecting microbatch empirical Fisher approximation; h depends on microbatch_size.")
    else:
        LOGGER.warning("Collecting per-example empirical Fisher; h is independent of microbatch grouping but requires one backward per example.")
    rank, world_size = _distributed_rank_and_world_size()
    if device is None:
        device = str(next(model.parameters()).device)
    examples = load_calibration_examples(calibration_path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_calibration_samples, shuffle=shuffle, seed=seed)
    total_examples = len(examples)
    if world_size > 1 and fisher_estimator == "per_example":
        examples = examples[rank::world_size]
        LOGGER.info("Rank %d/%d collecting per-example gradient stats on %d/%d calibration examples", rank, world_size, len(examples), total_examples)
    elif world_size > 1:
        LOGGER.info("Rank %d/%d collecting every %d-th microbatch from %d calibration examples", rank, world_size, world_size, total_examples)
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on=loss_on, microbatch_size=microbatch_size)
    modules = dict(iter_prunable_modules(model, prune_ops))
    grad_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    grad_sq_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    grad_abs_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    count = 0
    model.train(False)
    model.zero_grad(set_to_none=True)
    with temporarily_disable_cache(model):
        for batch_index, batch in enumerate(tqdm(dataloader, desc=f"gradient stats ({fisher_estimator})")):
            if world_size > 1 and fisher_estimator == "microbatch" and batch_index % world_size != rank:
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            if fisher_estimator == "microbatch":
                outputs = model(**batch)
                outputs.loss.backward()
                count += 1
                _accumulate_and_zero(model, modules, grad_sum, grad_sq_sum, grad_abs_sum)
            else:
                for example_batch in _iter_single_examples(batch):
                    outputs = model(**example_batch)
                    outputs.loss.backward()
                    count += 1
                    _accumulate_and_zero(model, modules, grad_sum, grad_sq_sum, grad_abs_sum)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_count = _reduce_gradient_sums(modules, grad_sum, grad_sq_sum, grad_abs_sum, count)
    stats = {name: {"g": grad_sum[name] / max(total_count, 1), "h": grad_sq_sum[name] / max(total_count, 1), "abs_g": grad_abs_sum[name] / max(total_count, 1), "count": torch.tensor(total_count)} for name in modules}
    if output_dir is not None and _is_main_process():
        note = "microbatch empirical Fisher approximation" if fisher_estimator == "microbatch" else "per-example empirical Fisher"
        save_gradient_stats(stats, output_dir, metadata={"model_name": model_name, "calibration_path": calibration_path, "number_of_examples": total_examples, "loss_on": loss_on, "microbatch_size": microbatch_size, "fisher_estimator": fisher_estimator, "gradient_accumulation_steps": gradient_accumulation_steps, "prune_ops": prune_ops, "dtype": dtype, "seed": seed, "shuffle": shuffle, "distributed_world_size": world_size, "note": note})
    return stats


def _distributed_rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_main_process() -> bool:
    rank, _ = _distributed_rank_and_world_size()
    return rank == 0


def _reduce_gradient_sums(modules, grad_sum, grad_sq_sum, grad_abs_sum, count: int) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return count
    count_tensor = torch.tensor([count], dtype=torch.long)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    for name in modules:
        dist.all_reduce(grad_sum[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_sq_sum[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_abs_sum[name], op=dist.ReduceOp.SUM)
    return int(count_tensor.item())


def _iter_single_examples(batch: dict[str, torch.Tensor]):
    first_tensor = next(iter(batch.values()))
    batch_size = first_tensor.shape[0]
    for idx in range(batch_size):
        yield {key: value[idx : idx + 1] for key, value in batch.items()}


def _accumulate_and_zero(model, modules, grad_sum, grad_sq_sum, grad_abs_sum):
    for name, module in modules.items():
        if module.weight.grad is None:
            continue
        grad = module.weight.grad.detach().float().cpu()
        grad_sum[name] += grad
        grad_sq_sum[name] += grad.pow(2)
    model.zero_grad(set_to_none=True)


def save_gradient_stats(stats: dict[str, dict[str, torch.Tensor]], output_dir: str | Path, metadata: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_index = {}
    for name, tensors in stats.items():
        safe = name.replace(".", "__") + ".pt"
        torch.save(tensors, output_dir / safe)
        safe_index[name] = safe
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"modules": safe_index, **metadata}, handle, indent=2, default=str)


def load_gradient_stats(stats_dir: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    stats_dir = Path(stats_dir)
    meta = json.loads((stats_dir / "metadata.json").read_text())
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in meta["modules"].items()}
