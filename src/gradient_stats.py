from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from calibration_loaders import load_calibration_examples, make_calibration_dataloader
from layer_utils import iter_prunable_modules
from model_utils import temporarily_disable_cache

LOGGER = logging.getLogger(__name__)


def collect_gradient_stats(model, tokenizer, *, calibration_path: str, output_dir: str | Path | None = None, calibration_type: str = "prompt_response", only_correct: bool = False, max_calibration_samples: int | None = None, microbatch_size: int = 1, gradient_accumulation_steps: int = 1, loss_on: str = "response_only", max_length: int = 4096, device: str | None = None, prune_ops=None, dtype: str = "bf16", seed: int = 42, model_name: str = "") -> dict[str, dict[str, torch.Tensor]]:
    LOGGER.warning("Collecting microbatch empirical Fisher approximation; not exact per-sample Fisher.")
    if device is None:
        device = str(next(model.parameters()).device)
    examples = load_calibration_examples(calibration_path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_calibration_samples, shuffle=False, seed=seed)
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on=loss_on, microbatch_size=microbatch_size)
    modules = dict(iter_prunable_modules(model, prune_ops))
    grad_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    grad_sq_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    count = 0
    model.train(False)
    model.zero_grad(set_to_none=True)
    with temporarily_disable_cache(model):
        for batch in tqdm(dataloader, desc="gradient stats"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            count += 1
            _accumulate_and_zero(model, modules, grad_sum, grad_sq_sum)
    stats = {name: {"g": grad_sum[name] / max(count, 1), "h": grad_sq_sum[name] / max(count, 1), "count": torch.tensor(count)} for name in modules}
    if output_dir is not None:
        save_gradient_stats(stats, output_dir, metadata={"model_name": model_name, "calibration_path": calibration_path, "number_of_examples": len(examples), "loss_on": loss_on, "microbatch_size": microbatch_size, "gradient_accumulation_steps": gradient_accumulation_steps, "prune_ops": prune_ops, "dtype": dtype, "seed": seed, "note": "microbatch empirical Fisher approximation"})
    return stats


def _accumulate_and_zero(model, modules, grad_sum, grad_sq_sum):
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
