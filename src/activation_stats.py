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


def collect_activation_stats(model, tokenizer, *, calibration_path: str, output_dir: str | Path | None = None, calibration_type: str = "prompt_response", only_correct: bool = False, max_calibration_samples: int | None = None, microbatch_size: int = 1, loss_on: str = "full_trajectory", max_length: int = 4096, device: str | None = None, prune_ops=None, seed: int = 42, model_name: str = "", shuffle: bool = False) -> dict[str, torch.Tensor]:
    if device is None:
        device = str(next(model.parameters()).device)
    rank, world_size = _distributed_rank_and_world_size()
    examples = load_calibration_examples(calibration_path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_calibration_samples, shuffle=shuffle, seed=seed)
    total_examples = len(examples)
    if world_size > 1:
        examples = examples[rank::world_size]
        LOGGER.info("Rank %d/%d collecting activation stats on %d/%d calibration examples", rank, world_size, len(examples), total_examples)
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on=loss_on, microbatch_size=microbatch_size)
    modules = dict(iter_prunable_modules(model, prune_ops))
    sq_sums = {name: torch.zeros(module.in_features, dtype=torch.float64) for name, module in modules.items()}
    counts = {name: 0 for name in modules}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            flat = x.reshape(-1, x.shape[-1])
            sq_sums[name].add_(flat.pow(2).sum(dim=0).cpu().double())
            counts[name] += flat.shape[0]
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        model.eval()
        with torch.no_grad(), temporarily_disable_cache(model):
            for batch in tqdm(dataloader, desc="activation stats"):
                batch = {key: value.to(device) for key, value in batch.items()}
                _forward_without_lm_head(model, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()
    _reduce_activation_sums(modules, sq_sums, counts)
    stats = {name: (sq_sums[name] / max(counts[name], 1)).sqrt().float() for name in modules}
    if output_dir is not None and _is_main_process():
        save_activation_stats(stats, output_dir, metadata={"model_name": model_name, "calibration_path": calibration_path, "number_of_examples": total_examples, "loss_on": loss_on, "microbatch_size": microbatch_size, "prune_ops": prune_ops, "seed": seed, "shuffle": shuffle, "distributed_world_size": world_size, "definition": "sqrt(mean over observed tokens of x_j^2)"})
    return stats


def _forward_without_lm_head(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    base_model = getattr(model, "model", None)
    if base_model is not None:
        base_model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        model(input_ids=input_ids, attention_mask=attention_mask)


def _distributed_rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_main_process() -> bool:
    rank, _ = _distributed_rank_and_world_size()
    return rank == 0


def _reduce_activation_sums(modules, sq_sums, counts) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for name in modules:
        dist.all_reduce(sq_sums[name], op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([counts[name]], dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts[name] = int(count_tensor.item())


def save_activation_stats(stats: dict[str, torch.Tensor], output_dir: str | Path, metadata: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_index = {}
    for name, tensor in stats.items():
        safe = name.replace(".", "__") + ".pt"
        torch.save(tensor.cpu(), output_dir / safe)
        safe_index[name] = safe
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"modules": safe_index, **metadata}, handle, indent=2, default=str)


def load_activation_stats(stats_dir: str | Path) -> dict[str, torch.Tensor]:
    stats_dir = Path(stats_dir)
    meta = json.loads((stats_dir / "metadata.json").read_text())
    return {name: torch.load(stats_dir / file_name, map_location="cpu") for name, file_name in meta["modules"].items()}
