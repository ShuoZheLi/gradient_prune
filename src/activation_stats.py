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


def collect_activation_stats(model, tokenizer, *, calibration_path: str, output_dir: str | Path | None = None, calibration_type: str = "prompt_response", only_correct: bool = False, max_calibration_samples: int | None = None, microbatch_size: int = 1, loss_on: str = "full_trajectory", max_length: int = 4096, device: str | None = None, prune_ops=None, seed: int = 42, model_name: str = "") -> dict[str, torch.Tensor]:
    if device is None:
        device = str(next(model.parameters()).device)
    examples = load_calibration_examples(calibration_path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_calibration_samples, shuffle=False, seed=seed)
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
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()
    stats = {name: (sq_sums[name] / max(counts[name], 1)).sqrt().float() for name in modules}
    if output_dir is not None:
        save_activation_stats(stats, output_dir, metadata={"model_name": model_name, "calibration_path": calibration_path, "number_of_examples": len(examples), "loss_on": loss_on, "microbatch_size": microbatch_size, "prune_ops": prune_ops, "seed": seed, "definition": "sqrt(mean over observed tokens of x_j^2)"})
    return stats


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
