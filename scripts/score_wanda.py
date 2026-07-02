from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from activation_stats import collect_activation_stats
from layer_utils import iter_prunable_modules, normalize_prune_ops
from model_utils import load_model_and_tokenizer
from pruning_scores import wanda_score

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect WANDA pruning scores for HF causal LM linear modules.")
    parser.add_argument("--model", required=True, help="Model name or local checkpoint path.")
    parser.add_argument("--calibration", required=True, help="Calibration dataset path/directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where WANDA scores are written.")
    parser.add_argument("--calibration-type", default="prompt_response", choices=["prompt_response", "text"], help="How to interpret calibration records.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on calibration examples.")
    parser.add_argument("--microbatch-size", type=int, default=1, help="Calibration forward microbatch size.")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum tokenized sequence length.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype: bf16, fp16, fp32, or auto.")
    parser.add_argument("--device", default=None, help="Device for single-process runs. Defaults to cuda:0 when available.")
    parser.add_argument("--prune-ops", nargs="*", default=None, help="Optional subset of prunable ops, e.g. q k v o up gate down.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle calibration examples before optional max-samples.")
    parser.add_argument("--only-correct", action="store_true", help="Filter calibration rows where is_correct is true when present.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing complete output directory.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed_from_torchrun() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if dist.is_initialized():
        return True
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo", init_method="env://")
    LOGGER.info("Initialized distributed WANDA scoring rank %d/%d", dist.get_rank(), dist.get_world_size())
    return True


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_prune_ops(prune_ops: list[str] | None) -> tuple[str, ...] | None:
    if not prune_ops:
        return None
    return normalize_prune_ops(prune_ops)


def output_is_complete(output_dir: Path) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False
    modules = metadata.get("modules")
    return isinstance(modules, dict) and bool(modules) and all((output_dir / file_name).is_file() for file_name in modules.values())


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    if not bool(finite.all()):
        valid = tensor[finite]
    else:
        valid = tensor
    if valid.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": int(tensor.numel()), "finite": int(finite.sum().item())}
    valid_float = valid.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "finite": int(finite.sum().item()),
        "min": float(valid_float.min().item()),
        "max": float(valid_float.max().item()),
        "mean": float(valid_float.mean().item()),
    }


def save_wanda_scores(model, activation_stats: dict[str, torch.Tensor], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    modules = dict(iter_prunable_modules(model, metadata.get("prune_ops")))
    index: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    total_numel = 0
    for name, module in tqdm(modules.items(), desc="Saving WANDA scores", unit="module"):
        if name not in activation_stats:
            raise KeyError(f"Missing activation stats for {name}")
        score = wanda_score(module.weight.detach().cpu(), activation_stats[name].cpu())
        if tuple(score.shape) != tuple(module.weight.shape):
            raise ValueError(f"WANDA score shape {tuple(score.shape)} for {name} does not match weight shape {tuple(module.weight.shape)}")
        safe_name = f"{name.replace('.', '__')}.pt"
        torch.save({"wanda": score.float()}, output_dir / safe_name)
        index[name] = safe_name
        summaries[name] = tensor_summary(score)
        total_numel += score.numel()
    full_metadata = {
        **metadata,
        "score_key": "wanda",
        "definition": "abs(weight) * sqrt(mean over observed calibration tokens of input_activation_j^2)",
        "modules": index,
        "summaries": summaries,
        "num_modules": len(index),
        "num_total_scores": int(total_numel),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(full_metadata, handle, indent=2, default=str)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_is_complete(output_dir) and not args.overwrite:
        LOGGER.info("Output already appears complete; use --overwrite to recompute: %s", output_dir)
        return 0

    distributed = init_distributed_from_torchrun()
    if distributed and torch.cuda.is_available():
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    prune_ops = unwrap_prune_ops(args.prune_ops)
    rank = distributed_rank()
    world_size = distributed_world_size()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    LOGGER.info("Rank %d/%d loading model on %s", rank, world_size, args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device, args.trust_remote_code)
    LOGGER.info("Rank %d/%d collecting activation statistics", rank, world_size)
    activation_output_dir = output_dir / "activation_stats" if rank == 0 else None
    activation_stats = collect_activation_stats(
        model,
        tokenizer,
        calibration_path=args.calibration,
        output_dir=activation_output_dir,
        calibration_type=args.calibration_type,
        only_correct=args.only_correct,
        max_calibration_samples=args.max_samples,
        microbatch_size=args.microbatch_size,
        loss_on="full_trajectory",
        max_length=args.max_length,
        device=args.device,
        prune_ops=prune_ops,
        seed=args.seed,
        model_name=args.model,
        shuffle=args.shuffle,
    )

    LOGGER.info("Rank %d/%d finished activation statistics", rank, world_size)
    if rank == 0:
        LOGGER.info("Rank 0 saving WANDA score tensors")
        save_wanda_scores(
            model,
            activation_stats,
            output_dir,
            {
                "model_name": args.model,
                "calibration_path": args.calibration,
                "calibration_type": args.calibration_type,
                "only_correct": args.only_correct,
                "max_calibration_samples": args.max_samples,
                "microbatch_size": args.microbatch_size,
                "max_length": args.max_length,
                "dtype": args.dtype,
                "prune_ops": prune_ops,
                "seed": args.seed,
                "shuffle": args.shuffle,
                "distributed_world_size": world_size,
            },
        )
        LOGGER.info("WANDA scores saved to %s", output_dir)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("WANDA scoring failed")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        raise
