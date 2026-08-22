from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import assert_disjoint_datasets, load_trajectory_dataset, make_trajectory_dataloader
from .factorized_factors import (
    collect_dataset_factors,
    compute_group_gradient_diagnostic,
    process_peak_cpu_memory_bytes,
)
from .factorized_scoring import factorized_score_tensors, module_diagnostics, save_score_shard
from .params import DEFAULT_CANDIDATE_MODULES, parse_module_patterns


LOGGER = logging.getLogger(__name__)
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate layer-factorized recoverability pruning scores")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ref_dataset_path", required=True)
    parser.add_argument("--kd_dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_lr_eta", "--eta", dest="eta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ref_batch_size", type=int)
    parser.add_argument("--kd_batch_size", type=int)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_ref_samples", type=int)
    parser.add_argument("--max_kd_samples", type=int)
    parser.add_argument("--candidate_modules", nargs="+", default=list(DEFAULT_CANDIDATE_MODULES))
    parser.add_argument("--module_names", nargs="*", help="Optional exact module-name subset")
    parser.add_argument("--layer_group_size", type=int, default=1)
    parser.add_argument("--factor_structure", choices=["full", "block", "diagonal_g"], default="full")
    parser.add_argument("--factor_storage_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--factor_chunk_size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--loss_on", choices=["full_trajectory", "response_only", "loss_mask"], required=True)
    parser.add_argument("--token_ids_column", default="prompt_generated_trajectory_ids")
    parser.add_argument("--prompt_length_column", default="prompt_length")
    parser.add_argument("--loss_mask_column", default="loss_mask")
    parser.add_argument("--disjoint_key_column", default="prompt")
    parser.add_argument("--truncation_side", choices=["left", "right"], default="right")
    parser.add_argument("--shuffle_ref", action="store_true")
    parser.add_argument("--shuffle_kd", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attention_implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--diagnostic_batches", type=int, default=1)
    parser.add_argument("--save_factor_diagnostics", action="store_true")
    parser.add_argument("--distributed_layer_sharding", action="store_true")
    parser.add_argument("--layer_shard_strategy", choices=["round_robin", "contiguous"], default="round_robin")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def initialize_distributed(args: argparse.Namespace) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return rank, world_size
    if not args.distributed_layer_sharding:
        raise ValueError(f"WORLD_SIZE={world_size} requires --distributed_layer_sharding")
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    return rank, world_size


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device(args.device))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    return model, tokenizer


def candidate_linear_modules(
    model: nn.Module,
    patterns: tuple[str, ...],
    exact_names: list[str] | None,
) -> OrderedDict[str, nn.Linear]:
    exact = set(exact_names or [])
    selected: OrderedDict[str, nn.Linear] = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        matched = any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)
        if matched and (not exact or name in exact):
            selected[name] = module
    if exact:
        missing = sorted(exact - selected.keys())
        if missing:
            raise ValueError(f"Requested module names were not selected linear candidates: {missing}")
    if not selected:
        raise ValueError(f"No nn.Linear modules matched candidate patterns {patterns}")
    return selected


def group_modules(modules: OrderedDict[str, nn.Linear], layer_group_size: int) -> list[OrderedDict[str, nn.Linear]]:
    if layer_group_size <= 0:
        raise ValueError(f"layer_group_size must be positive, got {layer_group_size}")
    by_layer: OrderedDict[str, OrderedDict[str, nn.Linear]] = OrderedDict()
    for name, module in modules.items():
        match = LAYER_PATTERN.search(name)
        key = f"layer_{int(match.group(1)):06d}" if match else f"other_{name}"
        by_layer.setdefault(key, OrderedDict())[name] = module
    layers = list(by_layer.values())
    return [
        OrderedDict((name, module) for layer in layers[start : start + layer_group_size] for name, module in layer.items())
        for start in range(0, len(layers), layer_group_size)
    ]


def shard_groups(groups, rank: int, world_size: int, strategy: str):
    if world_size == 1:
        return groups
    if strategy == "round_robin":
        return [group for index, group in enumerate(groups) if index % world_size == rank]
    start = len(groups) * rank // world_size
    end = len(groups) * (rank + 1) // world_size
    return groups[start:end]


def write_rank_manifest(output_dir: Path, rank: int, payload: dict[str, object]) -> None:
    path = output_dir / f"manifest.rank{rank:05d}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def merge_manifests(output_dir: Path, metadata: dict[str, object], world_size: int) -> None:
    modules = {}
    group_diagnostics = []
    for rank in range(world_size):
        payload = json.loads((output_dir / f"manifest.rank{rank:05d}.json").read_text(encoding="utf-8"))
        overlap = modules.keys() & payload["modules"].keys()
        if overlap:
            raise RuntimeError(f"Duplicate score shards across ranks: {sorted(overlap)}")
        modules.update(payload["modules"])
        group_diagnostics.extend(payload["group_diagnostics"])
    manifest = {**metadata, "modules": dict(sorted(modules.items())), "group_diagnostics": group_diagnostics}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.factor_structure != "full":
        raise NotImplementedError("Only --factor_structure full is implemented in the correctness-first version")
    rank, world_size = initialize_distributed(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    if Path(args.ref_dataset_path).resolve() == Path(args.kd_dataset_path).resolve():
        raise ValueError("Reference and KD dataset paths must differ")

    reference_dataset = load_trajectory_dataset(
        args.ref_dataset_path,
        token_ids_column=args.token_ids_column,
        loss_on=args.loss_on,
        prompt_length_column=args.prompt_length_column,
        loss_mask_column=args.loss_mask_column,
        disjoint_key_column=args.disjoint_key_column,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        max_samples=args.max_ref_samples,
        shuffle=args.shuffle_ref,
        seed=args.seed,
    )
    kd_dataset = load_trajectory_dataset(
        args.kd_dataset_path,
        token_ids_column=args.token_ids_column,
        loss_on=args.loss_on,
        prompt_length_column=args.prompt_length_column,
        loss_mask_column=args.loss_mask_column,
        disjoint_key_column=args.disjoint_key_column,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        max_samples=args.max_kd_samples,
        shuffle=args.shuffle_kd,
        seed=args.seed + 1,
    )
    assert_disjoint_datasets(reference_dataset, kd_dataset)
    model, tokenizer = load_model(args)
    patterns = parse_module_patterns(args.candidate_modules)
    modules = candidate_linear_modules(model, patterns, args.module_names)
    all_groups = group_modules(modules, args.layer_group_size)
    local_groups = shard_groups(all_groups, rank, world_size, args.layer_shard_strategy)
    LOGGER.info(
        "rank=%d/%d selected %d modules in %d/%d groups on %s",
        rank,
        world_size,
        sum(len(group) for group in local_groups),
        len(local_groups),
        len(all_groups),
        args.device,
    )

    ref_loader = make_trajectory_dataloader(
        reference_dataset,
        batch_size=args.ref_batch_size or args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    kd_loader = make_trajectory_dataloader(
        kd_dataset,
        batch_size=args.kd_batch_size or args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    storage_device = torch.device(args.device if args.factor_storage_device == "cuda" else "cpu")
    rank_payload: dict[str, object] = {"rank": rank, "modules": {}, "group_diagnostics": []}

    for group_index, group in enumerate(local_groups):
        names = list(group)
        LOGGER.info("rank=%d group=%d/%d modules=%s", rank, group_index + 1, len(local_groups), names)
        for name, module in group.items():
            LOGGER.info(
                "%s d_in=%d d_out=%d A=%.2f MiB G=%.2f MiB per dataset",
                name,
                module.in_features,
                module.out_features,
                module.in_features * module.in_features * 4 / 2**20,
                module.out_features * module.out_features * 4 / 2**20,
            )
        ref_fixed_point = compute_group_gradient_diagnostic(
            model, ref_loader, group, max_batches=args.diagnostic_batches
        )
        kd_fixed_point = compute_group_gradient_diagnostic(
            model, kd_loader, group, max_batches=args.diagnostic_batches
        )
        ref_factors, ref_pass = collect_dataset_factors(
            model,
            ref_loader,
            group,
            storage_device=storage_device,
            chunk_size=args.factor_chunk_size,
            dataset_name="reference",
        )
        kd_factors, kd_pass = collect_dataset_factors(
            model,
            kd_loader,
            group,
            storage_device=storage_device,
            chunk_size=args.factor_chunk_size,
            dataset_name="kd",
        )

        for name, module in group.items():
            tensors = factorized_score_tensors(module.weight, ref_factors[name], kd_factors[name], eta=args.eta)
            parameter_name = f"{name}.weight"
            shard_path = save_score_shard(
                output_dir,
                parameter_name,
                tensors,
                save_factor_diagnostics=args.save_factor_diagnostics,
            )
            diagnostics = module_diagnostics(tensors, ref_factors[name], kd_factors[name])
            factor_memory = {
                "A_bytes": int(ref_factors[name].activation.numel() * ref_factors[name].activation.element_size()),
                "G_bytes": int(ref_factors[name].output_gradient.numel() * ref_factors[name].output_gradient.element_size()),
            }
            rank_payload["modules"][parameter_name] = {
                "shard": str(shard_path.relative_to(output_dir)),
                "shape": list(module.weight.shape),
                "d_out": module.out_features,
                "d_in": module.in_features,
                "factor_memory_per_dataset": factor_memory,
                "diagnostics": diagnostics,
            }
            LOGGER.info(
                "%s score mean=%.6e std=%.6e recovery_positive=%.4f shard=%s",
                parameter_name,
                diagnostics["score"]["mean"],
                diagnostics["score"]["std"],
                diagnostics["fraction_recovery_positive"],
                shard_path,
            )
            del tensors

        rank_payload["group_diagnostics"].append(
            {
                "modules": names,
                "reference_fixed_point": ref_fixed_point,
                "kd_fixed_point": kd_fixed_point,
                "reference_pass": ref_pass,
                "kd_pass": kd_pass,
                "peak_cpu_memory_bytes": process_peak_cpu_memory_bytes(),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            }
        )
        write_rank_manifest(output_dir, rank, rank_payload)
        del ref_factors, kd_factors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "method": "layerwise_factorized_recoverability",
        "model_path": args.model_path,
        "ref_dataset": args.ref_dataset_path,
        "kd_dataset": args.kd_dataset_path,
        "eta": args.eta,
        "loss": "causal_sft_nll_token_mean",
        "output_gradient_convention": "loss_sum_gradient_rescaled_from_token_mean_backward",
        "factorization": "G_kron_A",
        "factor_structure": args.factor_structure,
        "cross_layer_curvature": False,
        "candidate_modules": list(patterns),
        "dtype_model": args.dtype,
        "dtype_factor_accumulation": "float32",
        "factor_storage_device": args.factor_storage_device,
        "loss_masking": args.loss_on,
        "max_length": args.max_length,
        "num_ref_examples": len(reference_dataset),
        "num_kd_examples": len(kd_dataset),
        "ref_batch_size": args.ref_batch_size or args.batch_size,
        "kd_batch_size": args.kd_batch_size or args.batch_size,
        "layer_group_size": args.layer_group_size,
        "distributed_layer_sharding": bool(world_size > 1),
        "distributed_world_size": world_size,
        "layer_shard_strategy": args.layer_shard_strategy,
        "attention_implementation": args.attention_implementation,
        "gradient_checkpointing": False,
        "seed": args.seed,
    }
    if not local_groups:
        write_rank_manifest(output_dir, rank, rank_payload)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_manifests(output_dir, metadata, world_size)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
