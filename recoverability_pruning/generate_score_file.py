from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import assert_disjoint_datasets, load_trajectory_dataset, make_trajectory_dataloader
from .params import DEFAULT_CANDIDATE_MODULES, build_parameter_space, parameter_numel, parse_module_patterns
from .scoring import (
    compute_dense_pruning_scores,
    compute_distributed_dense_pruning_scores,
    run_single_probe_smoke_test,
)


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recoverability-aware unstructured pruning scores")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ref_dataset_path", required=True)
    parser.add_argument("--kd_dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_probes", type=int, default=16)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument("--probe_lr_eta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ref_batch_size", type=int)
    parser.add_argument("--kd_batch_size", type=int)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_ref_samples", type=int)
    parser.add_argument("--max_kd_samples", type=int)
    parser.add_argument("--candidate_modules", nargs="+", default=list(DEFAULT_CANDIDATE_MODULES))
    parser.add_argument("--hvp_parameter_scope", choices=["transformer", "candidates", "all"], default="all")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0", help="Single device, or 'auto'/'balanced' for HF device mapping")
    parser.add_argument("--max_memory", nargs="*", help="Optional device_map limits such as 0=70GiB 1=70GiB cpu=200GiB")
    parser.add_argument("--loss_on", choices=["full_trajectory", "response_only", "loss_mask"], required=True)
    parser.add_argument("--token_ids_column", default="prompt_generated_trajectory_ids")
    parser.add_argument("--prompt_length_column", default="prompt_length")
    parser.add_argument("--loss_mask_column", default="loss_mask")
    parser.add_argument("--disjoint_key_column", default="prompt")
    parser.add_argument("--truncation_side", choices=["left", "right"], default="right")
    parser.add_argument("--shuffle_ref", action="store_true")
    parser.add_argument("--shuffle_kd", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--activation_offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--activation_offload_pin_memory", action="store_true")
    parser.add_argument("--save_intermediate_stats", action="store_true")
    parser.add_argument("--convergence_checkpoints", default="", help="Comma-separated cumulative probe counts")
    parser.add_argument("--smoke_test", action="store_true", help="Run exactly one shared ref/KD HVP and save diagnostics only")
    parser.add_argument("--distributed_probe_parallel", action="store_true")
    parser.add_argument("--distributed_state_dir")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _parse_max_memory(values: list[str] | None) -> dict[int | str, str] | None:
    if not values:
        return None
    parsed: dict[int | str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --max_memory value {item!r}; expected DEVICE=LIMIT")
        device, limit = item.split("=", 1)
        key: int | str = int(device) if device.isdigit() else device
        parsed[key] = limit
    return parsed


def _parse_checkpoints(raw: str, num_probes: int) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    checkpoints = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    invalid = [item for item in checkpoints if item < 2 or item > num_probes]
    if invalid:
        raise ValueError(f"Convergence checkpoints must be between 2 and num_probes={num_probes}: {invalid}")
    return tuple(checkpoints)


def _load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": _resolve_dtype(args.dtype),
        "attn_implementation": "eager",
    }
    if args.device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model_kwargs["device_map"] = args.device
        max_memory = _parse_max_memory(args.max_memory)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(torch.device(args.device))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError("Model does not support gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model.train()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        model.eval()
    else:
        model.eval()
    return model, tokenizer


def _initialize_distributed(args: argparse.Namespace) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        if args.distributed_probe_parallel:
            LOGGER.warning("--distributed_probe_parallel requested with WORLD_SIZE=1; using sequential scoring")
        return rank, world_size
    if not args.distributed_probe_parallel:
        raise ValueError(
            f"WORLD_SIZE={world_size} requires --distributed_probe_parallel to avoid duplicate score writers"
        )
    if args.smoke_test:
        raise ValueError("Run --smoke_test as a single process before the distributed full job")
    if args.device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        raise ValueError("Distributed probe parallelism requires one complete model per rank; use --device cuda:0")
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    LOGGER.info("Initialized distributed probe parallelism rank=%d world_size=%d device=%s", rank, world_size, args.device)
    return rank, world_size


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    rank, world_size = _initialize_distributed(args)
    if Path(args.ref_dataset_path).resolve() == Path(args.kd_dataset_path).resolve():
        raise ValueError("Reference and KD dataset paths must differ")
    if not args.smoke_test and args.num_probes < 2:
        raise ValueError("num_probes must be at least 2; use --smoke_test for the required M=1 validation")
    random.seed(args.probe_seed)
    np.random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.probe_seed)

    LOGGER.info("Loading trajectory datasets with explicit loss_on=%s", args.loss_on)
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
        seed=args.probe_seed,
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
        seed=args.probe_seed + 1,
    )
    assert_disjoint_datasets(reference_dataset, kd_dataset)

    LOGGER.info("Loading dense checkpoint with eager attention for double backward")
    model, tokenizer = _load_model(args)
    candidate_patterns = parse_module_patterns(args.candidate_modules)
    parameter_space = build_parameter_space(
        model,
        candidate_modules=candidate_patterns,
        hvp_parameter_scope=args.hvp_parameter_scope,
    )
    LOGGER.info(
        "HVP parameters=%d tensors/%d elements; candidates=%d tensors/%d elements",
        len(parameter_space.hvp_parameters),
        parameter_numel(parameter_space.hvp_parameters),
        len(parameter_space.candidate_parameters),
        parameter_numel(parameter_space.candidate_parameters),
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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": args.model_path,
        "ref_dataset_path": args.ref_dataset_path,
        "kd_dataset_path": args.kd_dataset_path,
        "loss_on": args.loss_on,
        "token_ids_column": args.token_ids_column,
        "prompt_length_column": args.prompt_length_column if args.loss_on == "response_only" else None,
        "loss_mask_column": args.loss_mask_column if args.loss_on == "loss_mask" else None,
        "disjoint_key_column": args.disjoint_key_column,
        "truncation_side": args.truncation_side,
        "max_length": args.max_length,
        "num_ref_samples": len(reference_dataset),
        "num_kd_samples": len(kd_dataset),
        "ref_batch_size": args.ref_batch_size or args.batch_size,
        "kd_batch_size": args.kd_batch_size or args.batch_size,
        "candidate_modules": list(candidate_patterns),
        "candidate_parameter_names": list(parameter_space.candidate_names),
        "hvp_parameter_scope": args.hvp_parameter_scope,
        "dtype": args.dtype,
        "device": args.device,
        "attention_implementation": "eager",
        "gradient_checkpointing": args.gradient_checkpointing,
        "activation_offload": args.activation_offload,
        "activation_offload_pin_memory": args.activation_offload_pin_memory,
        "datasets_disjoint_by_exact_token_hash": True,
        "datasets_disjoint_by_explicit_key_when_available": True,
        "distributed_probe_parallel": bool(args.distributed_probe_parallel and world_size > 1),
        "distributed_rank": rank,
        "distributed_world_size": world_size,
    }
    if args.smoke_test:
        diagnostics = run_single_probe_smoke_test(
            model,
            ref_loader,
            kd_loader,
            parameter_space,
            probe_seed=args.probe_seed,
            activation_offload=args.activation_offload,
            activation_offload_pin_memory=args.activation_offload_pin_memory,
        )
        smoke_path = output_path.with_suffix(output_path.suffix + ".smoke.json")
        smoke_path.write_text(json.dumps({"metadata": metadata, "diagnostics": diagnostics}, indent=2), encoding="utf-8")
        LOGGER.info("M=1 reference+KD HVP smoke test passed: %s", smoke_path)
        return 0

    convergence_checkpoints = _parse_checkpoints(args.convergence_checkpoints, args.num_probes)
    if world_size > 1:
        compute_distributed_dense_pruning_scores(
            model,
            ref_loader,
            kd_loader,
            parameter_space,
            num_probes=args.num_probes,
            probe_seed=args.probe_seed,
            eta=args.probe_lr_eta,
            output_path=output_path,
            metadata=metadata,
            save_intermediate_stats=args.save_intermediate_stats,
            convergence_checkpoints=convergence_checkpoints,
            rank=rank,
            world_size=world_size,
            distributed_state_dir=args.distributed_state_dir,
            activation_offload=args.activation_offload,
            activation_offload_pin_memory=args.activation_offload_pin_memory,
        )
        dist.destroy_process_group()
    else:
        compute_dense_pruning_scores(
            model,
            ref_loader,
            kd_loader,
            parameter_space,
            num_probes=args.num_probes,
            probe_seed=args.probe_seed,
            eta=args.probe_lr_eta,
            output_path=output_path,
            metadata=metadata,
            save_intermediate_stats=args.save_intermediate_stats,
            convergence_checkpoints=convergence_checkpoints,
            activation_offload=args.activation_offload,
            activation_offload_pin_memory=args.activation_offload_pin_memory,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
