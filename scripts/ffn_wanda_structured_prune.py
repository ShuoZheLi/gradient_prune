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
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration_loaders import load_calibration_examples, make_calibration_dataloader
from layer_utils import find_transformer_layers
from model_utils import load_model_and_tokenizer, temporarily_disable_cache

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Wanda-style FFN channel scores and optionally structurally prune HF SwiGLU MLPs."
    )
    parser.add_argument("--model", required=True, help="Model name or local checkpoint path.")
    parser.add_argument("--calibration", required=True, help="Calibration dataset path/directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where FFN channel scores are written.")
    parser.add_argument("--pruned-model-dir", default=None, help="Directory for the structurally pruned model. Defaults to output-dir/pruned_model.")
    parser.add_argument("--save-pruned-model", action="store_true", help="Structurally prune in memory and save the HF checkpoint.")
    parser.add_argument("--validate-pruned-forward", action="store_true", help="After scoring, structurally prune in memory and run a tiny forward pass without requiring checkpoint save.")
    parser.add_argument("--validation-text", default="hello", help="Text used by --validate-pruned-forward.")
    parser.add_argument("--sparsity", type=float, default=None, help="Fraction of FFN channels to remove from every layer.")
    parser.add_argument("--target-intermediate-size", type=int, default=None, help="Exact number of FFN channels to keep per layer. Overrides --sparsity.")
    parser.add_argument("--round-to-multiple", type=int, default=1, help="Round sparsity-derived target width to this multiple.")
    parser.add_argument("--min-keep", type=int, default=1, help="Minimum number of FFN channels to keep per layer.")
    parser.add_argument("--calibration-type", default="prompt_response", choices=["prompt_response", "text"], help="How to interpret calibration records.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on calibration examples.")
    parser.add_argument("--microbatch-size", type=int, default=1, help="Calibration forward microbatch size.")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum tokenized sequence length.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype: bf16, fp16, fp32, or auto.")
    parser.add_argument("--device", default=None, help="Device for single-process runs. Defaults to cuda:0 when available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle calibration examples before optional max-samples.")
    parser.add_argument("--only-correct", action="store_true", help="Filter calibration rows where is_correct is true when present.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing complete output directory.")
    parser.add_argument("--enable-thinking", choices=("auto", "true", "false"), default="auto", help="Qwen3 chat-template thinking mode for reconstructing prompt/response calibration text.")
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
    LOGGER.info("Initialized distributed FFN Wanda scoring rank %d/%d", dist.get_rank(), dist.get_world_size())
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


def is_main_process() -> bool:
    return distributed_rank() == 0


def output_is_complete(output_dir: Path) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    modules = metadata.get("modules")
    return isinstance(modules, dict) and bool(modules) and all((output_dir / file_name).is_file() for file_name in modules.values())


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    valid = tensor[finite]
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


def get_mlp_triplets(model) -> list[tuple[str, nn.Module, nn.Linear, nn.Linear, nn.Linear]]:
    triplets = []
    for layer_idx, block in enumerate(find_transformer_layers(model)):
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise ValueError(f"Layer {layer_idx} does not have an mlp module")
        try:
            gate_proj = mlp.gate_proj
            up_proj = mlp.up_proj
            down_proj = mlp.down_proj
        except AttributeError as exc:
            raise ValueError(f"Layer {layer_idx} MLP does not expose gate_proj/up_proj/down_proj") from exc
        if not all(isinstance(module, nn.Linear) for module in (gate_proj, up_proj, down_proj)):
            raise TypeError(f"Layer {layer_idx} MLP projections must be nn.Linear modules")
        if gate_proj.out_features != up_proj.out_features or gate_proj.out_features != down_proj.in_features:
            raise ValueError(
                f"Layer {layer_idx} inconsistent FFN width: "
                f"gate={gate_proj.out_features} up={up_proj.out_features} down_in={down_proj.in_features}"
            )
        if gate_proj.in_features != up_proj.in_features or gate_proj.in_features != down_proj.out_features:
            raise ValueError(
                f"Layer {layer_idx} inconsistent hidden size: "
                f"gate_in={gate_proj.in_features} up_in={up_proj.in_features} down_out={down_proj.out_features}"
            )
        triplets.append((f"model.layers.{layer_idx}.mlp", mlp, gate_proj, up_proj, down_proj))
    if not triplets:
        raise ValueError("No transformer MLP layers found")
    return triplets


def infer_original_width(triplets: list[tuple[str, nn.Module, nn.Linear, nn.Linear, nn.Linear]]) -> int:
    widths = {gate_proj.out_features for _, _, gate_proj, _, _ in triplets}
    if len(widths) != 1:
        raise ValueError(f"Expected a uniform FFN width across layers, got {sorted(widths)}")
    return next(iter(widths))


def resolve_target_width(args: argparse.Namespace, original_width: int) -> int:
    if args.target_intermediate_size is not None:
        target_width = int(args.target_intermediate_size)
    else:
        if args.sparsity is None:
            raise ValueError("Specify either --target-intermediate-size or --sparsity")
        if not 0.0 <= args.sparsity < 1.0:
            raise ValueError(f"--sparsity must be in [0, 1), got {args.sparsity}")
        keep = original_width * (1.0 - float(args.sparsity))
        multiple = max(int(args.round_to_multiple), 1)
        target_width = int(round(keep / multiple) * multiple)
    target_width = max(int(args.min_keep), target_width)
    if target_width > original_width:
        raise ValueError(f"Target FFN width {target_width} exceeds original width {original_width}")
    return target_width


def collect_ffn_activation_rms(
    model,
    tokenizer,
    *,
    calibration_path: str,
    calibration_type: str,
    only_correct: bool,
    max_samples: int | None,
    microbatch_size: int,
    max_length: int,
    device: str,
    seed: int,
    shuffle: bool,
    enable_thinking: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int], int]:
    rank = distributed_rank()
    world_size = distributed_world_size()
    examples = load_calibration_examples(
        calibration_path,
        calibration_type=calibration_type,
        only_correct=only_correct,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        enable_thinking=enable_thinking,
        tokenizer=tokenizer,
    )
    total_examples = len(examples)
    if world_size > 1:
        examples = examples[rank::world_size]
        LOGGER.info("Rank %d/%d scoring %d/%d calibration examples", rank, world_size, len(examples), total_examples)
    dataloader = make_calibration_dataloader(
        examples,
        tokenizer,
        max_length=max_length,
        loss_on="full_trajectory",
        microbatch_size=microbatch_size,
    )
    triplets = get_mlp_triplets(model)
    sq_sums = {name: torch.zeros(down_proj.in_features, dtype=torch.float64) for name, _, _, _, down_proj in triplets}
    counts = {name: 0 for name, *_ in triplets}
    current_attention_mask: dict[str, torch.Tensor | None] = {"value": None}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            hidden = inputs[0].detach().float()
            mask = current_attention_mask["value"]
            if hidden.dim() == 3 and mask is not None and tuple(mask.shape) == tuple(hidden.shape[:2]):
                flat = hidden[mask.to(hidden.device, dtype=torch.bool)]
            else:
                flat = hidden.reshape(-1, hidden.shape[-1])
                if hidden.dim() == 2 and mask is not None and mask.numel() == flat.shape[0]:
                    flat = flat[mask.reshape(-1).to(hidden.device, dtype=torch.bool)]
            if flat.numel() == 0:
                return
            sq_sums[name].add_(flat.pow(2).sum(dim=0).cpu().double())
            counts[name] += int(flat.shape[0])
        return hook

    for name, _, _, _, down_proj in triplets:
        handles.append(down_proj.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        with torch.no_grad(), temporarily_disable_cache(model):
            for batch in tqdm(dataloader, desc="FFN activation RMS", disable=not is_main_process()):
                batch = {key: value.to(device) for key, value in batch.items()}
                current_attention_mask["value"] = batch["attention_mask"]
                _forward_without_lm_head(model, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                current_attention_mask["value"] = None
    finally:
        current_attention_mask["value"] = None
        for handle in handles:
            handle.remove()
    reduce_activation_sums(sq_sums, counts)
    activation_rms = {name: (sq_sums[name] / max(counts[name], 1)).sqrt().float() for name in sq_sums}
    return activation_rms, counts, total_examples


def _forward_without_lm_head(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    base_model = getattr(model, "model", None)
    if base_model is not None:
        base_model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        model(input_ids=input_ids, attention_mask=attention_mask)


def reduce_activation_sums(sq_sums: dict[str, torch.Tensor], counts: dict[str, int]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for name in sq_sums:
        dist.all_reduce(sq_sums[name], op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([counts[name]], dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts[name] = int(count_tensor.item())


def compute_scores_and_indices(model, activation_rms: dict[str, torch.Tensor], target_width: int) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for name, _, _, _, down_proj in get_mlp_triplets(model):
        act = activation_rms[name].cpu().float()
        down_norm = down_proj.weight.detach().cpu().float().norm(p=2, dim=0)
        if act.shape != down_norm.shape:
            raise ValueError(f"Shape mismatch for {name}: activation_rms={tuple(act.shape)} down_norm={tuple(down_norm.shape)}")
        score = act * down_norm
        keep_idx = torch.topk(score, k=target_width, largest=True, sorted=True).indices.sort().values.to(torch.long)
        keep_mask = torch.zeros(score.numel(), dtype=torch.bool)
        keep_mask[keep_idx] = True
        prune_idx = torch.arange(score.numel(), dtype=torch.long)[~keep_mask]
        result[name] = {
            "ffn_wanda_channel": score.float(),
            "activation_rms": act.float(),
            "down_proj_col_l2": down_norm.float(),
            "keep_idx": keep_idx,
            "prune_idx": prune_idx,
        }
    return result


def save_scores(score_entries: dict[str, dict[str, torch.Tensor]], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    modules = {}
    summaries = {}
    total_channels = 0
    for name, entry in tqdm(score_entries.items(), desc="Saving FFN scores", unit="layer"):
        file_name = f"{name.replace('.', '__')}.pt"
        torch.save({key: value.cpu() for key, value in entry.items()}, output_dir / file_name)
        modules[name] = file_name
        summaries[name] = tensor_summary(entry["ffn_wanda_channel"])
        total_channels += int(entry["ffn_wanda_channel"].numel())
    full_metadata = {
        **metadata,
        "score_key": "ffn_wanda_channel",
        "definition": "sqrt(mean over non-padding calibration tokens of SwiGLU intermediate h_j(x)^2) * ||W_down[:, j]||_2",
        "modules": modules,
        "summaries": summaries,
        "num_modules": len(modules),
        "num_total_channels": total_channels,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(full_metadata, handle, indent=2, default=str)


def apply_structured_ffn_pruning(model, keep_indices: dict[str, torch.Tensor]) -> None:
    for name, _, gate_proj, up_proj, down_proj in get_mlp_triplets(model):
        if name not in keep_indices:
            raise KeyError(f"Missing keep indices for {name}")
        keep_idx = keep_indices[name].to(torch.long)
        _slice_linear_out_features(gate_proj, keep_idx)
        _slice_linear_out_features(up_proj, keep_idx)
        _slice_linear_in_features(down_proj, keep_idx)
    first_keep = next(iter(keep_indices.values()))
    target_width = int(first_keep.numel())
    if hasattr(model.config, "intermediate_size"):
        model.config.intermediate_size = target_width
    if hasattr(model.config, "ffn_hidden_size"):
        model.config.ffn_hidden_size = target_width


def _slice_linear_out_features(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    device_idx = keep_idx.to(linear.weight.device)
    old_requires_grad = linear.weight.requires_grad
    new_weight = linear.weight.detach().index_select(0, device_idx).contiguous()
    linear.weight = nn.Parameter(new_weight, requires_grad=old_requires_grad)
    if linear.bias is not None:
        old_bias_requires_grad = linear.bias.requires_grad
        new_bias = linear.bias.detach().index_select(0, device_idx).contiguous()
        linear.bias = nn.Parameter(new_bias, requires_grad=old_bias_requires_grad)
    linear.out_features = int(keep_idx.numel())


def _slice_linear_in_features(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    device_idx = keep_idx.to(linear.weight.device)
    old_requires_grad = linear.weight.requires_grad
    new_weight = linear.weight.detach().index_select(1, device_idx).contiguous()
    linear.weight = nn.Parameter(new_weight, requires_grad=old_requires_grad)
    linear.in_features = int(keep_idx.numel())


def save_structurally_pruned_model(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_use_cache = getattr(model.config, "use_cache", None)
    old_torch_dtype = getattr(model.config, "torch_dtype", None)
    old_dtype = getattr(model.config, "dtype", None)
    if old_use_cache is not None:
        model.config.use_cache = True
    if old_torch_dtype is None and old_dtype is not None:
        model.config.torch_dtype = old_dtype
    try:
        model.save_pretrained(output_dir)
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if old_torch_dtype is None and hasattr(model.config, "torch_dtype"):
            model.config.torch_dtype = old_torch_dtype
    tokenizer.save_pretrained(output_dir)


def validate_pruned_model_forward(model, tokenizer, *, device: str, target_width: int, validation_text: str) -> dict[str, Any]:
    for name, _, gate_proj, up_proj, down_proj in get_mlp_triplets(model):
        if gate_proj.out_features != target_width:
            raise ValueError(f"{name}.gate_proj.out_features={gate_proj.out_features}, expected {target_width}")
        if up_proj.out_features != target_width:
            raise ValueError(f"{name}.up_proj.out_features={up_proj.out_features}, expected {target_width}")
        if down_proj.in_features != target_width:
            raise ValueError(f"{name}.down_proj.in_features={down_proj.in_features}, expected {target_width}")
    inputs = tokenizer(validation_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad(), temporarily_disable_cache(model):
        outputs = model(**inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Validated forward pass did not return logits")
    return {
        "validation_text": validation_text,
        "logits_shape": list(logits.shape),
        "target_intermediate_size": target_width,
        "num_layers": len(get_mlp_triplets(model)),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    pruned_model_dir = Path(args.pruned_model_dir) if args.pruned_model_dir else output_dir / "pruned_model"
    if output_is_complete(output_dir) and not args.overwrite:
        if not args.validate_pruned_forward and (not args.save_pruned_model or (pruned_model_dir / "config.json").is_file()):
            LOGGER.info("Output already appears complete; use --overwrite to recompute: %s", output_dir)
            return 0
        LOGGER.info("Scores are complete but recomputing for requested prune validation/save: %s", output_dir)

    distributed = init_distributed_from_torchrun()
    if distributed and torch.cuda.is_available():
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    rank = distributed_rank()
    world_size = distributed_world_size()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    LOGGER.info("Rank %d/%d loading model on %s", rank, world_size, args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device, args.trust_remote_code)
    triplets = get_mlp_triplets(model)
    original_width = infer_original_width(triplets)
    target_width = resolve_target_width(args, original_width)
    effective_sparsity = 1.0 - (target_width / original_width)
    LOGGER.info("FFN structural target: original_width=%d target_width=%d effective_sparsity=%.6f", original_width, target_width, effective_sparsity)

    activation_rms, counts, total_examples = collect_ffn_activation_rms(
        model,
        tokenizer,
        calibration_path=args.calibration,
        calibration_type=args.calibration_type,
        only_correct=args.only_correct,
        max_samples=args.max_samples,
        microbatch_size=args.microbatch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
        shuffle=args.shuffle,
        enable_thinking=args.enable_thinking,
    )

    if rank == 0:
        score_entries = compute_scores_and_indices(model, activation_rms, target_width)
        save_scores(
            score_entries,
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
                "seed": args.seed,
                "shuffle": args.shuffle,
                "enable_thinking": args.enable_thinking,
                "distributed_world_size": world_size,
                "number_of_examples": total_examples,
                "activation_token_counts": counts,
                "original_intermediate_size": original_width,
                "target_intermediate_size": target_width,
                "requested_sparsity": args.sparsity,
                "effective_sparsity": effective_sparsity,
                "round_to_multiple": args.round_to_multiple,
            },
        )
        LOGGER.info("FFN Wanda channel scores saved to %s", output_dir)
        if args.save_pruned_model or args.validate_pruned_forward:
            keep_indices = {name: entry["keep_idx"] for name, entry in score_entries.items()}
            apply_structured_ffn_pruning(model, keep_indices)
        if args.validate_pruned_forward:
            validation = validate_pruned_model_forward(
                model,
                tokenizer,
                device=args.device,
                target_width=target_width,
                validation_text=args.validation_text,
            )
            with open(output_dir / "pruned_forward_validation.json", "w", encoding="utf-8") as handle:
                json.dump(validation, handle, indent=2, default=str)
            LOGGER.info("Pruned forward validation passed: %s", validation)
        if args.save_pruned_model:
            save_structurally_pruned_model(model, tokenizer, pruned_model_dir)
            LOGGER.info("Structurally pruned model saved to %s", pruned_model_dir)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("FFN Wanda structural pruning failed")
        cleanup_distributed()
        raise
