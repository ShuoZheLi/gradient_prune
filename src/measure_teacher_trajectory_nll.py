#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_EXCLUDE_KEYWORDS = ("embed", "lm_head", "norm", "layernorm", "rmsnorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure response-only teacher-trajectory NLL under a student causal LM."
    )
    parser.add_argument("--model", required=True, help="Student model path or Hugging Face id.")
    parser.add_argument("--data", required=True, help="Input parquet/jsonl with prompt and teacher response columns.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics.json and per_example_metrics files.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--text-key", default="prompt_generated_trajectory")
    parser.add_argument("--example-id-key", default="example_id")
    parser.add_argument("--response-index-key", default="response_index")
    parser.add_argument("--only-correct", action="store_true", help="Keep only rows with is_correct == True when present.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max rows after filtering; <=0 means all rows.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=14336)
    parser.add_argument(
        "--truncation",
        choices=("left", "right", "error"),
        default="right",
        help=(
            "right keeps the start of prompt+response and drops tokens from the end; "
            "left keeps the end and drops tokens from the start; error rejects overlength examples."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--mask-path", default=None, help="Sparse-update .pt mask from verl/tools/build_sparse_update_mask.py.")
    parser.add_argument(
        "--apply-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --mask-path is set, zero entries where mask is False before measuring NLL.",
    )
    parser.add_argument("--strict-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-per-example-parquet", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input extension for {path}; expected parquet/pq/jsonl")


def _first_present(row: pd.Series, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key and key in row.index:
            value = row[key]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def _normalize_prompt_response(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    prompt = _first_present(row, [args.prompt_key, "prompt", "query", "question"], "")
    response = _first_present(row, [args.response_key, "response", "answer", "solution"], None)
    if response is None:
        text = _first_present(row, [args.text_key, "prompt_generated_trajectory", "trajectory", "text"], "")
        text = str(text)
        prompt = str(prompt)
        if prompt and text.startswith(prompt):
            response = text[len(prompt) :]
        else:
            raise ValueError(
                "Could not infer response: response column missing and text does not start with prompt. "
                f"Available columns: {list(row.index)}"
            )
    return str(prompt), str(response)


def prepare_rows(df: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.only_correct and "is_correct" in df.columns:
        df = df[df["is_correct"] == True]
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples)
    rows: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(df.iterrows()):
        prompt, response = _normalize_prompt_response(row, args)
        rows.append(
            {
                "row_position": row_position,
                "example_id": _first_present(row, [args.example_id_key], row_position),
                "response_index": _first_present(row, [args.response_index_key], None),
                "prompt": prompt,
                "response": response,
                "is_correct": _first_present(row, ["is_correct"], None),
                "task_score": _first_present(row, ["task_score"], None),
            }
        )
    return rows


def torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_sparse_masks(mask_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "masks" not in payload:
        raise ValueError(f"Invalid sparse-update mask file {mask_path}: expected payload with a 'masks' field")
    masks = {str(name): tensor.detach().cpu().bool() for name, tensor in payload["masks"].items()}
    metadata = dict(payload.get("metadata", {}))
    return masks, metadata


def name_aliases(name: str) -> list[str]:
    canonical = name.replace("_fsdp_wrapped_module.", "")
    aliases = [canonical]
    if canonical.startswith("model."):
        aliases.append(canonical[len("model.") :])
    else:
        aliases.append(f"model.{canonical}")
    return list(dict.fromkeys(aliases))


def should_mask_param_name(name: str) -> bool:
    lower_name = name.lower()
    if any(keyword in lower_name for keyword in DEFAULT_EXCLUDE_KEYWORDS):
        return False
    if not name.endswith(".weight"):
        return False
    return any(f".{module}." in f".{name}" for module in DEFAULT_TARGET_MODULES)


def apply_sparse_mask(model: torch.nn.Module, mask_path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    masks, metadata = load_sparse_masks(mask_path)
    applied: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    model_param_names = {name for name, _ in model.named_parameters()}
    seen_mask_names: set[str] = set()
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            mask_name = next((alias for alias in name_aliases(param_name) if alias in masks), None)
            if mask_name is None:
                continue
            mask = masks[mask_name]
            seen_mask_names.add(mask_name)
            if tuple(mask.shape) != tuple(param.shape):
                shape_mismatches.append(
                    {"param": param_name, "mask": mask_name, "param_shape": list(param.shape), "mask_shape": list(mask.shape)}
                )
                continue
            param.mul_(mask.to(device=param.device, dtype=param.dtype))
            applied.append(param_name)
    unmatched_masks = sorted(name for name in masks if name not in seen_mask_names and not any(alias in model_param_names for alias in name_aliases(name)))
    uncovered_target_params = sorted(
        param_name
        for param_name, param in model.named_parameters()
        if param.ndim > 1 and should_mask_param_name(param_name) and not any(alias in seen_mask_names for alias in name_aliases(param_name))
    )
    if strict and shape_mismatches:
        raise ValueError(f"Sparse mask shape mismatches: {shape_mismatches[:10]}")
    if strict and unmatched_masks:
        raise ValueError(f"Sparse masks with no matching model parameter: {unmatched_masks[:10]}")
    if strict and uncovered_target_params:
        raise ValueError(f"Sparse mask is missing target model parameters: {uncovered_target_params[:10]}")
    return {
        "mask_path": str(mask_path),
        "mask_metadata": metadata,
        "num_mask_tensors": len(masks),
        "num_applied_tensors": len(applied),
        "num_shape_mismatches": len(shape_mismatches),
        "num_unmatched_masks": len(unmatched_masks),
        "num_uncovered_target_params": len(uncovered_target_params),
        "shape_mismatches_preview": shape_mismatches[:10],
        "unmatched_masks_preview": unmatched_masks[:10],
        "uncovered_target_params_preview": uncovered_target_params[:10],
    }


def build_features(rows: list[dict[str, Any]], tokenizer, max_length: int, truncation: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for item in rows:
        prompt_ids = tokenizer(item["prompt"], add_special_tokens=False).input_ids
        response_ids = tokenizer(item["response"], add_special_tokens=False).input_ids
        if not response_ids:
            raise ValueError(f"Empty response after tokenization at row_position={item['row_position']}")
        full_ids = prompt_ids + response_ids
        prompt_len_after_trunc = len(prompt_ids)
        truncated_left_tokens = 0
        truncated_right_tokens = 0
        if len(full_ids) > max_length:
            if truncation == "error":
                raise ValueError(
                    f"Example row_position={item['row_position']} has {len(full_ids)} tokens > max_length={max_length}"
                )
            if truncation == "left":
                truncated_left_tokens = len(full_ids) - max_length
                full_ids = full_ids[-max_length:]
                prompt_len_after_trunc = max(0, len(prompt_ids) - truncated_left_tokens)
            elif truncation == "right":
                truncated_right_tokens = len(full_ids) - max_length
                full_ids = full_ids[:max_length]
                prompt_len_after_trunc = min(len(prompt_ids), len(full_ids))
            else:
                raise ValueError(f"Unsupported truncation mode: {truncation}")
        labels = [IGNORE_INDEX] * prompt_len_after_trunc + full_ids[prompt_len_after_trunc:]
        response_token_count = sum(label != IGNORE_INDEX for label in labels)
        if response_token_count <= 0:
            raise ValueError(
                f"Example row_position={item['row_position']} lost all response tokens after truncation; "
                f"increase --max-length or use a different --truncation mode."
            )
        features.append(
            {
                **item,
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "prompt_tokens": len(prompt_ids),
                "response_tokens_raw": len(response_ids),
                "response_tokens_scored": response_token_count,
                "total_tokens_raw": len(prompt_ids) + len(response_ids),
                "total_tokens_scored_input": len(full_ids),
                "truncated_left_tokens": truncated_left_tokens,
                "truncated_right_tokens": truncated_right_tokens,
            }
        )
    return features


def collate(features: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(int(item["input_ids"].numel()) for item in features)
    input_ids, labels, attention_mask = [], [], []
    for item in features:
        ids = item["input_ids"]
        labs = item["labels"]
        pad = max_len - ids.numel()
        input_ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), ids]))
        labels.append(torch.cat([torch.full((pad,), IGNORE_INDEX, dtype=torch.long), labs]))
        attention_mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(ids.numel(), dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}


def batch_iter(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate(model, tokenizer, features: list[dict[str, Any]], *, device: str, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    total_nll_sum = 0.0
    total_tokens = 0
    results: list[dict[str, Any]] = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch_features in tqdm(list(batch_iter(features, batch_size)), desc="teacher-trajectory NLL"):
            batch = collate(batch_features, tokenizer)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            flat_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = flat_losses.view(shift_labels.shape)
            token_mask = shift_labels.ne(IGNORE_INDEX)
            per_example_nll_sum = (token_losses * token_mask).sum(dim=1)
            per_example_tokens = token_mask.sum(dim=1)
            for item, nll_sum_tensor, token_count_tensor in zip(batch_features, per_example_nll_sum, per_example_tokens):
                token_count = int(token_count_tensor.item())
                nll_sum = float(nll_sum_tensor.item())
                nll = nll_sum / token_count
                total_nll_sum += nll_sum
                total_tokens += token_count
                results.append(
                    {
                        "row_position": item["row_position"],
                        "example_id": item["example_id"],
                        "response_index": item["response_index"],
                        "nll": nll,
                        "nll_sum": nll_sum,
                        "response_tokens": token_count,
                        "prompt_tokens": item["prompt_tokens"],
                        "response_tokens_raw": item["response_tokens_raw"],
                        "total_tokens_raw": item["total_tokens_raw"],
                        "total_tokens_scored_input": item["total_tokens_scored_input"],
                        "truncated_left_tokens": item["truncated_left_tokens"],
                        "truncated_right_tokens": item["truncated_right_tokens"],
                        "perplexity": math.exp(nll) if nll < 50 else float("inf"),
                        "is_correct": item["is_correct"],
                        "task_score": item["task_score"],
                    }
                )
    aggregate_nll = total_nll_sum / total_tokens if total_tokens else float("nan")
    metrics = {
        "nll": aggregate_nll,
        "perplexity": math.exp(aggregate_nll) if aggregate_nll < 50 else float("inf"),
        "nll_sum": total_nll_sum,
        "num_response_tokens": total_tokens,
        "num_examples": len(results),
        "mean_example_nll": sum(item["nll"] for item in results) / len(results) if results else float("nan"),
    }
    return results, metrics


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype(args.dtype), "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device == "cpu":
        model_kwargs["device_map"] = None
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)

    pruning_info: dict[str, Any] | None = None
    if args.mask_path and args.apply_pruning:
        pruning_info = apply_sparse_mask(model, args.mask_path, strict=args.strict_mask)

    df = read_table(args.data)
    rows = prepare_rows(df, args)
    features = build_features(rows, tokenizer, args.max_length, args.truncation)
    results, metrics = evaluate(model, tokenizer, features, device=device, batch_size=args.batch_size)
    metrics.update(
        {
            "model": args.model,
            "data": args.data,
            "prompt_key": args.prompt_key,
            "response_key": args.response_key,
            "only_correct": args.only_correct,
            "max_samples": args.max_samples,
            "max_length": args.max_length,
            "truncation": args.truncation,
            "batch_size": args.batch_size,
            "device": device,
            "dtype": args.dtype,
            "pruning": pruning_info,
        }
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default) + "\n", encoding="utf-8")
    jsonl_path = output_dir / "per_example_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, default=json_default) + "\n")
    if args.save_per_example_parquet:
        pd.DataFrame(results).to_parquet(output_dir / "per_example_metrics.parquet", index=False)
    print(json.dumps(metrics, indent=2, default=json_default))
    print(f"metrics_path={metrics_path}")
    print(f"per_example_jsonl={jsonl_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
