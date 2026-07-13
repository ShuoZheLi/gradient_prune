from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_analysis.io_utils import read_jsonl
from response_analysis.metrics import selected_logprobs_from_logits, token_entropy_from_logits, top1_stats_from_logits
from response_analysis.pruning import apply_score_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute on-policy or fixed-prefix token entropy metrics.")
    parser.add_argument("--input", default="outputs/generations.jsonl")
    parser.add_argument("--output", default="outputs/token_metrics.parquet")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--pruning_sparsity", type=float, default=0.0)
    parser.add_argument("--prune_score_dir", default=None, help="Directory containing saved per-module score .pt files plus metadata.json.")
    parser.add_argument("--prune_score_key", default=None, help="Score key inside each .pt file; inferred from metadata for WANDA score dirs.")
    parser.add_argument("--prune_granularity", choices=["rowwise", "layerwise"], default="rowwise")
    parser.add_argument("--prune_ops", default=None, nargs="*")
    parser.add_argument("--prune_lambda", type=float, default=None)
    parser.add_argument("--mode", choices=["on_policy", "fixed_prefix"], default="on_policy")
    parser.add_argument("--prefix_bank", default=None, help="JSONL from build_fixed_prefix_bank.py for fixed-prefix mode.")
    parser.add_argument("--max_prefix_records", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> str | torch.dtype:
    return {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def iter_sequences(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.mode == "fixed_prefix":
        if not args.prefix_bank:
            raise ValueError("--prefix_bank is required for fixed-prefix mode")
        records = read_jsonl(args.prefix_bank)
        if args.max_prefix_records >= 0:
            records = records[: args.max_prefix_records]
        return records
    return read_jsonl(args.input)


def token_metrics_for_sequence(model, tokenizer, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    prompt = str(record.get("prompt", ""))
    token_ids = record.get("generated_token_ids") if args.mode == "on_policy" else record.get("prefix_token_ids")
    if not token_ids:
        return None

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
        return_token_type_ids=False,
    )
    prompt_ids = prompt_inputs["input_ids"][0].tolist()
    full_ids = prompt_ids + [int(token_id) for token_id in token_ids]
    if len(full_ids) < 2:
        return None
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=args.device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=args.device)
    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, start:, :]
    target = target_ids[0, start:]
    if logits.shape[0] != target.shape[0]:
        raise RuntimeError(f"Logit/target length mismatch: {logits.shape[0]} vs {target.shape[0]}")

    entropy = token_entropy_from_logits(logits)
    top1_prob, top1_margin = top1_stats_from_logits(logits)
    token_logprobs = selected_logprobs_from_logits(logits, target)
    length = int(entropy.numel())
    base = {
        "model_id": args.model_id or record.get("model_id") or Path(args.model_path).name,
        "model_path": args.model_path,
        "entropy_mode": args.mode,
        "prompt_id": record.get("prompt_id"),
        "sample_id": record.get("sample_id", record.get("prefix_id")),
        "decoding_seed": record.get("decoding_seed"),
        "decoding_config": json.dumps(record.get("decoding_config", {}), sort_keys=True),
        "pruning_sparsity": args.pruning_sparsity if args.prune_score_dir else record.get("pruning_sparsity"),
        "prune_score_dir": args.prune_score_dir,
        "prune_score_key": getattr(args, "_pruning_info", {}).get("score_key"),
        "prune_granularity": args.prune_granularity,
        "correctness": record.get("correctness"),
        "response_length": length,
        "token_entropy_mean": float(entropy.mean().item()) if length else 0.0,
        "token_entropy_sum": float(entropy.sum().item()),
        "top1_probability_mean": float(top1_prob.mean().item()) if length else 0.0,
        "top1_logit_margin_mean": float(top1_margin.mean().item()) if length else 0.0,
        "token_logprob_mean": float(token_logprobs.mean().item()) if length else 0.0,
        "token_logprob_sum": float(token_logprobs.sum().item()),
    }
    del input_ids, target_ids, logits, entropy, top1_prob, top1_margin, token_logprobs
    return base


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(args.device)
    args._pruning_info = apply_score_pruning(
        model,
        score_dir=args.prune_score_dir,
        sparsity=args.pruning_sparsity,
        score_key=args.prune_score_key,
        prune_ops=args.prune_ops,
        granularity=args.prune_granularity,
        lambda_value=args.prune_lambda,
    )
    model.eval()

    rows = []
    for record in tqdm(iter_sequences(args), desc=f"entropy:{args.mode}"):
        row = token_metrics_for_sequence(model, tokenizer, record, args)
        if row is not None:
            rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


if __name__ == "__main__":
    main()
