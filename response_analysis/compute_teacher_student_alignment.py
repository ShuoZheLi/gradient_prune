from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_analysis.io_utils import read_jsonl, write_jsonl
from response_analysis.metrics import selected_logprobs_from_logits
from response_analysis.pruning import apply_score_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure teacher-student alignment by scoring teacher responses under "
            "a Hugging Face causal language model."
        )
    )
    parser.add_argument("--input", required=True, help="Teacher generations JSONL containing prompt and generated_text.")
    parser.add_argument("--model_path", required=True, help="Student Hugging Face causal LM path or id.")
    parser.add_argument("--output_dir", default="outputs/teacher_student_alignment")
    parser.add_argument("--per_example_output", default=None)
    parser.add_argument("--aggregate_output", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--pruning_sparsity", type=float, default=0.0)
    parser.add_argument("--prune_score_dir", default=None, help="Directory containing saved per-module score .pt files plus metadata.json.")
    parser.add_argument("--prune_score_key", default=None, help="Score key inside each .pt file; inferred from metadata for WANDA score dirs.")
    parser.add_argument("--prune_granularity", choices=["rowwise", "layerwise"], default="rowwise")
    parser.add_argument("--prune_ops", default=None, nargs="*", help="Optional prunable op suffixes, e.g. q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.")
    parser.add_argument("--prune_lambda", type=float, default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--vllm_pruned_model_dir", default=None, help="HF checkpoint dir used when vLLM must load a score-pruned model.")
    parser.add_argument("--vllm_pruned_checkpoint_timeout", type=int, default=7200)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for vLLM scoring.")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Fail or skip examples longer than this many tokens.")
    parser.add_argument("--skip_overlength", action="store_true", help="Skip examples exceeding --max_sequence_length.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="Override vLLM max_model_len.")
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="Cap vLLM scheduled tokens per step; useful for prompt_logprobs memory.")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Cap concurrently scheduled vLLM sequences.")
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=True, help="Use vLLM eager execution.")
    parser.add_argument("--seed", type=int, default=42, help="vLLM sampling seed.")
    return parser.parse_args()


def resolve_dtype(name: str) -> str | torch.dtype:
    return {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_vllm_dtype(name: str) -> str:
    return {"auto": "auto", "bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[name]


def has_hf_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or any(path.glob("model*.safetensors"))


def default_vllm_pruned_model_dir(args: argparse.Namespace) -> Path:
    output_path = Path(args.output_dir)
    model_name = Path(args.model_path).name or "model"
    return output_path / f"vllm_pruned_{model_name}_s{args.pruning_sparsity:g}"


def wait_for_checkpoint(path: Path, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if has_hf_checkpoint(path):
            return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for pruned checkpoint: {path}")


def prepare_vllm_model_path(args: argparse.Namespace, tokenizer: Any) -> tuple[str, dict[str, Any]]:
    if not args.prune_score_dir or float(args.pruning_sparsity) <= 0.0:
        return args.model_path, {
            "enabled": False,
            "requested_sparsity": float(args.pruning_sparsity),
            "actual_sparsity": 0.0,
            "num_pruned_weights": 0,
            "num_total_prunable_weights": 0,
        }

    pruned_dir = Path(args.vllm_pruned_model_dir) if args.vllm_pruned_model_dir else default_vllm_pruned_model_dir(args)
    lock_dir = Path(str(pruned_dir) + ".lock")
    if has_hf_checkpoint(pruned_dir):
        return str(pruned_dir), {"enabled": True, "vllm_pruned_model_dir": str(pruned_dir), "checkpoint_reused": True, "requested_sparsity": float(args.pruning_sparsity)}

    acquired = False
    try:
        os.makedirs(lock_dir)
        acquired = True
    except FileExistsError:
        wait_for_checkpoint(pruned_dir, args.vllm_pruned_checkpoint_timeout)
        return str(pruned_dir), {"enabled": True, "vllm_pruned_model_dir": str(pruned_dir), "checkpoint_reused": True, "requested_sparsity": float(args.pruning_sparsity)}

    try:
        if has_hf_checkpoint(pruned_dir):
            return str(pruned_dir), {"enabled": True, "vllm_pruned_model_dir": str(pruned_dir), "checkpoint_reused": True, "requested_sparsity": float(args.pruning_sparsity)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=resolve_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
            device_map=None,
        ).to(args.device)
        pruning_info = apply_score_pruning(
            model,
            score_dir=args.prune_score_dir,
            sparsity=args.pruning_sparsity,
            score_key=args.prune_score_key,
            prune_ops=args.prune_ops,
            granularity=args.prune_granularity,
            lambda_value=args.prune_lambda,
        )
        pruned_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(pruned_dir, safe_serialization=True)
        tokenizer.save_pretrained(pruned_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return str(pruned_dir), {**pruning_info, "vllm_pruned_model_dir": str(pruned_dir), "checkpoint_reused": False}
    finally:
        if acquired:
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass


def finite_exp(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 709.0:
        return float("inf")
    return float(math.exp(value))


def response_token_mask_from_offsets(offsets: Sequence[tuple[int, int]], prompt_char_count: int) -> list[bool]:
    """Return token-position mask for tokens containing generated response text.

    Fast tokenizers return ``(0, 0)`` for special tokens; those are intentionally
    masked out.  A token is counted as response if any non-empty part of its
    character span lies after the prompt/response boundary.  This handles rare
    BPE merges across ``prompt + generated_text`` without dropping generated
    characters from the score.
    """
    mask: list[bool] = []
    for start, end in offsets:
        mask.append(end > start and end > prompt_char_count)
    return mask


def fallback_response_token_mask(tokenizer: Any, prompt: str, full_token_count: int) -> list[bool]:
    prompt_ids = tokenizer(prompt, return_token_type_ids=False, add_special_tokens=False)["input_ids"]
    prompt_token_count = len(prompt_ids)
    return [idx >= prompt_token_count for idx in range(full_token_count)]


def encode_with_response_mask(tokenizer: Any, prompt: str, generated_text: str) -> tuple[list[int], list[bool], str]:
    full_text = prompt + generated_text
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(full_text, return_offsets_mapping=True, return_token_type_ids=False, add_special_tokens=False)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        return input_ids, response_token_mask_from_offsets(offsets, len(prompt)), "offset_mapping"
    encoded = tokenizer(full_text, return_token_type_ids=False, add_special_tokens=False)
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    return input_ids, fallback_response_token_mask(tokenizer, prompt, len(input_ids)), "prompt_token_count"


def metric_values_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float | int]:
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    if target_ids.ndim != 1:
        raise ValueError(f"Expected 1D target_ids, got shape {tuple(target_ids.shape)}")
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(f"Logit/target length mismatch: {logits.shape[0]} vs {target_ids.shape[0]}")
    response_token_count = int(target_ids.numel())
    if response_token_count == 0:
        return {
            "response_token_count": 0,
            "alignment_nll_sum": 0.0,
            "alignment_nll_mean": None,
            "response_logprob_mean": None,
            "response_logprob_sum": 0.0,
            "perplexity": None,
            "teacher_token_top1_count": 0,
            "teacher_token_top1_rate": None,
        }

    token_logprobs = selected_logprobs_from_logits(logits, target_ids)
    logprob_sum = float(token_logprobs.sum().item())
    nll_sum = -logprob_sum
    nll_mean = nll_sum / response_token_count
    top1_count = int((logits.argmax(dim=-1) == target_ids).sum().item())
    return {
        "response_token_count": response_token_count,
        "alignment_nll_sum": nll_sum,
        "alignment_nll_mean": nll_mean,
        "response_logprob_mean": logprob_sum / response_token_count,
        "response_logprob_sum": logprob_sum,
        "perplexity": finite_exp(nll_mean),
        "teacher_token_top1_count": top1_count,
        "teacher_token_top1_rate": top1_count / response_token_count,
    }


def _vllm_logprob_value(value: Any) -> float:
    return float(value.logprob) if hasattr(value, "logprob") else float(value)


def _extract_vllm_token_logprob(prompt_logprob_entry: Any, token_id: int) -> float:
    if prompt_logprob_entry is None:
        raise KeyError(f"Token id {token_id} missing because vLLM prompt_logprobs entry is None")
    if token_id not in prompt_logprob_entry:
        available = list(prompt_logprob_entry.keys())[:10]
        raise KeyError(f"Token id {token_id} missing from vLLM prompt_logprobs; available token ids include {available}")
    return _vllm_logprob_value(prompt_logprob_entry[token_id])


def _vllm_token_is_top1(prompt_logprob_entry: Any, token_id: int) -> bool:
    value = prompt_logprob_entry[token_id]
    rank = getattr(value, "rank", None)
    if rank is not None:
        return int(rank) == 1
    max_token_id = max(prompt_logprob_entry, key=lambda item: _vllm_logprob_value(prompt_logprob_entry[item]))
    return int(max_token_id) == int(token_id)


def metric_values_from_prompt_logprobs(
    prompt_logprobs: Sequence[Any] | None,
    input_ids: Sequence[int],
    label_mask: Sequence[bool],
) -> dict[str, float | int | None]:
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs")
    if len(prompt_logprobs) != len(input_ids):
        raise RuntimeError(f"prompt_logprobs length {len(prompt_logprobs)} != input length {len(input_ids)}")
    if len(label_mask) != len(input_ids):
        raise RuntimeError(f"label_mask length {len(label_mask)} != input length {len(input_ids)}")

    logprob_sum = 0.0
    response_token_count = 0
    top1_count = 0
    for pos, should_score in enumerate(label_mask):
        if not should_score:
            continue
        entry = prompt_logprobs[pos]
        if entry is None:
            continue
        token_id = int(input_ids[pos])
        logprob_sum += _extract_vllm_token_logprob(entry, token_id)
        top1_count += int(_vllm_token_is_top1(entry, token_id))
        response_token_count += 1

    if response_token_count == 0:
        return metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long))
    nll_sum = -logprob_sum
    nll_mean = nll_sum / response_token_count
    return {
        "response_token_count": response_token_count,
        "alignment_nll_sum": nll_sum,
        "alignment_nll_mean": nll_mean,
        "response_logprob_mean": logprob_sum / response_token_count,
        "response_logprob_sum": logprob_sum,
        "perplexity": finite_exp(nll_mean),
        "teacher_token_top1_count": top1_count,
        "teacher_token_top1_rate": top1_count / response_token_count,
    }


def score_record(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    index: int,
    model_path: str,
    model_id: str | None,
    device: str,
    max_sequence_length: int | None = None,
    skip_overlength: bool = False,
) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    generated_text = str(record.get("generated_text", ""))
    input_ids_list, response_position_mask, mask_method = encode_with_response_mask(tokenizer, prompt, generated_text)
    full_token_count = len(input_ids_list)
    response_token_count_before_shift = int(sum(response_position_mask))

    row: dict[str, Any] = {
        "prompt_id": record.get("prompt_id", index),
        "prompt_index": record.get("prompt_index"),
        "sample_id": record.get("sample_id"),
        "model_id": model_id or Path(model_path).name,
        "model_path": model_path,
        "backend": "hf",
        "teacher_model_id": record.get("model_id"),
        "teacher_model_path": record.get("model_path"),
        "correctness": record.get("correctness"),
        "reward": record.get("reward"),
        "response_length": record.get("response_length", len(generated_text)),
        "response_char_count": len(generated_text),
        "full_token_count": full_token_count,
        "response_token_count_before_shift": response_token_count_before_shift,
        "mask_method": mask_method,
        "skipped": False,
        "skip_reason": None,
    }

    if full_token_count < 2 or response_token_count_before_shift == 0:
        row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
        row["skipped"] = True
        row["skip_reason"] = "empty_response_or_too_short"
        return row

    if max_sequence_length is not None and full_token_count > max_sequence_length:
        message = f"sequence_length_{full_token_count}_exceeds_{max_sequence_length}"
        if skip_overlength:
            row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
            row["skipped"] = True
            row["skip_reason"] = message
            return row
        raise ValueError(
            f"Example index {index} has {full_token_count} tokens, exceeding --max_sequence_length "
            f"{max_sequence_length}. Pass --skip_overlength to skip it."
        )

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    target_response_mask = torch.tensor(response_position_mask[1:], dtype=torch.bool, device=device)
    if not bool(target_response_mask.any().item()):
        row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
        row["skipped"] = True
        row["skip_reason"] = "no_shifted_response_tokens"
        del input_ids, target_response_mask
        return row

    with torch.no_grad():
        logits = model(input_ids=input_ids[:, :-1]).logits[0]
    target_ids = input_ids[0, 1:]
    selected_logits = logits[target_response_mask]
    selected_targets = target_ids[target_response_mask]
    row.update(metric_values_from_logits(selected_logits, selected_targets))
    del input_ids, target_response_mask, logits, target_ids, selected_logits, selected_targets
    return row


def prepare_vllm_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    index: int,
    model_path: str,
    model_id: str | None,
    max_sequence_length: int | None = None,
    skip_overlength: bool = False,
) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    generated_text = str(record.get("generated_text", ""))
    input_ids_list, response_position_mask, mask_method = encode_with_response_mask(tokenizer, prompt, generated_text)
    full_token_count = len(input_ids_list)
    response_token_count_before_shift = int(sum(response_position_mask))
    label_mask = [False] + [bool(item) for item in response_position_mask[1:]] if input_ids_list else []

    row: dict[str, Any] = {
        "prompt_id": record.get("prompt_id", index),
        "prompt_index": record.get("prompt_index"),
        "sample_id": record.get("sample_id"),
        "model_id": model_id or Path(model_path).name,
        "model_path": model_path,
        "backend": "vllm",
        "teacher_model_id": record.get("model_id"),
        "teacher_model_path": record.get("model_path"),
        "correctness": record.get("correctness"),
        "reward": record.get("reward"),
        "response_length": record.get("response_length", len(generated_text)),
        "response_char_count": len(generated_text),
        "full_token_count": full_token_count,
        "response_token_count_before_shift": response_token_count_before_shift,
        "mask_method": mask_method,
        "skipped": False,
        "skip_reason": None,
    }

    if full_token_count < 2 or response_token_count_before_shift == 0:
        row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
        row["skipped"] = True
        row["skip_reason"] = "empty_response_or_too_short"
    elif max_sequence_length is not None and full_token_count > max_sequence_length:
        message = f"sequence_length_{full_token_count}_exceeds_{max_sequence_length}"
        if skip_overlength:
            row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
            row["skipped"] = True
            row["skip_reason"] = message
        else:
            raise ValueError(
                f"Example index {index} has {full_token_count} tokens, exceeding --max_sequence_length "
                f"{max_sequence_length}. Pass --skip_overlength to skip it."
            )
    elif not any(label_mask):
        row.update(metric_values_from_logits(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)))
        row["skipped"] = True
        row["skip_reason"] = "no_shifted_response_tokens"

    return {"row": row, "input_ids": input_ids_list, "label_mask": label_mask}


def score_records_vllm(tokenizer: Any, records: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    prepared = [
        prepare_vllm_record(
            tokenizer,
            record,
            index=index,
            model_path=args.model_path,
            model_id=args.model_id,
            max_sequence_length=args.max_sequence_length,
            skip_overlength=args.skip_overlength,
        )
        for index, record in enumerate(records)
    ]
    rows = [item["row"] for item in prepared]
    scorable = [item for item in prepared if not item["row"].get("skipped")]
    if not scorable:
        return rows

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("Install vllm to use --backend vllm") from exc

    max_prompt_len = max(len(item["input_ids"]) for item in scorable)
    max_model_len = args.vllm_max_model_len or args.max_sequence_length or (max_prompt_len + 1)
    if max_prompt_len + 1 > max_model_len:
        raise ValueError(
            f"Longest vLLM prompt has {max_prompt_len} tokens plus 1 generated token, "
            f"but max_model_len is {max_model_len}. Increase --vllm_max_model_len."
        )

    vllm_model_path, pruning_info = prepare_vllm_model_path(args, tokenizer)
    for row in rows:
        row["pruning_sparsity"] = float(args.pruning_sparsity)
        row["prune_score_dir"] = args.prune_score_dir
        row["prune_score_key"] = pruning_info.get("score_key")
        row["prune_granularity"] = args.prune_granularity
        row["pruning_info"] = pruning_info

    llm_kwargs = {
        "model": str(vllm_model_path),
        "tokenizer": str(vllm_model_path),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": resolve_vllm_dtype(args.dtype),
        "trust_remote_code": bool(args.trust_remote_code),
        "enforce_eager": bool(args.enforce_eager),
        "max_model_len": int(max_model_len),
    }
    if args.vllm_max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(args.vllm_max_num_batched_tokens)
    if args.vllm_max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(args.vllm_max_num_seqs)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=int(args.seed))
    batch_size = max(int(args.batch_size), 1)
    for start in tqdm(range(0, len(scorable), batch_size), desc="teacher-student alignment:vllm"):
        batch = scorable[start : start + batch_size]
        prompts = [{"prompt_token_ids": item["input_ids"]} for item in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for item, output in zip(batch, outputs):
            item["row"].update(metric_values_from_prompt_logprobs(output.prompt_logprobs, item["input_ids"], item["label_mask"]))
    return rows


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if not row.get("skipped") and row.get("alignment_nll_mean") is not None]
    nll_values = _numeric_values(scored, "alignment_nll_mean")
    perplexity_values = _numeric_values(scored, "perplexity")
    top1_values = _numeric_values(scored, "teacher_token_top1_rate")
    token_count = int(sum(int(row.get("response_token_count") or 0) for row in scored))
    nll_sum = float(sum(float(row.get("alignment_nll_sum") or 0.0) for row in scored))
    top1_count = int(sum(int(row.get("teacher_token_top1_count") or 0) for row in scored))
    return {
        "num_examples": len(rows),
        "num_scored_examples": len(scored),
        "num_skipped_examples": len(rows) - len(scored),
        "response_token_count": token_count,
        "mean_alignment_nll": statistics.fmean(nll_values) if nll_values else None,
        "median_alignment_nll": statistics.median(nll_values) if nll_values else None,
        "std_alignment_nll": statistics.pstdev(nll_values) if len(nll_values) > 1 else 0.0 if nll_values else None,
        "mean_perplexity": statistics.fmean(perplexity_values) if perplexity_values else None,
        "mean_teacher_token_top1_rate": statistics.fmean(top1_values) if top1_values else None,
        "token_weighted_alignment_nll": nll_sum / token_count if token_count else None,
        "token_weighted_perplexity": finite_exp(nll_sum / token_count) if token_count else None,
        "token_weighted_teacher_token_top1_rate": top1_count / token_count if token_count else None,
    }


def aggregate_alignment(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct_rows = [row for row in rows if row.get("correctness") is True]
    incorrect_rows = [row for row in rows if row.get("correctness") is False]
    return {
        "overall": summarize_rows(rows),
        "correct_responses_only": summarize_rows(correct_rows),
        "incorrect_responses_only": summarize_rows(incorrect_rows),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    per_example_output = Path(args.per_example_output) if args.per_example_output else output_dir / "per_example_alignment.jsonl"
    aggregate_output = Path(args.aggregate_output) if args.aggregate_output else output_dir / "aggregate_alignment.json"

    records = read_jsonl(args.input)
    if args.max_examples >= 0:
        records = records[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.backend == "vllm":
        rows = score_records_vllm(tokenizer, records, args)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=resolve_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
            device_map=None,
        ).to(args.device)
        pruning_info = apply_score_pruning(
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
        for index, record in enumerate(tqdm(records, desc="teacher-student alignment:hf")):
            row = score_record(
                model,
                tokenizer,
                record,
                index=index,
                model_path=args.model_path,
                model_id=args.model_id,
                device=args.device,
                max_sequence_length=args.max_sequence_length,
                skip_overlength=args.skip_overlength,
            )
            row["pruning_sparsity"] = float(args.pruning_sparsity)
            row["prune_score_dir"] = args.prune_score_dir
            row["prune_score_key"] = pruning_info.get("score_key")
            row["prune_granularity"] = args.prune_granularity
            row["pruning_info"] = pruning_info
            rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_jsonl(per_example_output, rows)
    aggregate = aggregate_alignment(rows)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} per-example rows to {per_example_output}")
    print(f"Wrote aggregate metrics to {aggregate_output}")


if __name__ == "__main__":
    main()
