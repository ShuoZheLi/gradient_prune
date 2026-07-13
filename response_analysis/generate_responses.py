from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CREATE_DIR = REPO_ROOT / "create_calibration_dataset"
for path in (SRC_DIR, CREATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_accuracy_test import load_examples  # noqa: E402
from task_scoring import extract_math_answer, normalize_enable_thinking, scalarize_score, score_task_response  # noqa: E402

from response_analysis.io_utils import write_jsonl
from response_analysis.pruning import apply_score_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K responses per prompt and store reproducible metadata.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--pruning_sparsity", type=float, default=0.0)
    parser.add_argument("--prune_score_dir", default=None, help="Directory containing saved per-module score .pt files plus metadata.json.")
    parser.add_argument("--prune_score_key", default=None, help="Score key inside each .pt file; inferred from metadata for WANDA score dirs.")
    parser.add_argument("--prune_granularity", choices=["rowwise", "layerwise"], default="rowwise")
    parser.add_argument("--prune_ops", default=None, nargs="*", help="Optional prunable op suffixes, e.g. q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.")
    parser.add_argument("--prune_lambda", type=float, default=None)
    parser.add_argument("--generation_backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--vllm_pruned_model_dir", default=None, help="HF checkpoint dir used when vLLM must load a score-pruned model.")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_pruned_checkpoint_timeout", type=int, default=7200)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output", default="outputs/generations.jsonl")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--debug_subset", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--enable_thinking", default="true", help="Passed through repo prompt normalization; use true/false/auto.")
    return parser.parse_args()


def resolve_dtype(name: str) -> str | torch.dtype:
    return {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_vllm_dtype(name: str) -> str:
    return {"auto": "auto", "bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[name]


def sample_seed(base_seed: int, prompt_index: int, sample_id: int) -> int:
    return int(base_seed + prompt_index * 1000003 + sample_id)


def generation_kwargs(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "use_cache": bool(args.use_cache),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.top_k and args.top_k > 0:
        kwargs["top_k"] = args.top_k
    return {key: value for key, value in kwargs.items() if value is not None}


def vllm_sampling_kwargs(args: argparse.Namespace, *, seed: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": float(args.temperature) if float(args.temperature) > 0 else 0.0,
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_new_tokens),
        "n": 1,
        "seed": int(seed),
    }
    if int(args.top_k) > 0:
        kwargs["top_k"] = int(args.top_k)
    return kwargs


def has_hf_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or any(path.glob("model*.safetensors"))


def default_vllm_pruned_model_dir(args: argparse.Namespace) -> Path:
    output_path = Path(args.output)
    model_name = Path(args.model_path).name or "model"
    return output_path.parent / f"vllm_pruned_{model_name}_s{args.pruning_sparsity:g}"


def wait_for_checkpoint(path: Path, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if has_hf_checkpoint(path):
            return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for pruned checkpoint: {path}")


def prepare_vllm_model_path(args: argparse.Namespace, tokenizer) -> tuple[str, dict[str, Any]]:
    """Return a model path that vLLM can load, saving pruned HF weights if needed."""
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


def score_response(data_source: str, response: str, ground_truth: Any) -> float:
    try:
        score_result = score_task_response(response, ground_truth, data_source=data_source)
        score = score_result[0] if isinstance(score_result, tuple) else score_result
    except Exception:
        score = 0.0
    return scalarize_score(score)


def make_record(
    *,
    args: argparse.Namespace,
    model_id: str,
    pruning_info: dict[str, Any],
    config: dict[str, Any],
    example,
    prompt_index: int,
    sample_id: int,
    seed: int,
    generated_ids: list[int],
    text: str,
) -> dict[str, Any]:
    parsed_answer = extract_math_answer(text)
    score = score_response(example.data_source, text, example.ground_truth)
    return {
        "model_id": model_id,
        "model_path": args.model_path,
        "pruning_sparsity": args.pruning_sparsity,
        "pruning_info": pruning_info,
        "prompt_id": example.example_id,
        "prompt_index": prompt_index,
        "sample_id": sample_id,
        "decoding_seed": seed,
        "decoding_config": config,
        "prompt": example.prompt_text,
        "data_source": example.data_source,
        "ground_truth": example.ground_truth,
        "generated_token_ids": [int(token_id) for token_id in generated_ids],
        "generated_text": text,
        "parsed_final_answer": parsed_answer,
        "reward": score,
        "correctness": score > 0,
        "response_length": len(generated_ids),
    }


def generate_with_vllm(args: argparse.Namespace, tokenizer, examples, model_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    vllm_cache_root = Path(os.environ.get("VLLM_CACHE_ROOT", Path(args.output).parent / "vllm_cache"))
    vllm_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_CACHE_ROOT", str(vllm_cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(vllm_cache_root / "xdg"))
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("--generation_backend vllm requires the vllm package in the active environment") from exc

    vllm_model_path, pruning_info = prepare_vllm_model_path(args, tokenizer)
    llm_kwargs: dict[str, Any] = {
        "model": vllm_model_path,
        "tokenizer": vllm_model_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": resolve_vllm_dtype(args.dtype),
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enforce_eager": args.vllm_enforce_eager,
    }
    if args.vllm_max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len
    llm = LLM(**llm_kwargs)

    records: list[dict[str, Any]] = []
    indexed_examples = list(enumerate(examples))
    for sample_id in tqdm(range(args.k), desc="generating:vllm_samples"):
        seed = sample_seed(args.seed, 0, sample_id)
        sampling_params = SamplingParams(**vllm_sampling_kwargs(args, seed=seed))
        outputs = llm.generate([example.prompt_text for _, example in indexed_examples], sampling_params, use_tqdm=True)
        for output, (prompt_index, example) in zip(outputs, indexed_examples):
            completion = output.outputs[0]
            generated_ids = list(getattr(completion, "token_ids", None) or [])
            text = completion.text
            records.append(
                make_record(
                    args=args,
                    model_id=model_id,
                    pruning_info=pruning_info,
                    config=config,
                    example=example,
                    prompt_index=prompt_index,
                    sample_id=sample_id,
                    seed=seed,
                    generated_ids=generated_ids,
                    text=text,
                )
            )
    return records


def generate_with_transformers(args: argparse.Namespace, tokenizer, examples, model_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
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

    records: list[dict[str, Any]] = []
    for prompt_index, example in enumerate(tqdm(examples, desc="generating:transformers")):
        inputs = tokenizer(
            example.prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
            return_token_type_ids=False,
        ).to(args.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        for sample_id in range(args.k):
            seed = sample_seed(args.seed, prompt_index, sample_id)
            set_seed(seed)
            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs(args, tokenizer))
            generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            records.append(
                make_record(
                    args=args,
                    model_id=model_id,
                    pruning_info=pruning_info,
                    config=config,
                    example=example,
                    prompt_index=prompt_index,
                    sample_id=sample_id,
                    seed=seed,
                    generated_ids=generated_ids,
                    text=text,
                )
            )
            del generated
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    args.enable_thinking = normalize_enable_thinking(args.enable_thinking)
    model_id = args.model_id or Path(args.model_path).name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = load_examples(
        args.dataset_path,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.debug_subset if args.debug_subset is not None else args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )

    config = {
        "generation_backend": args.generation_backend,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "k": args.k,
        "enable_thinking": args.enable_thinking,
        "prune_score_dir": args.prune_score_dir,
        "prune_score_key": args.prune_score_key,
        "prune_granularity": args.prune_granularity,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size if args.generation_backend == "vllm" else None,
    }

    if args.generation_backend == "vllm":
        records = generate_with_vllm(args, tokenizer, examples, model_id, config)
    else:
        records = generate_with_transformers(args, tokenizer, examples, model_id, config)

    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
