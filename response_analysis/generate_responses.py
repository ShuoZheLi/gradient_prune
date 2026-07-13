from __future__ import annotations

import argparse
import json
import random
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K responses per prompt and store reproducible metadata.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--pruning_sparsity", type=float, default=0.0)
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


def score_response(data_source: str, response: str, ground_truth: Any) -> float:
    try:
        score_result = score_task_response(response, ground_truth, data_source=data_source)
        score = score_result[0] if isinstance(score_result, tuple) else score_result
    except Exception:
        score = 0.0
    return scalarize_score(score)


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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(args.device)
    model.eval()

    records: list[dict[str, Any]] = []
    config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "k": args.k,
        "enable_thinking": args.enable_thinking,
    }

    for prompt_index, example in enumerate(tqdm(examples, desc="generating")):
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
            parsed_answer = extract_math_answer(text)
            score = score_response(example.data_source, text, example.ground_truth)
            records.append(
                {
                    "model_id": model_id,
                    "model_path": args.model_path,
                    "pruning_sparsity": args.pruning_sparsity,
                    "prompt_id": example.example_id,
                    "prompt_index": prompt_index,
                    "sample_id": sample_id,
                    "decoding_seed": seed,
                    "decoding_config": config,
                    "prompt": example.prompt_text,
                    "data_source": example.data_source,
                    "ground_truth": example.ground_truth,
                    "generated_token_ids": generated_ids,
                    "generated_text": text,
                    "parsed_final_answer": parsed_answer,
                    "reward": score,
                    "correctness": score > 0,
                    "response_length": len(generated_ids),
                }
            )
            del generated
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
