from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apply_pruning import apply_masks, load_masks  # noqa: E402
from config import load_config  # noqa: E402
from evaluate_vllm_shared import load_accuracy_examples  # noqa: E402
from model_utils import load_model_and_tokenizer  # noqa: E402


CONDITIONS = ("dense", "wanda_0p1", "rerank_0p1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose first-token EOS/empty-generation behavior for dense, WANDA, and rerank masks.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "exp_track/06_27_2026/qwen25_1p5b_math_wanda_filter_signed_taylor.yaml")
    parser.add_argument("--wanda-mask-dir", type=Path, default=REPO_ROOT / "results/qwen25_1p5b_wanda_filter_signed_taylor_math7500/masks/method=wanda/sparsity=0.1/lambda=none")
    parser.add_argument("--rerank-mask-dir", type=Path, default=REPO_ROOT / "results/qwen25_1p5b_wanda_filter_signed_taylor_math7500/masks/method=wanda_filter_signed_taylor/sparsity=0.1/lambda=none")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/qwen25_1p5b_wanda_filter_signed_taylor_math7500/first_token_diagnostics")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--logit-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = {
        "dense": None,
        "wanda_0p1": args.wanda_mask_dir,
        "rerank_0p1": args.rerank_mask_dir,
    }

    print("Loading tokenizer and prompts...", flush=True)
    _tmp_model, tokenizer = load_model_and_tokenizer(
        config.model.model_name_or_path,
        config.model.dtype,
        config.model.device,
        config.model.trust_remote_code,
    )
    del _tmp_model
    _clear_memory()

    examples = load_accuracy_examples(
        config.task_accuracy.dataset_path,
        config.task_accuracy.prompt_key,
        config.task_accuracy.response_key,
        args.num_prompts,
        tokenizer,
    )
    prompts = [example["prompt"] for example in examples]
    eos_token_ids = _eos_token_ids(tokenizer)
    primary_eos_id = tokenizer.eos_token_id
    if primary_eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    dense_log_probs = None
    all_summary_rows = []
    all_per_prompt_rows = []
    all_first_token_rows = []
    top10_by_condition = {}

    for condition, mask_dir in conditions.items():
        print(f"Running condition={condition}...", flush=True)
        model, _ = load_model_and_tokenizer(
            config.model.model_name_or_path,
            config.model.dtype,
            config.model.device,
            config.model.trust_remote_code,
        )
        if mask_dir is not None:
            masks = load_masks(mask_dir)
            apply_masks(model, masks, config.pruning.prune_ops)
        model.eval()

        logit_info = _first_token_logit_info(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.logit_batch_size,
            max_prompt_length=config.task_accuracy.max_prompt_length,
            primary_eos_id=primary_eos_id,
            eos_token_ids=eos_token_ids,
        )
        if condition == "dense":
            dense_log_probs = logit_info["log_probs"]
        if dense_log_probs is None:
            raise RuntimeError("Dense log-probs must be computed before non-dense conditions")
        kl_values = _kl_dense_to_model(dense_log_probs, logit_info["log_probs"])

        generation_info = _generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_prompt_length=config.task_accuracy.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=config.task_accuracy.temperature,
            top_p=config.task_accuracy.top_p,
            top_k=config.task_accuracy.top_k,
            seed=args.seed,
            eos_token_ids=eos_token_ids,
        )

        summary = _summary_row(condition, logit_info, generation_info, kl_values, tokenizer)
        all_summary_rows.append(summary)
        all_per_prompt_rows.extend(_per_prompt_rows(condition, examples, logit_info, generation_info, kl_values, tokenizer))
        all_first_token_rows.extend(_first_token_distribution_rows(condition, generation_info, tokenizer))
        top10_by_condition[condition] = _top10_payload(condition, logit_info, tokenizer)
        _write_generation_jsonl(args.output_dir / f"{condition}_generations.jsonl", condition, examples, generation_info, logit_info, kl_values, tokenizer)

        del model
        _clear_memory()

    _write_csv(args.output_dir / "summary.csv", all_summary_rows)
    _write_csv(args.output_dir / "per_prompt.csv", all_per_prompt_rows)
    _write_csv(args.output_dir / "first_generated_token_distribution.csv", all_first_token_rows)
    (args.output_dir / "top10_first_token_logits.json").write_text(json.dumps(top10_by_condition, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(args.output_dir, args, config, primary_eos_id, eos_token_ids)
    print(f"Wrote diagnostics to {args.output_dir}", flush=True)


def _first_token_logit_info(*, model, tokenizer, prompts: list[str], batch_size: int, max_prompt_length: int, primary_eos_id: int, eos_token_ids: list[int]) -> dict[str, Any]:
    device = next(model.parameters()).device
    all_logits = []
    all_log_probs = []
    all_probs = []
    all_eos_probs = []
    all_any_eos_probs = []
    all_eos_ranks = []
    all_top_ids = []
    all_top_logits = []
    all_top_probs = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            logits = outputs.logits[:, -1, :].float().cpu()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            eos_logits = logits[:, primary_eos_id]
            eos_ranks = (logits > eos_logits.unsqueeze(1)).sum(dim=1) + 1
            top_probs, top_ids = probs.topk(k=10, dim=-1)
            top_logits = logits.gather(1, top_ids)
            all_logits.append(logits)
            all_log_probs.append(log_probs)
            all_probs.append(probs)
            all_eos_probs.extend(probs[:, primary_eos_id].tolist())
            all_any_eos_probs.extend(probs[:, eos_token_ids].sum(dim=1).tolist())
            all_eos_ranks.extend(eos_ranks.tolist())
            all_top_ids.extend(top_ids.tolist())
            all_top_logits.extend(top_logits.tolist())
            all_top_probs.extend(top_probs.tolist())
    return {
        "logits": torch.cat(all_logits, dim=0),
        "log_probs": torch.cat(all_log_probs, dim=0),
        "probs": torch.cat(all_probs, dim=0),
        "eos_probs": all_eos_probs,
        "any_eos_probs": all_any_eos_probs,
        "eos_ranks": all_eos_ranks,
        "top_ids": all_top_ids,
        "top_logits": all_top_logits,
        "top_probs": all_top_probs,
    }


def _generate_responses(*, model, tokenizer, prompts: list[str], batch_size: int, max_prompt_length: int, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int, eos_token_ids: list[int]) -> dict[str, Any]:
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    responses = []
    generated_lengths = []
    first_token_ids = []
    first_token_is_eos = []
    raw_new_token_ids = []
    do_sample = bool(temperature and temperature > 0)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = int(top_k)
        with torch.no_grad():
            output_ids = model.generate(**encoded, **gen_kwargs)
        new_ids = output_ids[:, encoded["input_ids"].shape[1] :].detach().cpu()
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for ids, text in zip(new_ids, decoded):
            ids_list = ids.tolist()
            effective_ids = _strip_after_first_eos(ids_list, eos_token_ids)
            first_id = ids_list[0] if ids_list else None
            responses.append(text)
            generated_lengths.append(len(effective_ids))
            first_token_ids.append(first_id)
            first_token_is_eos.append(first_id in eos_token_ids if first_id is not None else False)
            raw_new_token_ids.append(ids_list)
    return {
        "responses": responses,
        "generated_lengths": generated_lengths,
        "first_token_ids": first_token_ids,
        "first_token_is_eos": first_token_is_eos,
        "raw_new_token_ids": raw_new_token_ids,
    }


def _strip_after_first_eos(ids: list[int], eos_token_ids: list[int]) -> list[int]:
    stripped = []
    eos_set = set(eos_token_ids)
    for token_id in ids:
        if token_id in eos_set:
            break
        stripped.append(token_id)
    return stripped


def _kl_dense_to_model(dense_log_probs: torch.Tensor, model_log_probs: torch.Tensor) -> list[float]:
    dense_probs = dense_log_probs.exp()
    kl = (dense_probs * (dense_log_probs - model_log_probs)).sum(dim=-1)
    return kl.tolist()


def _summary_row(condition: str, logit_info: dict[str, Any], generation_info: dict[str, Any], kl_values: list[float], tokenizer) -> dict[str, Any]:
    lengths = generation_info["generated_lengths"]
    responses = generation_info["responses"]
    first_token_ids = generation_info["first_token_ids"]
    first_token_is_eos = generation_info["first_token_is_eos"]
    first_counter = Counter(first_token_ids)
    most_common = first_counter.most_common(10)
    return {
        "condition": condition,
        "num_prompts": len(lengths),
        "avg_generated_length_tokens": _mean(lengths),
        "median_generated_length_tokens": _median(lengths),
        "fraction_empty_generations": _mean([1.0 if response.strip() == "" else 0.0 for response in responses]),
        "fraction_first_token_eos": _mean([1.0 if item else 0.0 for item in first_token_is_eos]),
        "mean_eos_probability_first_step": _mean(logit_info["eos_probs"]),
        "median_eos_probability_first_step": _median(logit_info["eos_probs"]),
        "mean_any_eos_probability_first_step": _mean(logit_info["any_eos_probs"]),
        "median_eos_rank_first_step": _median(logit_info["eos_ranks"]),
        "mean_eos_rank_first_step": _mean(logit_info["eos_ranks"]),
        "mean_kl_dense_to_condition_first_token": _mean(kl_values),
        "median_kl_dense_to_condition_first_token": _median(kl_values),
        "top_generated_first_tokens": json.dumps([_token_count_payload(token_id, count, tokenizer) for token_id, count in most_common], ensure_ascii=False),
    }


def _per_prompt_rows(condition: str, examples: list[dict[str, Any]], logit_info: dict[str, Any], generation_info: dict[str, Any], kl_values: list[float], tokenizer) -> list[dict[str, Any]]:
    rows = []
    for i, example in enumerate(examples):
        first_id = generation_info["first_token_ids"][i]
        rows.append(
            {
                "condition": condition,
                "example_id": example["example_id"],
                "generated_length_tokens": generation_info["generated_lengths"][i],
                "empty_generation": generation_info["responses"][i].strip() == "",
                "first_generated_token_id": first_id,
                "first_generated_token_text": _decode_token(tokenizer, first_id),
                "first_generated_token_is_eos": generation_info["first_token_is_eos"][i],
                "eos_probability_first_step": logit_info["eos_probs"][i],
                "any_eos_probability_first_step": logit_info["any_eos_probs"][i],
                "eos_rank_first_step": int(logit_info["eos_ranks"][i]),
                "kl_dense_to_condition_first_token": kl_values[i],
                "top10_first_token_ids": json.dumps(logit_info["top_ids"][i]),
                "top10_first_token_texts": json.dumps([_decode_token(tokenizer, token_id) for token_id in logit_info["top_ids"][i]], ensure_ascii=False),
                "top10_first_token_logits": json.dumps(logit_info["top_logits"][i]),
                "top10_first_token_probs": json.dumps(logit_info["top_probs"][i]),
            }
        )
    return rows


def _first_token_distribution_rows(condition: str, generation_info: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    total = len(generation_info["first_token_ids"])
    rows = []
    for token_id, count in Counter(generation_info["first_token_ids"]).most_common():
        rows.append(
            {
                "condition": condition,
                "token_id": token_id,
                "token_text": _decode_token(tokenizer, token_id),
                "count": count,
                "fraction": count / max(total, 1),
            }
        )
    return rows


def _top10_payload(condition: str, logit_info: dict[str, Any], tokenizer) -> dict[str, Any]:
    mean_logits = logit_info["logits"].mean(dim=0)
    mean_probs = logit_info["probs"].mean(dim=0)
    top_mean_prob_values, top_mean_prob_ids = mean_probs.topk(k=10)
    top_mean_logit_values, top_mean_logit_ids = mean_logits.topk(k=10)
    return {
        "condition": condition,
        "top10_by_mean_probability": [
            {"token_id": int(token_id), "token_text": _decode_token(tokenizer, int(token_id)), "mean_probability": float(prob), "mean_logit": float(mean_logits[int(token_id)])}
            for token_id, prob in zip(top_mean_prob_ids.tolist(), top_mean_prob_values.tolist(), strict=True)
        ],
        "top10_by_mean_logit": [
            {"token_id": int(token_id), "token_text": _decode_token(tokenizer, int(token_id)), "mean_logit": float(logit), "mean_probability": float(mean_probs[int(token_id)])}
            for token_id, logit in zip(top_mean_logit_ids.tolist(), top_mean_logit_values.tolist(), strict=True)
        ],
    }


def _write_generation_jsonl(path: Path, condition: str, examples: list[dict[str, Any]], generation_info: dict[str, Any], logit_info: dict[str, Any], kl_values: list[float], tokenizer) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, example in enumerate(examples):
            row = {
                "condition": condition,
                "example_id": example["example_id"],
                "prompt": example["prompt"],
                "response": generation_info["responses"][i],
                "generated_length_tokens": generation_info["generated_lengths"][i],
                "first_generated_token_id": generation_info["first_token_ids"][i],
                "first_generated_token_text": _decode_token(tokenizer, generation_info["first_token_ids"][i]),
                "first_generated_token_is_eos": generation_info["first_token_is_eos"][i],
                "eos_probability_first_step": logit_info["eos_probs"][i],
                "eos_rank_first_step": int(logit_info["eos_ranks"][i]),
                "kl_dense_to_condition_first_token": kl_values[i],
            }
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _token_count_payload(token_id: int | None, count: int, tokenizer) -> dict[str, Any]:
    return {"token_id": token_id, "token_text": _decode_token(tokenizer, token_id), "count": count}


def _decode_token(tokenizer, token_id: int | None) -> str:
    if token_id is None:
        return "<NONE>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _eos_token_ids(tokenizer) -> list[int]:
    ids = []
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        ids.extend(int(item) for item in eos)
    elif eos is not None:
        ids.append(int(eos))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return sorted(set(ids))


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def _median(values) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path, args, config, primary_eos_id: int, eos_token_ids: list[int]) -> None:
    text = f"""# First-Token Generation Diagnostics

Compares dense, WANDA-0.1, and WANDA-filter signed-Taylor rerank-0.1 on the same first `{args.num_prompts}` task prompts.

Generation uses Hugging Face `model.generate` with the config task-accuracy settings:
- max_new_tokens: `{args.max_new_tokens}`
- temperature: `{config.task_accuracy.temperature}`
- top_p: `{config.task_accuracy.top_p}`
- top_k: `{config.task_accuracy.top_k}`
- seed: `{args.seed}`

First-token probabilities/logits are computed by a direct forward pass on the prompt. KL is `KL(dense || condition)` over the full first-token vocabulary.

Primary EOS token id: `{primary_eos_id}`
All EOS-like token ids counted for generated first-token EOS: `{eos_token_ids}`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
