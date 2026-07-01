from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
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

METHOD = None


def main() -> None:
    parser = argparse.ArgumentParser(description="First-token diagnostics for all pruned conditions in an experiment.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--logit-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or args.result_root / "config.yaml"
    config = load_config(config_path)
    config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    output_dir = args.output_dir or args.result_root / "first_token_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = _read_csv(args.result_root / "tables" / "main_results.csv")
    eval_map = {(row["method"], row["sparsity"], row["lambda_value"] or "none"): row for row in eval_rows}

    print("Loading tokenizer/prompts and dense reference...", flush=True)
    model, tokenizer = load_model_and_tokenizer(config.model.model_name_or_path, config.model.dtype, config.model.device, config.model.trust_remote_code)
    examples = load_accuracy_examples(config.task_accuracy.dataset_path, config.task_accuracy.prompt_key, config.task_accuracy.response_key, args.num_prompts, tokenizer)
    prompts = [example["prompt"] for example in examples]
    eos_token_ids = _eos_token_ids(tokenizer)
    primary_eos_id = tokenizer.eos_token_id
    if primary_eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    dense_log_probs = _first_token_log_probs(model, tokenizer, prompts, args.logit_batch_size, config.task_accuracy.max_prompt_length)
    del model
    _clear_memory()

    condition_specs: list[dict[str, Any]] = []
    for row in eval_rows:
        method = row["method"]
        sparsity = row["sparsity"]
        lambda_value = row["lambda_value"] or "none"
        mask_dir = args.result_root / "masks" / f"method={method}" / f"sparsity={sparsity}" / f"lambda={lambda_value}"
        label = f"{method}_s{sparsity}_lambda{lambda_value}"
        if not (mask_dir / "metadata.json").is_file():
            continue
        condition_specs.append({"label": label, "method": method, "sparsity": sparsity, "lambda_value": lambda_value, "mask_dir": mask_dir})

    summary_rows = []
    distribution_rows = []
    for spec in condition_specs:
        print(f"Running first-token diagnostic for {spec['label']}...", flush=True)
        model, _ = load_model_and_tokenizer(config.model.model_name_or_path, config.model.dtype, config.model.device, config.model.trust_remote_code)
        masks = load_masks(spec["mask_dir"])
        apply_masks(model, masks, config.pruning.prune_ops)
        model.eval()
        logit_info = _first_token_logit_info(model, tokenizer, prompts, args.logit_batch_size, config.task_accuracy.max_prompt_length, primary_eos_id, eos_token_ids)
        kl_values = _kl_dense_to_model(dense_log_probs, logit_info["log_probs"])
        gen_info = _generate(model, tokenizer, prompts, args.batch_size, config.task_accuracy.max_prompt_length, args.max_new_tokens, eos_token_ids)
        eval_key = (spec["method"], spec["sparsity"], spec["lambda_value"])
        eval_row = eval_map.get(eval_key, {})
        summary_rows.append(_summary_row(spec, eval_row, logit_info, gen_info, kl_values))
        distribution_rows.extend(_distribution_rows(spec, gen_info, tokenizer))
        del model
        _clear_memory()

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "first_generated_token_distribution.csv", distribution_rows)
    (output_dir / "README.md").write_text(
        f"First-token diagnostics on {args.num_prompts} prompts. KL is KL(dense || condition). Generation max_new_tokens={args.max_new_tokens}. EOS ids={eos_token_ids}.\n",
        encoding="utf-8",
    )
    print(f"Wrote first-token diagnostics to {output_dir}", flush=True)


def _first_token_log_probs(model, tokenizer, prompts, batch_size, max_prompt_length):
    return _first_token_logit_info(model, tokenizer, prompts, batch_size, max_prompt_length, tokenizer.eos_token_id, _eos_token_ids(tokenizer))["log_probs"]


def _first_token_logit_info(model, tokenizer, prompts, batch_size, max_prompt_length, primary_eos_id, eos_token_ids):
    device = next(model.parameters()).device
    log_probs_all = []
    eos_probs = []
    any_eos_probs = []
    eos_ranks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start:start+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
            logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits[:, -1, :].float().cpu()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            eos_logits = logits[:, primary_eos_id]
            eos_ranks.extend(((logits > eos_logits.unsqueeze(1)).sum(dim=1) + 1).tolist())
            eos_probs.extend(probs[:, primary_eos_id].tolist())
            any_eos_probs.extend(probs[:, eos_token_ids].sum(dim=1).tolist())
            log_probs_all.append(log_probs)
    return {"log_probs": torch.cat(log_probs_all, dim=0), "eos_probs": eos_probs, "any_eos_probs": any_eos_probs, "eos_ranks": eos_ranks}


def _generate(model, tokenizer, prompts, batch_size, max_prompt_length, max_new_tokens, eos_token_ids):
    device = next(model.parameters()).device
    responses = []
    lengths = []
    first_ids = []
    first_is_eos = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        new_ids = out[:, encoded["input_ids"].shape[1]:].cpu()
        responses.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        for ids in new_ids.tolist():
            first_id = ids[0] if ids else None
            first_ids.append(first_id)
            first_is_eos.append(first_id in eos_token_ids if first_id is not None else False)
            lengths.append(len(_strip_after_first_eos(ids, eos_token_ids)))
    return {"responses": responses, "lengths": lengths, "first_ids": first_ids, "first_is_eos": first_is_eos}


def _strip_after_first_eos(ids, eos_token_ids):
    eos = set(eos_token_ids)
    out = []
    for token_id in ids:
        if token_id in eos:
            break
        out.append(token_id)
    return out


def _kl_dense_to_model(dense_log_probs, model_log_probs):
    dense_probs = dense_log_probs.exp()
    return (dense_probs * (dense_log_probs - model_log_probs)).sum(dim=-1).tolist()


def _summary_row(spec, eval_row, logit_info, gen_info, kl_values):
    return {
        "condition": spec["label"],
        "method": spec["method"],
        "sparsity": spec["sparsity"],
        "lambda_value": spec["lambda_value"],
        "wikitext_ppl": _float_or_nan(eval_row.get("wikitext_ppl")),
        "task_accuracy": _float_or_nan(eval_row.get("task_accuracy")),
        "avg_generated_length_tokens": _mean(gen_info["lengths"]),
        "fraction_empty_generations": _mean([1.0 if x.strip() == "" else 0.0 for x in gen_info["responses"]]),
        "fraction_first_token_eos": _mean([1.0 if x else 0.0 for x in gen_info["first_is_eos"]]),
        "mean_first_token_eos_probability": _mean(logit_info["eos_probs"]),
        "median_first_token_eos_rank": _median(logit_info["eos_ranks"]),
        "mean_first_token_kl_vs_dense": _mean(kl_values),
        "median_first_token_kl_vs_dense": _median(kl_values),
    }


def _distribution_rows(spec, gen_info, tokenizer):
    from collections import Counter
    total = len(gen_info["first_ids"])
    return [{"condition": spec["label"], "method": spec["method"], "sparsity": spec["sparsity"], "lambda_value": spec["lambda_value"], "token_id": token_id, "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False) if token_id is not None else "<NONE>", "count": count, "fraction": count / max(total, 1)} for token_id, count in Counter(gen_info["first_ids"]).most_common()]


def _eos_token_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return sorted(set(ids))


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    values = [float(x) for x in values]
    return sum(values) / max(len(values), 1)


def _median(values):
    values = sorted(float(x) for x in values)
    if not values:
        return math.nan
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def _float_or_nan(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
