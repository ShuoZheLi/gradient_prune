from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from activation_stats import collect_activation_stats
from apply_pruning import apply_masks, build_masks_for_model, save_masks, save_pruned_model
from config import load_config, save_config
from evaluate_accuracy import evaluate_task_accuracy
from evaluate_ce import evaluate_ce
from evaluate_ppl import evaluate_wikitext_ppl
from gradient_stats import collect_gradient_stats
from layer_utils import iter_prunable_modules
from masks import compute_mask_sparsity
from model_utils import load_model_and_tokenizer
from plotting import make_plots
from pruning_scores import signed_first_order_score, signed_taylor_score

LOGGER = logging.getLogger(__name__)
RESULT_COLUMNS = ["model_name", "calibration_type", "calibration_path", "loss_on", "method", "sparsity", "lambda_value", "calibration_ce", "heldout_ce", "wikitext_ppl", "task_accuracy", "accuracy_drop", "generalization_gap", "num_pruned_weights", "num_total_prunable_weights", "actual_sparsity", "seed", "notes"]
GRAD_METHODS = {"gradient_norm", "signed_first_order", "signed_taylor", "hybrid_wanda_signed_taylor"}
ACT_METHODS = {"wanda", "hybrid_wanda_signed_taylor"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config_path: str | Path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    set_seed(config.seed)
    root = Path(config.output.root_dir)
    results_dir = root / "tables"
    stats_dir = root / "stats"
    masks_root = root / "masks"
    scores_dir = root / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config.yaml")

    model, tokenizer = load_model_and_tokenizer(config.model.model_name_or_path, config.model.dtype, config.model.device, config.model.trust_remote_code)
    methods = list(config.methods)
    need_grad = any(method in GRAD_METHODS for method in methods)
    need_act = any(method in ACT_METHODS for method in methods)
    gradient_stats = None
    activation_stats = None
    if need_grad:
        gradient_stats = collect_gradient_stats(model, tokenizer, calibration_path=config.calibration.path, output_dir=stats_dir / "gradients" if config.output.save_stats else None, calibration_type=config.calibration.type, only_correct=config.calibration.only_correct, max_calibration_samples=config.calibration.max_samples, microbatch_size=config.calibration.microbatch_size, loss_on=config.calibration.loss_on, max_length=config.calibration.max_length, device=config.model.device, prune_ops=config.pruning.prune_ops, dtype=config.model.dtype, seed=config.seed, model_name=config.model.model_name_or_path)
    if need_act:
        activation_stats = collect_activation_stats(model, tokenizer, calibration_path=config.calibration.path, output_dir=stats_dir / "activations" if config.output.save_stats else None, calibration_type=config.calibration.type, only_correct=config.calibration.only_correct, max_calibration_samples=config.calibration.max_samples, microbatch_size=config.calibration.microbatch_size, loss_on="full_trajectory", max_length=config.calibration.max_length, device=config.model.device, prune_ops=config.pruning.prune_ops, seed=config.seed, model_name=config.model.model_name_or_path)
    _save_representative_scores(model, gradient_stats, scores_dir, config.pruning.prune_ops)

    rows = []
    dense_accuracy = None
    for method in methods:
        lambda_values = config.hybrid.lambda_values if method == "hybrid_wanda_signed_taylor" else [None]
        for lambda_value in lambda_values:
            for sparsity in config.pruning.sparsities:
                if method == "dense" and float(sparsity) != 0.0:
                    continue
                LOGGER.info("Running method=%s sparsity=%s lambda=%s", method, sparsity, lambda_value)
                run_model = model if method == "dense" and float(sparsity) == 0.0 else copy.deepcopy(model)
                masks = build_masks_for_model(run_model, method=method, sparsity=float(sparsity), prune_ops=config.pruning.prune_ops, gradient_stats=gradient_stats, activation_stats=activation_stats, lambda_value=lambda_value, granularity=config.pruning.granularity)
                apply_masks(run_model, masks, config.pruning.prune_ops)
                mask_dir = masks_root / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
                if config.output.save_masks:
                    save_masks(masks, mask_dir, {"method": method, "sparsity": sparsity, "lambda_value": lambda_value})
                if config.pruning.save_pruned_models and method != "dense":
                    save_pruned_model(run_model, tokenizer, root / "models" / f"method={method}" / f"sparsity={sparsity}")
                metrics = _evaluate_all(run_model, tokenizer, config, root, method, sparsity, lambda_value)
                sparsity_metrics = compute_mask_sparsity(masks)
                if method == "dense" and dense_accuracy is None:
                    dense_accuracy = metrics.get("task_accuracy")
                accuracy_drop = None
                if dense_accuracy is not None and metrics.get("task_accuracy") is not None:
                    accuracy_drop = dense_accuracy - metrics["task_accuracy"]
                row = {"model_name": config.model.model_name_or_path, "calibration_type": config.calibration.type, "calibration_path": config.calibration.path, "loss_on": config.calibration.loss_on, "method": method, "sparsity": float(sparsity), "lambda_value": lambda_value, **metrics, "accuracy_drop": accuracy_drop, "generalization_gap": _diff(metrics.get("heldout_ce"), metrics.get("calibration_ce")), **sparsity_metrics, "seed": config.seed, "notes": "microbatch Fisher for gradient methods" if method in GRAD_METHODS else ""}
                rows.append(row)
                _write_results(rows, results_dir)
                if run_model is not model:
                    del run_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    if config.output.save_plots:
        make_plots(results_dir / "main_results.csv", root / "plots", scores_dir)
    return rows


def _evaluate_all(model, tokenizer, config, root: Path, method: str, sparsity: float, lambda_value):
    out = {"calibration_ce": None, "heldout_ce": None, "wikitext_ppl": None, "task_accuracy": None}
    cal_cfg = config.calibration_ce
    cal = evaluate_ce(
        model,
        tokenizer,
        path=cal_cfg.path or config.calibration.path,
        calibration_type=cal_cfg.type or config.calibration.type,
        loss_on=cal_cfg.loss_on or config.calibration.loss_on,
        max_samples=cal_cfg.max_samples if cal_cfg.max_samples is not None else config.calibration.max_samples,
        max_length=cal_cfg.max_length or config.calibration.max_length,
        batch_size=cal_cfg.batch_size,
        device=config.model.device,
        only_correct=config.calibration.only_correct if cal_cfg.only_correct is None else cal_cfg.only_correct,
        text_key=cal_cfg.text_key or config.calibration.text_key,
        prompt_key=cal_cfg.prompt_key or config.calibration.prompt_key,
        response_key=cal_cfg.response_key or config.calibration.response_key,
    )
    out["calibration_ce"] = cal["ce"]
    if config.heldout_ce.path:
        held = evaluate_ce(model, tokenizer, path=config.heldout_ce.path, calibration_type=config.calibration.type, loss_on=config.heldout_ce.loss_on, max_samples=config.heldout_ce.max_samples, max_length=config.heldout_ce.max_length, batch_size=config.heldout_ce.batch_size, device=config.model.device, text_key=config.heldout_ce.text_key, prompt_key=config.heldout_ce.prompt_key, response_key=config.heldout_ce.response_key)
        out["heldout_ce"] = held["ce"]
    if config.wikitext.enabled:
        try:
            ppl = evaluate_wikitext_ppl(model, tokenizer, dataset_name=config.wikitext.dataset_name, dataset_config=config.wikitext.dataset_config, split=config.wikitext.split, text_key=config.wikitext.text_key, max_samples=config.wikitext.max_samples, max_length=config.wikitext.max_length, batch_size=1, device=config.model.device)
            out["wikitext_ppl"] = ppl["perplexity"]
        except Exception as exc:
            LOGGER.warning("WikiText PPL failed: %s", exc)
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        acc_dir = root / "accuracy" / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
        accuracy_model = model
        if config.task_accuracy.backend == "vllm":
            accuracy_model = acc_dir / "vllm_model"
            if not _hf_checkpoint_exists(accuracy_model):
                save_pruned_model(model, tokenizer, accuracy_model)
        acc = evaluate_task_accuracy(
            accuracy_model,
            tokenizer,
            config.task_accuracy.dataset_path,
            config.task_accuracy.backend,
            acc_dir / "responses.jsonl",
            acc_dir / "metrics.json",
            config.task_accuracy.prompt_key,
            config.task_accuracy.response_key,
            config.task_accuracy.reward_score_dir,
            config.task_accuracy.max_examples,
            config.task_accuracy.max_prompt_length,
            config.task_accuracy.max_new_tokens,
            config.task_accuracy.temperature,
            config.task_accuracy.top_p,
            config.task_accuracy.top_k,
            config.task_accuracy.batch_size,
            config.seed,
            config.task_accuracy.data_parallel_size,
            config.task_accuracy.tensor_parallel_size,
            config.task_accuracy.gpu_memory_utilization,
            config.task_accuracy.dtype,
            config.task_accuracy.enforce_eager,
            config.task_accuracy.trust_remote_code,
        )
        out["task_accuracy"] = acc["accuracy"]
    return out


def _hf_checkpoint_exists(path: Path) -> bool:
    has_config = (path / "config.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return has_config and has_weights


def _save_representative_scores(model, gradient_stats, scores_dir: Path, prune_ops):
    if not gradient_stats:
        return
    for name, module in iter_prunable_modules(model, prune_ops):
        entry = gradient_stats[name]
        torch.save({"signed_first_order": signed_first_order_score(module.weight.detach().cpu(), entry["g"]), "signed_taylor": signed_taylor_score(module.weight.detach().cpu(), entry["g"], entry["h"])}, scores_dir / f"{name.replace('.', '__')}.pt")


def _diff(a, b):
    return None if a is None or b is None else a - b


def _write_results(rows, results_dir: Path):
    csv_path = results_dir / "main_results.csv"
    json_path = results_dir / "main_results.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
