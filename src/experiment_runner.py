from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from activation_stats import collect_activation_stats
from apply_pruning import apply_masks, build_masks_for_model, load_masks, save_masks, save_pruned_model
from config import load_config, save_config
from evaluate_accuracy import evaluate_task_accuracy
from evaluate_ce import evaluate_ce, evaluate_ce_vllm
from evaluate_ppl import evaluate_text_ppl, evaluate_text_ppl_vllm, load_text_examples_from_dataset
from evaluate_vllm_shared import SharedVLLMEvaluator, load_accuracy_examples
from gradient_stats import collect_gradient_stats
from layer_utils import iter_prunable_modules
from masks import compute_mask_sparsity, make_layerwise_mask, make_rowwise_mask
from model_utils import load_model_and_tokenizer
from plotting import make_plots
from pruning_scores import signed_first_order_score, signed_taylor_score

LOGGER = logging.getLogger(__name__)
RESULT_COLUMNS = ["model_name", "calibration_type", "calibration_path", "loss_on", "method", "sparsity", "lambda_value", "calibration_ce", "heldout_ce", "wikitext_ppl", "task_accuracy", "accuracy_drop", "generalization_gap", "num_pruned_weights", "num_total_prunable_weights", "actual_sparsity", "seed", "notes"]
GRAD_METHODS = {"gradient_norm", "signed_first_order", "signed_taylor", "hybrid_wanda_signed_taylor"}
ACT_METHODS = {"wanda", "hybrid_wanda_signed_taylor"}
SAVED_SCORE_METHODS = {"signed_first_order", "signed_taylor"}
WEIGHT_ONLY_METHODS = {"dense", "magnitude"}


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
    scores_dir = _resolve_scores_dir(config, root)
    results_dir.mkdir(parents=True, exist_ok=True)
    if not config.pruning.load_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config.yaml")

    model, tokenizer = load_model_and_tokenizer(config.model.model_name_or_path, config.model.dtype, config.model.device, config.model.trust_remote_code)
    methods = list(config.methods)
    _validate_pruning_source(config, methods)
    need_grad = (not config.pruning.load_masks and not config.pruning.load_scores) and any(method in GRAD_METHODS for method in methods)
    need_act = (not config.pruning.load_masks and not config.pruning.load_scores) and any(method in ACT_METHODS for method in methods)
    gradient_stats = None
    activation_stats = None
    if need_grad:
        gradient_stats = collect_gradient_stats(model, tokenizer, calibration_path=config.calibration.path, output_dir=stats_dir / "gradients" if config.output.save_stats else None, calibration_type=config.calibration.type, only_correct=config.calibration.only_correct, max_calibration_samples=config.calibration.max_samples, microbatch_size=config.calibration.microbatch_size, fisher_estimator=config.calibration.fisher_estimator, loss_on=config.calibration.loss_on, max_length=config.calibration.max_length, device=config.model.device, prune_ops=config.pruning.prune_ops, dtype=config.model.dtype, seed=getattr(config, "seed", 42), model_name=config.model.model_name_or_path)
    if need_act:
        activation_stats = collect_activation_stats(model, tokenizer, calibration_path=config.calibration.path, output_dir=stats_dir / "activations" if config.output.save_stats else None, calibration_type=config.calibration.type, only_correct=config.calibration.only_correct, max_calibration_samples=config.calibration.max_samples, microbatch_size=config.calibration.microbatch_size, loss_on="full_trajectory", max_length=config.calibration.max_length, device=config.model.device, prune_ops=config.pruning.prune_ops, seed=getattr(config, "seed", 42), model_name=config.model.model_name_or_path)
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
                mask_dir = masks_root / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
                if config.pruning.load_scores:
                    if method in SAVED_SCORE_METHODS:
                        source_scores_dir = _resolve_scores_dir(config, root)
                        masks = _build_masks_from_saved_scores(run_model, method=method, sparsity=float(sparsity), scores_dir=source_scores_dir, prune_ops=config.pruning.prune_ops, granularity=config.pruning.granularity)
                        LOGGER.info("Built masks from saved scores in %s", source_scores_dir)
                        if config.output.save_masks:
                            save_masks(masks, mask_dir, {"method": method, "sparsity": sparsity, "lambda_value": lambda_value, "source": "saved_scores", "score_root": str(source_scores_dir)})
                    else:
                        masks = build_masks_for_model(run_model, method=method, sparsity=float(sparsity), prune_ops=config.pruning.prune_ops, gradient_stats=gradient_stats, activation_stats=activation_stats, lambda_value=lambda_value, granularity=config.pruning.granularity)
                elif config.pruning.load_masks:
                    source_mask_dir = _resolve_mask_dir(config, method, sparsity, lambda_value)
                    if _mask_dir_exists(source_mask_dir):
                        masks = load_masks(source_mask_dir)
                        LOGGER.info("Loaded masks from %s", source_mask_dir)
                    else:
                        if method in SAVED_SCORE_METHODS:
                            source_scores_dir = _resolve_scores_dir(config, root)
                            masks = _build_masks_from_saved_scores(run_model, method=method, sparsity=float(sparsity), scores_dir=source_scores_dir, prune_ops=config.pruning.prune_ops, granularity=config.pruning.granularity)
                            LOGGER.info("Built masks from saved scores in %s because %s is missing", source_scores_dir, source_mask_dir)
                            if config.output.save_masks:
                                save_masks(masks, mask_dir, {"method": method, "sparsity": sparsity, "lambda_value": lambda_value, "source": "saved_scores", "score_root": str(source_scores_dir)})
                        elif method in WEIGHT_ONLY_METHODS:
                            masks = build_masks_for_model(run_model, method=method, sparsity=float(sparsity), prune_ops=config.pruning.prune_ops, gradient_stats=gradient_stats, activation_stats=activation_stats, lambda_value=lambda_value, granularity=config.pruning.granularity)
                            LOGGER.info("Built %s masks from model weights because %s is missing", method, source_mask_dir)
                            if config.output.save_masks:
                                save_masks(masks, mask_dir, {"method": method, "sparsity": sparsity, "lambda_value": lambda_value, "source": "model_weights"})
                        else:
                            raise FileNotFoundError(f"Mask metadata not found and no safe fallback is available for method={method}: {source_mask_dir / 'metadata.json'}")
                else:
                    masks = build_masks_for_model(run_model, method=method, sparsity=float(sparsity), prune_ops=config.pruning.prune_ops, gradient_stats=gradient_stats, activation_stats=activation_stats, lambda_value=lambda_value, granularity=config.pruning.granularity)
                apply_masks(run_model, masks, config.pruning.prune_ops)
                if config.output.save_masks and not config.pruning.load_masks and not config.pruning.load_scores:
                    save_masks(masks, mask_dir, {"method": method, "sparsity": sparsity, "lambda_value": lambda_value})
                if config.pruning.save_pruned_models and method != "dense":
                    save_pruned_model(run_model, tokenizer, root / "models" / f"method={method}" / f"sparsity={sparsity}")
                metrics = _evaluate_all(run_model, tokenizer, config, root, method, sparsity, lambda_value, models_to_offload=[model, run_model])
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


def _evaluate_all(model, tokenizer, config, root: Path, method: str, sparsity: float, lambda_value, models_to_offload=None):
    out = {"calibration_ce": None, "heldout_ce": None, "wikitext_ppl": None, "task_accuracy": None}
    eval_model_path = None
    if _needs_vllm_checkpoint(config):
        eval_model_path = root / "eval_models" / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
        if not _hf_checkpoint_exists(eval_model_path):
            save_pruned_model(model, tokenizer, eval_model_path)
    try:
        if _can_use_shared_vllm_evaluator(config):
            LOGGER.info("Using shared vLLM evaluator for method=%s sparsity=%s lambda=%s", method, sparsity, lambda_value)
            with _vllm_offload_context(models_to_offload, already_offloaded=False):
                return _evaluate_all_shared_vllm(config, root, method, sparsity, lambda_value, eval_model_path, out)
        offload_all = _all_enabled_evals_use_vllm(config)
        with _vllm_offload_context(models_to_offload, already_offloaded=False) if offload_all else contextlib.nullcontext():
            return _evaluate_all_impl(model, tokenizer, config, root, method, sparsity, lambda_value, eval_model_path, out, models_to_offload, offload_all)
    finally:
        _cleanup_eval_model_checkpoint(eval_model_path)


def _evaluate_all_shared_vllm(config, root: Path, method: str, sparsity: float, lambda_value, eval_model_path: Path | None, out: dict):
    if eval_model_path is None:
        raise ValueError("Shared vLLM evaluation requires a saved eval_model_path")
    max_model_len = _shared_vllm_max_model_len(config)
    base_vllm_cfg = _shared_vllm_base_config(config)
    with SharedVLLMEvaluator(
        model_path=eval_model_path,
        data_parallel_size=base_vllm_cfg.data_parallel_size,
        tensor_parallel_size=base_vllm_cfg.tensor_parallel_size,
        gpu_memory_utilization=base_vllm_cfg.gpu_memory_utilization,
        dtype=base_vllm_cfg.dtype,
        enforce_eager=base_vllm_cfg.enforce_eager,
        trust_remote_code=_shared_vllm_trust_remote_code(config),
        max_model_len=max_model_len,
        seed=getattr(config, "seed", 42),
    ) as evaluator:
        cal_cfg = config.calibration_ce
        if cal_cfg.enabled:
            cal_examples = _load_calibration_ce_examples(config)
            cal = evaluator.evaluate_ce_examples(
                examples=cal_examples,
                loss_on=cal_cfg.loss_on or config.calibration.loss_on,
                max_length=cal_cfg.max_length or config.calibration.max_length,
                batch_size=cal_cfg.batch_size,
                desc="vLLM calibration CE",
            )
            out["calibration_ce"] = cal["ce"]
        if config.heldout_ce.enabled and config.heldout_ce.path:
            held_examples = _load_heldout_ce_examples(config)
            held = evaluator.evaluate_ce_examples(
                examples=held_examples,
                loss_on=config.heldout_ce.loss_on,
                max_length=config.heldout_ce.max_length,
                batch_size=config.heldout_ce.batch_size,
                desc="vLLM heldout CE",
            )
            out["heldout_ce"] = held["ce"]
        if config.text_ppl.enabled:
            text_examples = load_text_examples_from_dataset(
                dataset_name=config.text_ppl.dataset_name,
                dataset_config=config.text_ppl.dataset_config,
                split=config.text_ppl.split,
                text_key=config.text_ppl.text_key,
                max_samples=config.text_ppl.max_samples,
            )
            ppl = evaluator.evaluate_ce_examples(
                examples=text_examples,
                loss_on="full_text",
                max_length=config.text_ppl.max_length,
                batch_size=config.text_ppl.batch_size,
                desc="vLLM text PPL",
            )
            out["wikitext_ppl"] = ppl["perplexity"]
        if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
            acc_dir = root / "accuracy" / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
            accuracy_examples = load_accuracy_examples(
                config.task_accuracy.dataset_path,
                config.task_accuracy.prompt_key,
                config.task_accuracy.response_key,
                config.task_accuracy.max_examples,
                evaluator.get_tokenizer(),
            )
            acc = evaluator.evaluate_accuracy_examples(
                examples=accuracy_examples,
                output_jsonl=acc_dir / "responses.jsonl",
                metrics_json=acc_dir / "metrics.json",
                max_prompt_length=config.task_accuracy.max_prompt_length,
                max_new_tokens=config.task_accuracy.max_new_tokens,
                temperature=config.task_accuracy.temperature,
                top_p=config.task_accuracy.top_p,
                top_k=config.task_accuracy.top_k,
                batch_size=config.task_accuracy.batch_size,
                reward_score_dir=config.task_accuracy.reward_score_dir,
                desc="vLLM accuracy",
            )
            out["task_accuracy"] = acc["accuracy"]
    return out


def _evaluate_all_impl(model, tokenizer, config, root: Path, method: str, sparsity: float, lambda_value, eval_model_path: Path | None, out: dict, models_to_offload, already_offloaded: bool):
    cal_cfg = config.calibration_ce
    if cal_cfg.enabled:
        cal_kwargs = dict(
            path=cal_cfg.path or config.calibration.path,
            calibration_type=cal_cfg.type or config.calibration.type,
            loss_on=cal_cfg.loss_on or config.calibration.loss_on,
            max_samples=cal_cfg.max_samples if cal_cfg.max_samples is not None else config.calibration.max_samples,
            max_length=cal_cfg.max_length or config.calibration.max_length,
            batch_size=cal_cfg.batch_size,
            only_correct=config.calibration.only_correct if cal_cfg.only_correct is None else cal_cfg.only_correct,
            text_key=cal_cfg.text_key or config.calibration.text_key,
            prompt_key=cal_cfg.prompt_key or config.calibration.prompt_key,
            response_key=cal_cfg.response_key or config.calibration.response_key,
        )
        if cal_cfg.backend == "vllm":
            with _vllm_offload_context(models_to_offload, already_offloaded):
                cal = evaluate_ce_vllm(model_path=eval_model_path, data_parallel_size=cal_cfg.data_parallel_size, tensor_parallel_size=cal_cfg.tensor_parallel_size, gpu_memory_utilization=cal_cfg.gpu_memory_utilization, dtype=cal_cfg.dtype, enforce_eager=cal_cfg.enforce_eager, trust_remote_code=cal_cfg.trust_remote_code, seed=getattr(config, "seed", 42), **cal_kwargs)
        elif cal_cfg.backend == "transformers":
            cal = evaluate_ce(model, tokenizer, device=config.model.device, **cal_kwargs)
        else:
            raise ValueError(f"Unsupported calibration_ce backend: {cal_cfg.backend}")
        out["calibration_ce"] = cal["ce"]
    if config.heldout_ce.enabled and config.heldout_ce.path:
        held_kwargs = dict(path=config.heldout_ce.path, calibration_type=config.calibration.type, loss_on=config.heldout_ce.loss_on, max_samples=config.heldout_ce.max_samples, max_length=config.heldout_ce.max_length, batch_size=config.heldout_ce.batch_size, text_key=config.heldout_ce.text_key, prompt_key=config.heldout_ce.prompt_key, response_key=config.heldout_ce.response_key)
        if config.heldout_ce.backend == "vllm":
            with _vllm_offload_context(models_to_offload, already_offloaded):
                held = evaluate_ce_vllm(model_path=eval_model_path, data_parallel_size=config.heldout_ce.data_parallel_size, tensor_parallel_size=config.heldout_ce.tensor_parallel_size, gpu_memory_utilization=config.heldout_ce.gpu_memory_utilization, dtype=config.heldout_ce.dtype, enforce_eager=config.heldout_ce.enforce_eager, trust_remote_code=config.heldout_ce.trust_remote_code, seed=getattr(config, "seed", 42), **held_kwargs)
        elif config.heldout_ce.backend == "transformers":
            held = evaluate_ce(model, tokenizer, device=config.model.device, **held_kwargs)
        else:
            raise ValueError(f"Unsupported heldout_ce backend: {config.heldout_ce.backend}")
        out["heldout_ce"] = held["ce"]
    if config.text_ppl.enabled:
        try:
            if config.text_ppl.backend == "vllm":
                with _vllm_offload_context(models_to_offload, already_offloaded):
                    ppl = evaluate_text_ppl_vllm(model_path=eval_model_path, dataset_name=config.text_ppl.dataset_name, dataset_config=config.text_ppl.dataset_config, split=config.text_ppl.split, text_key=config.text_ppl.text_key, max_samples=config.text_ppl.max_samples, max_length=config.text_ppl.max_length, batch_size=config.text_ppl.batch_size, data_parallel_size=config.text_ppl.data_parallel_size, tensor_parallel_size=config.text_ppl.tensor_parallel_size, gpu_memory_utilization=config.text_ppl.gpu_memory_utilization, dtype=config.text_ppl.dtype, enforce_eager=config.text_ppl.enforce_eager, trust_remote_code=config.text_ppl.trust_remote_code, seed=getattr(config, "seed", 42))
            elif config.text_ppl.backend == "transformers":
                ppl = evaluate_text_ppl(model, tokenizer, dataset_name=config.text_ppl.dataset_name, dataset_config=config.text_ppl.dataset_config, split=config.text_ppl.split, text_key=config.text_ppl.text_key, max_samples=config.text_ppl.max_samples, max_length=config.text_ppl.max_length, batch_size=config.text_ppl.batch_size, device=config.model.device)
            else:
                raise ValueError(f"Unsupported text_ppl backend: {config.text_ppl.backend}")
            out["wikitext_ppl"] = ppl["perplexity"]
        except Exception as exc:
            LOGGER.warning("Text PPL failed: %s", exc)
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        acc_dir = root / "accuracy" / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")
        accuracy_model = model
        if config.task_accuracy.backend == "vllm":
            accuracy_model = eval_model_path or (acc_dir / "vllm_model")
            if not _hf_checkpoint_exists(accuracy_model):
                save_pruned_model(model, tokenizer, accuracy_model)
        with _vllm_offload_context(models_to_offload, already_offloaded) if config.task_accuracy.backend == "vllm" else contextlib.nullcontext():
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


def _validate_pruning_source(config, methods: list[str]) -> None:
    if not config.pruning.load_scores:
        return
    unsupported = sorted(set(methods) - SAVED_SCORE_METHODS - WEIGHT_ONLY_METHODS)
    if unsupported:
        raise ValueError(
            "pruning.load_scores only supports saved-score methods "
            f"{sorted(SAVED_SCORE_METHODS)} and weight-only methods {sorted(WEIGHT_ONLY_METHODS)}; "
            f"unsupported methods: {unsupported}. Set pruning.load_scores: false to recompute required stats."
        )
    if any(method in SAVED_SCORE_METHODS for method in methods):
        score_dir = Path(config.pruning.score_root) if config.pruning.score_root else Path(config.output.root_dir) / "scores"
        if not score_dir.is_dir():
            raise FileNotFoundError(f"pruning.load_scores is true but score_root does not exist: {score_dir}")


def _cleanup_eval_model_checkpoint(eval_model_path: Path | None) -> None:
    if eval_model_path is None or not eval_model_path.exists():
        return
    shutil.rmtree(eval_model_path)
    LOGGER.info("Deleted temporary vLLM eval checkpoint %s", eval_model_path)


def _can_use_shared_vllm_evaluator(config) -> bool:
    if not _all_enabled_evals_use_vllm(config):
        return False
    base = _shared_vllm_base_config(config)
    for candidate in _enabled_vllm_configs(config):
        if not _matching_vllm_settings(base, candidate):
            return False
    return True


def _enabled_vllm_configs(config):
    if config.calibration_ce.enabled:
        yield config.calibration_ce
    if config.heldout_ce.enabled and config.heldout_ce.path:
        yield config.heldout_ce
    if config.text_ppl.enabled:
        yield config.text_ppl
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        yield config.task_accuracy


def _shared_vllm_base_config(config):
    for candidate in _enabled_vllm_configs(config):
        return candidate
    raise ValueError("No enabled vLLM evaluation config found")


def _matching_vllm_settings(base, other) -> bool:
    return (
        base.data_parallel_size == other.data_parallel_size
        and base.tensor_parallel_size == other.tensor_parallel_size
        and base.gpu_memory_utilization == other.gpu_memory_utilization
        and base.dtype == other.dtype
        and base.enforce_eager == other.enforce_eager
    )


def _shared_vllm_trust_remote_code(config) -> bool:
    values = []
    if config.calibration_ce.enabled:
        values.append(config.calibration_ce.trust_remote_code)
    if config.heldout_ce.enabled and config.heldout_ce.path:
        values.append(config.heldout_ce.trust_remote_code)
    if config.text_ppl.enabled:
        values.append(config.text_ppl.trust_remote_code)
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        values.append(config.task_accuracy.trust_remote_code)
    return any(bool(value) for value in values)


def _shared_vllm_max_model_len(config) -> int:
    lengths = []
    if config.calibration_ce.enabled:
        lengths.append(config.calibration_ce.max_length or config.calibration.max_length)
    if config.heldout_ce.enabled and config.heldout_ce.path:
        lengths.append(config.heldout_ce.max_length)
    if config.text_ppl.enabled:
        lengths.append(config.text_ppl.max_length)
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        lengths.append(config.task_accuracy.max_prompt_length + config.task_accuracy.max_new_tokens)
    return max(lengths) if lengths else 2048


def _load_calibration_ce_examples(config):
    from calibration_loaders import load_calibration_examples

    cal_cfg = config.calibration_ce
    return load_calibration_examples(
        cal_cfg.path or config.calibration.path,
        calibration_type=cal_cfg.type or config.calibration.type,
        only_correct=config.calibration.only_correct if cal_cfg.only_correct is None else cal_cfg.only_correct,
        max_samples=cal_cfg.max_samples if cal_cfg.max_samples is not None else config.calibration.max_samples,
        text_key=cal_cfg.text_key or config.calibration.text_key,
        prompt_key=cal_cfg.prompt_key or config.calibration.prompt_key,
        response_key=cal_cfg.response_key or config.calibration.response_key,
    )


def _load_heldout_ce_examples(config):
    from calibration_loaders import load_calibration_examples

    return load_calibration_examples(
        config.heldout_ce.path,
        calibration_type=config.calibration.type,
        max_samples=config.heldout_ce.max_samples,
        text_key=config.heldout_ce.text_key,
        prompt_key=config.heldout_ce.prompt_key,
        response_key=config.heldout_ce.response_key,
    )


def _all_enabled_evals_use_vllm(config) -> bool:
    backends = []
    if config.calibration_ce.enabled:
        backends.append(config.calibration_ce.backend)
    if config.heldout_ce.enabled and config.heldout_ce.path:
        backends.append(config.heldout_ce.backend)
    if config.text_ppl.enabled:
        backends.append(config.text_ppl.backend)
    if config.task_accuracy.enabled and config.task_accuracy.dataset_path:
        backends.append(config.task_accuracy.backend)
    return bool(backends) and all(backend == "vllm" for backend in backends)


def _vllm_offload_context(models, already_offloaded: bool):
    if already_offloaded:
        return contextlib.nullcontext()
    return _temporarily_offload_cuda_models(models)


@contextlib.contextmanager
def _temporarily_offload_cuda_models(models):
    unique_models = []
    seen_ids = set()
    for candidate in models or []:
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        unique_models.append(candidate)

    original_devices = []
    moved_count = 0
    for candidate in unique_models:
        try:
            original_device = next(candidate.parameters()).device
        except StopIteration:
            original_device = torch.device("cpu")
        original_devices.append(original_device)
        if original_device.type == "cuda":
            candidate.to("cpu")
            moved_count += 1

    if moved_count:
        gc.collect()
        _clear_cuda_cache()
        LOGGER.info("Temporarily offloaded %d PyTorch model(s) to CPU before vLLM evaluation", moved_count)

    try:
        yield
    finally:
        if moved_count:
            gc.collect()
            _clear_cuda_cache()
        for candidate, original_device in zip(unique_models, original_devices):
            if original_device.type == "cuda":
                candidate.to(original_device)
        if moved_count:
            _clear_cuda_cache()
            LOGGER.info("Restored %d PyTorch model(s) to GPU after vLLM evaluation", moved_count)


def _clear_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _resolve_scores_dir(config, root: Path) -> Path:
    return Path(config.pruning.score_root) if config.pruning.score_root else root / "scores"


def _mask_dir_exists(mask_dir: Path) -> bool:
    return (mask_dir / "metadata.json").is_file()


def _build_masks_from_saved_scores(model, *, method: str, sparsity: float, scores_dir: Path, prune_ops, granularity: str) -> dict[str, torch.Tensor]:
    if method not in {"signed_first_order", "signed_taylor"}:
        raise FileNotFoundError(
            f"Missing saved masks and cannot rebuild method={method!r} from saved scores. "
            "Either generate masks first or set pruning.load_masks: false."
        )
    masks = {}
    for name, module in iter_prunable_modules(model, prune_ops):
        score_path = scores_dir / f"{name.replace('.', '__')}.pt"
        if not score_path.is_file():
            raise FileNotFoundError(f"Score file not found for {name}: {score_path}")
        score_entry = torch.load(score_path, map_location="cpu")
        if method not in score_entry:
            raise KeyError(f"Score file {score_path} does not contain key {method!r}")
        score = score_entry[method]
        if tuple(score.shape) != tuple(module.weight.shape):
            raise ValueError(f"Score shape {tuple(score.shape)} for {name} does not match weight shape {tuple(module.weight.shape)}")
        if granularity == "rowwise":
            mask = make_rowwise_mask(score, sparsity)
        elif granularity == "layerwise":
            mask = make_layerwise_mask(score, sparsity)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        masks[name] = mask.cpu()
    return masks


def _resolve_mask_dir(config, method: str, sparsity, lambda_value) -> Path:
    root = Path(config.pruning.mask_root) if config.pruning.mask_root else Path(config.output.root_dir) / "masks"
    return root / f"method={method}" / f"sparsity={sparsity}" / (f"lambda={lambda_value}" if lambda_value is not None else "lambda=none")


def _needs_vllm_checkpoint(config) -> bool:
    return (
        (config.calibration_ce.enabled and config.calibration_ce.backend == "vllm")
        or (config.heldout_ce.enabled and config.heldout_ce.path and config.heldout_ce.backend == "vllm")
        or (config.text_ppl.enabled and config.text_ppl.backend == "vllm")
        or (config.task_accuracy.enabled and config.task_accuracy.dataset_path and config.task_accuracy.backend == "vllm")
    )


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
