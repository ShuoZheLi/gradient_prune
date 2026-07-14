from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from response_analysis.aggregate_results import main as aggregate_main
from response_analysis.compute_surface_diversity import compute_surface_metrics
from response_analysis.io_utils import read_jsonl, write_jsonl
from response_analysis.judge_strategy_diversity import (
    JUDGE_SYSTEM_PROMPT,
    build_user_payload,
    call_openai,
    canonical_cluster_signature,
)
from response_analysis.metrics import group_records, stable_hash, strategy_diversity

CONDITION_RE = re.compile(r"slurm-(?P<job_id>\d+)_.*?_sparsity_(?P<label>[^_]+)_(?P<kind>generation|on_policy_entropy|fixed_prefix_entropy)\.(?P<ext>jsonl|parquet)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process downloaded response-analysis generation/entropy files with parallel semantic judging.")
    parser.add_argument("--input_dir", default="/data/shuozhe/gradient_prune/results/07_13_2026")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_prefix", default="qwen3_8b_wanda")
    parser.add_argument("--preserve_record_model_id", action="store_true", help="Do not override model_id/pruning_sparsity from filename condition labels.")
    parser.add_argument(
        "--allow_metadata_mismatch",
        action="store_true",
        help="Allow filename sparsity labels to override contradictory record metadata. By default nonzero conditions fail on dense/zero-sparsity records.",
    )
    parser.add_argument("--skip_surface", action="store_true")
    parser.add_argument("--surface_workers", type=int, default=8, help="Parallel workers for surface/answer diversity per condition; use 1 for serial.")
    parser.add_argument("--skip_semantic_judge", action="store_true")
    parser.add_argument("--semantic_max_prompts", type=int, default=-1)
    parser.add_argument("--judge_workers", type=int, default=8)
    parser.add_argument("--judge_shuffle_repeats", type=int, default=2)
    parser.add_argument("--judge_seed", type=int, default=42)
    parser.add_argument("--judge_model", default=os.getenv("OPENAI_EVALUATOR_MODEL", "@irom-ll37364-op-b37b3e/gpt-5.5"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.portkey.ai/v1"))
    parser.add_argument("--json_mode", choices=["schema", "object", "none"], default="schema")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=180.0, help="Per Portkey/OpenAI-compatible judge request timeout in seconds.")
    parser.add_argument("--disable_api", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--baseline_model", default=None)
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def sparsity_label_to_float(label: str) -> float:
    return float(label.replace("d", "."))


def condition_model_id(model_prefix: str, label: str) -> str:
    return f"{model_prefix}_s{label}"


def numeric_values(values: Any) -> list[float]:
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return numbers


def assert_condition_metadata_matches(
    *,
    condition: dict[str, Any],
    original_models: list[str],
    original_sparsities: list[Any],
    source_path: Path,
    allow_metadata_mismatch: bool,
) -> None:
    if allow_metadata_mismatch:
        return
    condition_sparsity = float(condition["sparsity"])
    if condition_sparsity <= 0.0:
        return

    numeric_sparsities = numeric_values(original_sparsities)
    has_zero_sparsity_records = bool(numeric_sparsities) and all(abs(value) < 1e-12 for value in numeric_sparsities)
    has_dense_model_records = any(model.endswith("_dense") or model == "qwen3_8b_dense" for model in original_models)
    if has_zero_sparsity_records or has_dense_model_records:
        raise ValueError(
            "Refusing to relabel dense/zero-sparsity records as a pruned condition: "
            f"source={source_path}, condition_sparsity={condition_sparsity}, "
            f"original_models={original_models}, original_sparsities={original_sparsities}. "
            "Use --preserve_record_model_id to analyze the records as-is, or --allow_metadata_mismatch to override intentionally."
        )


def parse_condition_file(path: Path) -> dict[str, Any] | None:
    match = CONDITION_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    label = info["label"]
    return {
        "path": path,
        "job_id": info["job_id"],
        "sparsity_label": label,
        "sparsity": sparsity_label_to_float(label),
        "kind": info["kind"],
        "condition_key": f"slurm-{info['job_id']}_sparsity_{label}",
    }


def discover_inputs(input_dir: str | Path) -> dict[str, dict[str, Any]]:
    conditions: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(input_dir).glob("slurm-*resp_analysis_sparsity_*_*")):
        info = parse_condition_file(path)
        if info is None:
            continue
        condition = conditions.setdefault(
            info["condition_key"],
            {
                "job_id": info["job_id"],
                "sparsity_label": info["sparsity_label"],
                "sparsity": info["sparsity"],
                "condition_key": info["condition_key"],
            },
        )
        condition[info["kind"]] = path
    return conditions


def normalize_generation_file(
    condition: dict[str, Any],
    output_path: Path,
    *,
    model_prefix: str,
    preserve_record_model_id: bool,
    allow_metadata_mismatch: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(condition["generation"])
    intended_model_id = condition_model_id(model_prefix, condition["sparsity_label"])
    original_models = sorted({str(row.get("model_id")) for row in rows})
    original_sparsities = sorted({str(row.get("pruning_sparsity")) for row in rows})
    assert_condition_metadata_matches(
        condition=condition,
        original_models=original_models,
        original_sparsities=original_sparsities,
        source_path=Path(condition["generation"]),
        allow_metadata_mismatch=allow_metadata_mismatch or preserve_record_model_id,
    )
    normalized = []
    for row in rows:
        item = dict(row)
        item["source_file"] = Path(condition["generation"]).name
        item["condition_key"] = condition["condition_key"]
        item["condition_sparsity_label"] = condition["sparsity_label"]
        item["condition_sparsity"] = condition["sparsity"]
        item["original_model_id"] = row.get("model_id")
        item["original_pruning_sparsity"] = row.get("pruning_sparsity")
        if not preserve_record_model_id:
            item["model_id"] = intended_model_id
            item["pruning_sparsity"] = condition["sparsity"]
        normalized.append(item)
    write_jsonl(output_path, normalized)
    return {
        "condition_key": condition["condition_key"],
        "generation_rows": len(normalized),
        "model_id": normalized[0].get("model_id") if normalized else intended_model_id,
        "original_models": original_models,
        "original_sparsities": original_sparsities,
        "generation_output": str(output_path),
    }


def normalize_entropy_file(
    input_path: Path | None,
    output_path: Path,
    condition: dict[str, Any],
    *,
    model_prefix: str,
    preserve_record_model_id: bool,
    allow_metadata_mismatch: bool = False,
) -> str | None:
    if input_path is None or not Path(input_path).is_file():
        return None
    df = pd.read_parquet(input_path)
    original_models = sorted(str(value) for value in df["model_id"].dropna().unique()) if "model_id" in df.columns else []
    original_sparsities = sorted(str(value) for value in df["pruning_sparsity"].dropna().unique()) if "pruning_sparsity" in df.columns else []
    assert_condition_metadata_matches(
        condition=condition,
        original_models=original_models,
        original_sparsities=original_sparsities,
        source_path=Path(input_path),
        allow_metadata_mismatch=allow_metadata_mismatch or preserve_record_model_id,
    )
    df["source_file"] = Path(input_path).name
    df["condition_key"] = condition["condition_key"]
    df["condition_sparsity_label"] = condition["sparsity_label"]
    df["condition_sparsity"] = condition["sparsity"]
    if "model_id" in df.columns:
        df["original_model_id"] = df["model_id"]
    if "pruning_sparsity" in df.columns:
        df["original_pruning_sparsity"] = df["pruning_sparsity"]
    if not preserve_record_model_id:
        df["model_id"] = condition_model_id(model_prefix, condition["sparsity_label"])
        df["pruning_sparsity"] = condition["sparsity"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return str(output_path)


def request_payload_for_group(group: list[dict[str, Any]], model_id: str, prompt_id: Any, repeat: int, args: argparse.Namespace) -> dict[str, Any]:
    import random

    ordered = sorted(group, key=lambda row: row.get("sample_id", 0))
    try:
        prompt_seed = int(prompt_id or 0)
    except (TypeError, ValueError):
        prompt_seed = int(stable_hash(prompt_id)[:8], 16)
    rng = random.Random(args.judge_seed + repeat + prompt_seed * 9973)
    order = list(range(len(ordered)))
    rng.shuffle(order)
    response_ids = {f"r{ordered[idx].get('sample_id', idx)}" for idx in order}
    user_content = build_user_payload(ordered[0].get("prompt", ""), ordered[0].get("ground_truth"), ordered, order)
    return {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "expected_response_ids": sorted(response_ids),
        "prompt_id": prompt_id,
        "model_id": model_id,
        "repeat": repeat,
        "order": order,
    }


def judge_one(request_payload: dict[str, Any], args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    local_args = SimpleNamespace(
        cache_dir=str(cache_dir),
        disable_api=args.disable_api,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.judge_model,
        temperature=0.0,
        json_mode=args.json_mode,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )
    judgment = call_openai(request_payload, local_args)
    return {
        "model_id": request_payload["model_id"],
        "prompt_id": request_payload["prompt_id"],
        "repeat": request_payload["repeat"],
        "order": request_payload["order"],
        "request_hash": stable_hash(request_payload),
        "judge_failed": False,
        "judgment": judgment,
    }


def fallback_judgment_for_request(request_payload: dict[str, Any], error: BaseException) -> dict[str, Any]:
    responses = [
        {
            "response_id": response_id,
            "normalized_answer": "",
            "strategy_label": "judge_failed",
            "cluster_id": 0,
            "valid_reasoning": False,
        }
        for response_id in request_payload["expected_response_ids"]
    ]
    return {
        "model_id": request_payload["model_id"],
        "prompt_id": request_payload["prompt_id"],
        "repeat": request_payload["repeat"],
        "order": request_payload["order"],
        "request_hash": stable_hash(request_payload),
        "judge_failed": True,
        "judge_error": repr(error),
        "judgment": {
            "responses": responses,
            "clusters": [{"cluster_id": 0, "description": "judge_failed"}],
        },
    }


def semantic_metrics_from_judgments(records: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> pd.DataFrame:
    original_groups = group_records(records, ["model_id", "prompt_id"])
    by_prompt: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for judgment in judgments:
        by_prompt.setdefault((judgment["model_id"], judgment["prompt_id"]), []).append(judgment)

    rows = []
    for key, prompt_judgments in by_prompt.items():
        model_id, prompt_id = key
        prompt_judgments = sorted(prompt_judgments, key=lambda item: item["repeat"])
        inconsistent = len({canonical_cluster_signature(item["judgment"]) for item in prompt_judgments}) > 1
        chosen = prompt_judgments[-1]
        original = sorted(original_groups[key], key=lambda row: row.get("sample_id", 0))
        correctness_by_id = {f"r{row.get('sample_id', idx)}": bool(row.get("correctness", False)) for idx, row in enumerate(original)}
        clusters = {item["response_id"]: item["cluster_id"] for item in chosen["judgment"]["responses"]}
        response_ids = sorted(clusters)
        rows.append(
            {
                "model_id": model_id,
                "prompt_id": prompt_id,
                "order_inconsistent_across_repeats": inconsistent,
                "judge_failed_any_repeat": any(bool(item.get("judge_failed", False)) for item in prompt_judgments),
                "judge_failed_all_repeats": all(bool(item.get("judge_failed", False)) for item in prompt_judgments),
                **strategy_diversity([clusters[rid] for rid in response_ids], [correctness_by_id.get(rid, False) for rid in response_ids]),
            }
        )
    return pd.DataFrame(rows)


def run_semantic_judge_parallel(generation_path: Path, output_jsonl: Path, metrics_output: Path, cache_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(generation_path)
    groups = list(group_records(records, ["model_id", "prompt_id"]).items())
    if args.semantic_max_prompts >= 0:
        groups = groups[: args.semantic_max_prompts]
    tasks = []
    for (model_id, prompt_id), group in groups:
        for repeat in range(args.judge_shuffle_repeats):
            tasks.append(request_payload_for_group(group, model_id, prompt_id, repeat, args))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    judgments: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as executor:
            futures = {executor.submit(judge_one, task, args, cache_dir): task for task in tasks}
            iterator = tqdm(as_completed(futures), total=len(futures), desc="semantic_judge", unit="request", disable=args.disable_tqdm)
            for future in iterator:
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = fallback_judgment_for_request(task, exc)
                judgments.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

    metrics = semantic_metrics_from_judgments(records, judgments)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_output, index=False)
    return {"semantic_requests": len(tasks), "semantic_prompts": len(groups), "semantic_output": str(output_jsonl), "strategy_metrics": str(metrics_output)}


def concat_jsonl(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for path in paths:
            if not path or not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    output.write(line)


def concat_parquets(paths: list[Path], output_path: Path) -> str | None:
    frames = [pd.read_parquet(path) for path in paths if path and path.is_file()]
    if not frames:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output_path, index=False)
    return str(output_path)


def run_aggregate(combined_dir: Path, args: argparse.Namespace) -> None:
    argv = [
        "aggregate_results",
        "--generations", str(combined_dir / "generations.jsonl"),
        "--per_prompt_output", str(combined_dir / "per_prompt_metrics.csv"),
        "--aggregate_output", str(combined_dir / "aggregate_metrics.csv"),
        "--paired_output", str(combined_dir / "paired_comparisons.csv"),
        "--figures_dir", str(combined_dir / "figures"),
        "--bootstrap_samples", str(args.bootstrap_samples),
        "--seed", str(args.judge_seed),
    ]
    if args.baseline_model:
        argv.extend(["--baseline_model", args.baseline_model])
    for flag, name in [
        ("--token_metrics", "token_metrics.parquet"),
        ("--fixed_token_metrics", "fixed_token_metrics.parquet"),
        ("--response_metrics", "response_metrics.parquet"),
        ("--strategy_metrics", "strategy_metrics.parquet"),
    ]:
        path = combined_dir / name
        if path.is_file():
            argv.extend([flag, str(path)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        aggregate_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = discover_inputs(input_dir)
    if not conditions:
        raise FileNotFoundError(f"No downloaded result files found in {input_dir}")

    summaries = []
    normalized_generation_paths: list[Path] = []
    response_metric_paths: list[Path] = []
    on_policy_paths: list[Path] = []
    fixed_prefix_paths: list[Path] = []
    strategy_metric_paths: list[Path] = []

    iterator = tqdm(sorted(conditions.values(), key=lambda item: item["sparsity"]), desc="conditions", unit="condition", disable=args.disable_tqdm)
    for condition in iterator:
        if "generation" not in condition:
            continue
        condition_dir = output_dir / condition["condition_key"]
        condition_dir.mkdir(parents=True, exist_ok=True)
        generation_path = condition_dir / "generations.jsonl"
        summary = normalize_generation_file(
            condition,
            generation_path,
            model_prefix=args.model_prefix,
            preserve_record_model_id=args.preserve_record_model_id,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
        )
        normalized_generation_paths.append(generation_path)

        on_policy = normalize_entropy_file(
            condition.get("on_policy_entropy"),
            condition_dir / "token_metrics.parquet",
            condition,
            model_prefix=args.model_prefix,
            preserve_record_model_id=args.preserve_record_model_id,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
        )
        fixed = normalize_entropy_file(
            condition.get("fixed_prefix_entropy"),
            condition_dir / "fixed_token_metrics.parquet",
            condition,
            model_prefix=args.model_prefix,
            preserve_record_model_id=args.preserve_record_model_id,
            allow_metadata_mismatch=args.allow_metadata_mismatch,
        )
        if on_policy:
            on_policy_paths.append(Path(on_policy))
        if fixed:
            fixed_prefix_paths.append(Path(fixed))

        if not args.skip_surface:
            response_metrics = condition_dir / "response_metrics.parquet"
            compute_surface_metrics(generation_path, response_metrics, workers=args.surface_workers, disable_tqdm=args.disable_tqdm)
            response_metric_paths.append(response_metrics)
            summary["response_metrics"] = str(response_metrics)

        if not args.skip_semantic_judge:
            semantic_summary = run_semantic_judge_parallel(
                generation_path,
                condition_dir / "semantic_judgments.jsonl",
                condition_dir / "strategy_metrics.parquet",
                output_dir / "api_cache" / condition["condition_key"],
                args,
            )
            strategy_metric_paths.append(condition_dir / "strategy_metrics.parquet")
            summary.update(semantic_summary)
        summaries.append(summary)

    combined_dir = output_dir / "combined"
    concat_jsonl(normalized_generation_paths, combined_dir / "generations.jsonl")
    concat_parquets(response_metric_paths, combined_dir / "response_metrics.parquet")
    concat_parquets(on_policy_paths, combined_dir / "token_metrics.parquet")
    concat_parquets(fixed_prefix_paths, combined_dir / "fixed_token_metrics.parquet")
    concat_parquets(strategy_metric_paths, combined_dir / "strategy_metrics.parquet")
    run_aggregate(combined_dir, args)

    summary_path = output_dir / "processing_summary.json"
    summary_path.write_text(json.dumps({"input_dir": str(input_dir), "output_dir": str(output_dir), "conditions": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote processed results to {output_dir}")


if __name__ == "__main__":
    main()
