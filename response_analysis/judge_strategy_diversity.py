from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from response_analysis.io_utils import read_jsonl, write_jsonl
from response_analysis.metrics import group_records, stable_hash, strategy_diversity

JUDGE_SYSTEM_PROMPT = """You are a careful mathematical response evaluator. Return strict JSON only.
Cluster responses by the same underlying solution method. Ignore wording, notation,
verbosity, and minor algebraic rearrangements. Distinguish genuinely different methods
such as direct algebra, enumeration, proportional reasoning, geometric construction,
case analysis, guessing, or unsupported answers. Mark invalid or incoherent reasoning separately."""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["responses", "clusters"],
    "properties": {
        "responses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["response_id", "normalized_answer", "strategy_label", "cluster_id", "valid_reasoning"],
                "properties": {
                    "response_id": {"type": "string"},
                    "normalized_answer": {"type": "string"},
                    "strategy_label": {"type": "string"},
                    "cluster_id": {"type": "integer"},
                    "valid_reasoning": {"type": "boolean"},
                },
            },
        },
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cluster_id", "description"],
                "properties": {
                    "cluster_id": {"type": "integer"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge semantic/reasoning-strategy diversity with cached OpenAI-compatible API calls.")
    parser.add_argument("--input", default="outputs/generations.jsonl")
    parser.add_argument("--output", default="outputs/semantic_judgments.jsonl")
    parser.add_argument("--cache_dir", default="outputs/api_cache")
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_EVALUATOR_MODEL", "@irom-ll37364-op-b37b3e/gpt-5.5"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=180.0, help="Per OpenAI-compatible API request timeout in seconds.")
    parser.add_argument("--shuffle_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_api", action="store_true", help="Only read cached results; fail on cache miss.")
    parser.add_argument("--json_mode", choices=["schema", "object", "none"], default="schema")
    return parser.parse_args()


def build_user_payload(prompt: str, ground_truth: Any, group: list[dict[str, Any]], order: list[int]) -> str:
    responses = []
    for output_index, group_index in enumerate(order):
        record = group[group_index]
        responses.append(
            {
                "response_id": f"r{record.get('sample_id', group_index)}",
                "text": record.get("generated_text", ""),
                "repo_parsed_answer": record.get("parsed_final_answer"),
                "repo_correctness": bool(record.get("correctness", False)),
            }
        )
    payload = {
        "task": {
            "identify_final_answer": True,
            "summarize_strategy_short_phrase": True,
            "cluster_same_underlying_solution_method": True,
            "ignore": ["wording", "notation", "verbosity", "minor algebraic rearrangements"],
            "mark_invalid_or_incoherent_separately": True,
            "required_json_shape": {
                "responses": [
                    {
                        "response_id": "r0",
                        "normalized_answer": "...",
                        "strategy_label": "...",
                        "cluster_id": 0,
                        "valid_reasoning": True,
                    }
                ],
                "clusters": [{"cluster_id": 0, "description": "..."}],
            },
        },
        "problem": prompt,
        "ground_truth_answer": ground_truth,
        "responses": responses,
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_judgment(value: Any, expected_response_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judgment is not an object")
    responses = value.get("responses")
    clusters = value.get("clusters")
    if not isinstance(responses, list) or not isinstance(clusters, list):
        raise ValueError("missing responses/clusters lists")
    seen = set()
    for item in responses:
        if not isinstance(item, dict):
            raise ValueError("response item is not object")
        response_id = item.get("response_id")
        if response_id not in expected_response_ids:
            raise ValueError(f"unexpected response_id {response_id!r}")
        if not isinstance(item.get("cluster_id"), int):
            raise ValueError("cluster_id must be integer")
        if not isinstance(item.get("valid_reasoning"), bool):
            raise ValueError("valid_reasoning must be boolean")
        for field in ("normalized_answer", "strategy_label"):
            if item.get(field) is None:
                item[field] = ""
            if not isinstance(item.get(field), str):
                raise ValueError(f"{field} must be string")
        seen.add(response_id)
    if seen != expected_response_ids:
        raise ValueError(f"missing response ids: {sorted(expected_response_ids - seen)}")
    cluster_ids = {item["cluster_id"] for item in responses}
    described = set()
    for cluster in clusters:
        if not isinstance(cluster, dict) or not isinstance(cluster.get("cluster_id"), int) or not isinstance(cluster.get("description"), str):
            raise ValueError("invalid cluster object")
        described.add(cluster["cluster_id"])
    if not cluster_ids.issubset(described):
        raise ValueError("not all used clusters are described")
    return value


def cache_path(cache_dir: str | Path, request_payload: dict[str, Any]) -> Path:
    return Path(cache_dir) / f"{stable_hash(request_payload)}.json"


def parse_json_content(content: str) -> Any:
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json") :].strip()
    if content.startswith("```"):
        content = content[len("```") :].strip()
    if content.endswith("```"):
        content = content[: -len("```")].strip()
    return json.loads(content)


def call_openai(request_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = cache_path(args.cache_dir, request_payload)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["judgment"]
    if args.disable_api:
        raise FileNotFoundError(f"cache miss with --disable_api: {path}")
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api_key is required for cache misses")

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": args.api_key, "timeout": getattr(args, "request_timeout", 180.0), "max_retries": 0}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    last_error: Exception | None = None
    include_temperature = args.temperature is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(args.max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": args.model,
                "messages": request_payload["messages"],
            }
            if include_temperature:
                kwargs["temperature"] = args.temperature
            if args.json_mode == "schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "strategy_diversity_judgment", "strict": True, "schema": JUDGE_SCHEMA},
                }
            elif args.json_mode == "object":
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            judgment = validate_judgment(parse_json_content(content), set(request_payload["expected_response_ids"]))
            path.write_text(json.dumps({"request": request_payload, "judgment": judgment}, ensure_ascii=False, indent=2), encoding="utf-8")
            return judgment
        except Exception as exc:
            last_error = exc
            if "temperature" in str(exc).lower() and "unsupported" in str(exc).lower():
                include_temperature = False
            if args.json_mode == "schema":
                args.json_mode = "object"
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to obtain valid judgment after {args.max_retries} retries: {last_error}")


def canonical_cluster_signature(judgment: dict[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    """Return order- and label-invariant co-clustering signature.

    Cluster IDs and natural-language descriptions may change between shuffled judge
    calls. The reliability check should flag only changed response partitions or
    changed validity decisions, not harmless cluster renumbering/wording.
    """
    assignments = {item["response_id"]: (item["cluster_id"], item["valid_reasoning"]) for item in judgment["responses"]}
    response_ids = sorted(assignments)
    signature: list[tuple[str, str, bool]] = []
    for left_index, left_id in enumerate(response_ids):
        left_cluster, left_valid = assignments[left_id]
        signature.append((left_id, left_id, left_valid))
        for right_id in response_ids[left_index + 1 :]:
            right_cluster, _ = assignments[right_id]
            signature.append((left_id, right_id, bool(left_cluster == right_cluster)))
    return tuple(signature)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    outputs = []
    groups = list(group_records(records, ["model_id", "prompt_id"]).items())
    if args.max_prompts >= 0:
        groups = groups[: args.max_prompts]

    for (model_id, prompt_id), group in groups:
        group = sorted(group, key=lambda row: row.get("sample_id", 0))
        repeat_judgments = []
        for repeat in range(args.shuffle_repeats):
            rng = random.Random(args.seed + repeat + int(prompt_id or 0) * 9973)
            order = list(range(len(group)))
            rng.shuffle(order)
            response_ids = {f"r{group[idx].get('sample_id', idx)}" for idx in order}
            user_content = build_user_payload(group[0].get("prompt", ""), group[0].get("ground_truth"), group, order)
            request_payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "expected_response_ids": sorted(response_ids),
                "prompt_id": prompt_id,
                "model_id": model_id,
                "repeat": repeat,
            }
            judgment = call_openai(request_payload, args)
            repeat_judgments.append(judgment)
            outputs.append(
                {
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "order": order,
                    "request_hash": stable_hash(request_payload),
                    "judgment": judgment,
                }
            )
        inconsistent = len({canonical_cluster_signature(judgment) for judgment in repeat_judgments}) > 1
        outputs[-1]["order_inconsistent_across_repeats"] = inconsistent

    write_jsonl(args.output, outputs)

    if args.metrics_output:
        import pandas as pd

        metric_rows = []
        latest_by_prompt = {}
        for output in outputs:
            latest_by_prompt[(output["model_id"], output["prompt_id"])] = output
        for (model_id, prompt_id), output in latest_by_prompt.items():
            original = sorted(group_records(records, ["model_id", "prompt_id"])[(model_id, prompt_id)], key=lambda row: row.get("sample_id", 0))
            correctness_by_id = {f"r{row.get('sample_id', idx)}": bool(row.get("correctness", False)) for idx, row in enumerate(original)}
            clusters = {item["response_id"]: item["cluster_id"] for item in output["judgment"]["responses"]}
            response_ids = sorted(clusters)
            metric_rows.append(
                {
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "order_inconsistent_across_repeats": bool(output.get("order_inconsistent_across_repeats", False)),
                    **strategy_diversity([clusters[rid] for rid in response_ids], [correctness_by_id.get(rid, False) for rid in response_ids]),
                }
            )
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metric_rows).to_parquet(metrics_path, index=False)


if __name__ == "__main__":
    main()
