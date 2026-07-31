from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def count_response_tokens(response_text: str, tokenizer: TokenizerLike) -> int:
    return len(tokenizer.encode(response_text or "", add_special_tokens=False))


def load_tokenizer(tokenizer_path: str) -> TokenizerLike:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)


def aggregate_eval_metrics(
    *,
    metrics_path: str | Path,
    shard_dir: str | Path,
    max_examples: int,
    num_responses: int,
    tokenizer: TokenizerLike | None = None,
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    shard_dir = Path(shard_dir)
    if tokenizer is None:
        if not tokenizer_path:
            raise ValueError("tokenizer or tokenizer_path is required for token length metrics")
        tokenizer = load_tokenizer(tokenizer_path)

    scores: list[float] = []
    correct: list[bool] = []
    response_lengths: list[int] = []
    correct_response_lengths: list[int] = []
    wrong_response_lengths: list[int] = []
    num_examples = 0
    num_generations = 0
    num_scored = 0
    num_unscored = 0
    for shard_metrics_path in sorted(shard_dir.glob("metrics_shard_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        shard_metrics = json.loads(shard_metrics_path.read_text(encoding="utf-8"))
        num_examples += int(shard_metrics.get("num_examples", 0))
        num_generations += int(shard_metrics.get("num_generations", 0))
        num_scored += int(shard_metrics.get("num_scored", 0))
        num_unscored += int(shard_metrics.get("num_unscored", 0))

    responses_path = metrics_path.parent / "responses.jsonl"
    prompt_has_correct: dict[Any, bool] = {}
    with responses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            response_text = row.get("response") or ""
            response_length = count_response_tokens(response_text, tokenizer)
            response_lengths.append(response_length)
            if row.get("task_score") is not None:
                score = float(row.get("task_score", 0.0))
                is_correct = bool(row.get("is_correct", score == 1.0))
                scores.append(score)
                correct.append(is_correct)
                if is_correct:
                    correct_response_lengths.append(response_length)
                else:
                    wrong_response_lengths.append(response_length)
                prompt_id = row.get("example_id")
                prompt_has_correct[prompt_id] = bool(prompt_has_correct.get(prompt_id, False) or is_correct)

    metrics: dict[str, Any] = {
        "num_examples": num_examples,
        "expected_num_examples": max_examples,
        "num_responses_per_prompt": num_responses,
        "num_generations": num_generations,
        "num_scored": num_scored,
        "num_unscored": num_unscored,
    }
    if scores:
        num_correct = sum(1 for item in correct if item)
        metrics.update(
            {
                "pass@1": num_correct / len(correct),
                "accuracy": num_correct / len(correct),
                "response_accuracy": num_correct / len(correct),
                "prompt_pass_rate": sum(1 for item in prompt_has_correct.values() if item) / num_examples if num_examples else None,
                "num_prompts_with_correct_response": sum(1 for item in prompt_has_correct.values() if item),
                "mean_score": sum(scores) / len(scores),
                "score_sum": sum(scores),
                "num_correct": num_correct,
            }
        )
    metrics.update(
        {
            "avg_response_length_tokens": sum(response_lengths) / len(response_lengths) if response_lengths else None,
            "avg_correct_response_length_tokens": sum(correct_response_lengths) / len(correct_response_lengths) if correct_response_lengths else None,
            "avg_wrong_response_length_tokens": sum(wrong_response_lengths) / len(wrong_response_lengths) if wrong_response_lengths else None,
            "num_responses_for_length": len(response_lengths),
            "num_correct_responses_for_length": len(correct_response_lengths),
            "num_wrong_responses_for_length": len(wrong_response_lengths),
        }
    )
    return metrics


def write_eval_metrics(
    *,
    metrics_path: str | Path,
    shard_dir: str | Path,
    max_examples: int,
    num_responses: int,
    tokenizer_path: str,
) -> dict[str, Any]:
    metrics = aggregate_eval_metrics(
        metrics_path=metrics_path,
        shard_dir=shard_dir,
        max_examples=max_examples,
        num_responses=num_responses,
        tokenizer_path=tokenizer_path,
    )
    Path(metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
