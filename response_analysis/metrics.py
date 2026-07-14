from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

import numpy as np
import torch

try:  # Optional native backend; exact and much faster for long generations.
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except Exception:  # pragma: no cover - depends on the local environment.
    _rapidfuzz_levenshtein = None

MAX_EXACT_EDIT_CHARS = int(os.getenv("RESPONSE_ANALYSIS_MAX_EXACT_EDIT_CHARS", "512"))
MAX_APPROX_EDIT_TOKENS = int(os.getenv("RESPONSE_ANALYSIS_MAX_APPROX_EDIT_TOKENS", "256"))


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def categorical_entropy(values: Sequence[Any], *, ignore_none: bool = False) -> float:
    items = [value for value in values if not (ignore_none and value is None)]
    if not items:
        return 0.0
    counts = Counter(items)
    total = float(sum(counts.values()))
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count > 0)


def effective_count(entropy: float) -> float:
    return float(math.exp(entropy))


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return entropy in nats for each row of logits.

    Supports shape (..., vocab). The returned tensor has shape (...).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def top1_stats_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-1 probability and logit margin for logits rows."""
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    top_log_probs, _ = torch.topk(log_probs, k=min(2, log_probs.shape[-1]), dim=-1)
    top_logits, _ = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
    top1_prob = top_log_probs[..., 0].exp()
    if logits.shape[-1] == 1:
        margin = torch.full_like(top1_prob, float("inf"))
    else:
        margin = top_logits[..., 0] - top_logits[..., 1]
    return top1_prob, margin


def selected_logprobs_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def whitespace_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(whitespace_tokens(text), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    if _rapidfuzz_levenshtein is not None:
        return float(_rapidfuzz_levenshtein.normalized_distance(a, b))
    if len(a) * len(b) > MAX_EXACT_EDIT_CHARS * MAX_EXACT_EDIT_CHARS:
        tokens_a = whitespace_tokens(a.lower())[:MAX_APPROX_EDIT_TOKENS]
        tokens_b = whitespace_tokens(b.lower())[:MAX_APPROX_EDIT_TOKENS]
        if tokens_a or tokens_b:
            return 1.0 - SequenceMatcher(None, tokens_a, tokens_b, autojunk=True).ratio()
        return 1.0 - SequenceMatcher(None, a[:MAX_EXACT_EDIT_CHARS], b[:MAX_EXACT_EDIT_CHARS], autojunk=True).ratio()
    denom = max(len(a), len(b), 1)
    return levenshtein_distance(a, b) / denom


def lexical_similarity(a: str, b: str) -> float:
    tokens_a = set(whitespace_tokens(a.lower()))
    tokens_b = set(whitespace_tokens(b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def mean_pairwise(values: Sequence[Any], fn) -> float:
    if len(values) < 2:
        return 0.0
    scores = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            scores.append(fn(values[i], values[j]))
    return float(np.mean(scores)) if scores else 0.0


def surface_diversity(texts: Sequence[str]) -> dict[str, float]:
    total = len(texts)
    return {
        "num_responses": float(total),
        "unique_response_ratio": (len(set(texts)) / total) if total else 0.0,
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "mean_pairwise_normalized_edit_distance": mean_pairwise(texts, normalized_edit_distance),
        "mean_pairwise_lexical_similarity": mean_pairwise(texts, lexical_similarity),
    }


def answer_diversity(answers: Sequence[str | None], correctness: Sequence[bool | int | float | None] | None = None) -> dict[str, float]:
    all_entropy = categorical_entropy(list(answers), ignore_none=False)
    valid_answers = [answer for answer in answers if answer is not None and str(answer) != ""]
    valid_entropy = categorical_entropy(valid_answers, ignore_none=False)
    correct_values = [bool(value) for value in correctness] if correctness is not None else []
    pass_at_k = float(any(correct_values)) if correct_values else 0.0
    accuracy = float(np.mean(correct_values)) if correct_values else 0.0
    return {
        "answer_entropy_all": all_entropy,
        "answer_entropy_valid": valid_entropy,
        "answer_entropy": all_entropy,
        "num_unique_answers": float(len(set(valid_answers))),
        "effective_num_answers": effective_count(all_entropy) if answers else 0.0,
        "valid_parse_rate": (len(valid_answers) / len(answers)) if answers else 0.0,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k,
    }


def strategy_diversity(cluster_ids: Sequence[Any], correctness: Sequence[bool | int | float | None] | None = None) -> dict[str, float]:
    def stats(items: Sequence[Any], suffix: str) -> dict[str, float]:
        valid_items = [item for item in items if item is not None]
        entropy = categorical_entropy(valid_items, ignore_none=False)
        return {
            f"strategy_entropy{suffix}": entropy,
            f"effective_num_strategies{suffix}": effective_count(entropy) if valid_items else 0.0,
            f"num_strategy_clusters{suffix}": float(len(set(valid_items))),
        }

    result = stats(cluster_ids, "")
    if correctness is not None:
        correct_items = [cluster for cluster, ok in zip(cluster_ids, correctness) if bool(ok)]
        incorrect_items = [cluster for cluster, ok in zip(cluster_ids, correctness) if not bool(ok)]
        result.update(stats(correct_items, "_correct"))
        result.update(stats(incorrect_items, "_incorrect"))
    return result


def group_records(records: Iterable[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in keys)].append(record)
    return groups
