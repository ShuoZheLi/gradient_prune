from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from response_analysis.compute_teacher_student_alignment import (
    aggregate_alignment,
    encode_with_response_mask,
    fallback_response_token_mask,
    metric_values_from_logits,
    metric_values_from_prompt_logprobs,
    response_token_mask_from_offsets,
)


class SpecialAddingTokenizer:
    is_fast = False

    def __call__(self, text, return_token_type_ids=False, add_special_tokens=True, **kwargs):
        ids = [ord(char) for char in text]
        if add_special_tokens:
            ids = [999] + ids
        return {"input_ids": ids}


class FastOffsetTokenizer:
    is_fast = True

    def __call__(self, text, return_offsets_mapping=False, return_token_type_ids=False, add_special_tokens=True, **kwargs):
        ids = [ord(char) for char in text]
        offsets = [(idx, idx + 1) for idx in range(len(text))]
        if add_special_tokens:
            ids = [999] + ids
            offsets = [(0, 0)] + offsets
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_response_mask_uses_offsets_and_skips_special_tokens():
    offsets = [(0, 0), (0, 3), (3, 5), (5, 8), (0, 0)]
    assert response_token_mask_from_offsets(offsets, prompt_char_count=5) == [False, False, False, True, False]


def test_response_mask_counts_boundary_spanning_token():
    offsets = [(0, 4), (4, 7)]
    assert response_token_mask_from_offsets(offsets, prompt_char_count=5) == [False, True]


def test_encode_with_response_mask_does_not_insert_special_tokens():
    input_ids, mask, method = encode_with_response_mask(FastOffsetTokenizer(), "ab", "cd")
    assert input_ids == [97, 98, 99, 100]
    assert mask == [False, False, True, True]
    assert method == "offset_mapping"


def test_fallback_response_mask_does_not_count_implicit_special_tokens():
    input_ids, mask, method = encode_with_response_mask(SpecialAddingTokenizer(), "ab", "cd")
    assert input_ids == [97, 98, 99, 100]
    assert mask == [False, False, True, True]
    assert method == "prompt_token_count"
    assert fallback_response_token_mask(SpecialAddingTokenizer(), "ab", 4) == [False, False, True, True]


def test_metric_values_from_logits_compute_nll_and_top1_rate():
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0],
            [5.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
        ]
    )
    targets = torch.tensor([1, 2, 2])
    metrics = metric_values_from_logits(logits, targets)
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    assert metrics["response_token_count"] == 3
    assert metrics["teacher_token_top1_count"] == 1
    assert metrics["teacher_token_top1_rate"] == pytest.approx(1 / 3)
    assert metrics["response_logprob_sum"] == pytest.approx(float(expected_logprobs.sum()))
    assert metrics["alignment_nll_sum"] == pytest.approx(float(-expected_logprobs.sum()))
    assert metrics["alignment_nll_mean"] == pytest.approx(float(-expected_logprobs.mean()))
    assert metrics["perplexity"] == pytest.approx(math.exp(float(-expected_logprobs.mean())))


def test_metric_values_from_vllm_prompt_logprobs_compute_top1_from_rank():
    input_ids = [10, 11, 12, 13]
    prompt_logprobs = [
        None,
        {11: SimpleNamespace(logprob=-0.1, rank=1)},
        {12: SimpleNamespace(logprob=-2.0, rank=5), 99: SimpleNamespace(logprob=-0.2, rank=1)},
        {13: SimpleNamespace(logprob=-0.3, rank=1)},
    ]
    metrics = metric_values_from_prompt_logprobs(prompt_logprobs, input_ids, [False, False, True, True])
    assert metrics["response_token_count"] == 2
    assert metrics["alignment_nll_sum"] == pytest.approx(2.3)
    assert metrics["alignment_nll_mean"] == pytest.approx(1.15)
    assert metrics["teacher_token_top1_count"] == 1
    assert metrics["teacher_token_top1_rate"] == pytest.approx(0.5)


def test_metric_values_from_vllm_prompt_logprobs_falls_back_to_logprob_argmax():
    input_ids = [10, 11, 12]
    prompt_logprobs = [None, {11: -0.1}, {12: -0.5, 99: -1.0}]
    metrics = metric_values_from_prompt_logprobs(prompt_logprobs, input_ids, [False, True, True])
    assert metrics["teacher_token_top1_count"] == 2
    assert metrics["teacher_token_top1_rate"] == pytest.approx(1.0)


def test_aggregate_alignment_splits_by_correctness():
    rows = [
        {
            "correctness": True,
            "skipped": False,
            "response_token_count": 2,
            "alignment_nll_mean": 1.0,
            "alignment_nll_sum": 2.0,
            "perplexity": math.e,
            "teacher_token_top1_count": 1,
            "teacher_token_top1_rate": 0.5,
        },
        {
            "correctness": False,
            "skipped": False,
            "response_token_count": 3,
            "alignment_nll_mean": 2.0,
            "alignment_nll_sum": 6.0,
            "perplexity": math.e**2,
            "teacher_token_top1_count": 3,
            "teacher_token_top1_rate": 1.0,
        },
        {"correctness": True, "skipped": True, "alignment_nll_mean": None, "response_token_count": 0},
    ]
    aggregate = aggregate_alignment(rows)
    assert aggregate["overall"]["num_examples"] == 3
    assert aggregate["overall"]["num_scored_examples"] == 2
    assert aggregate["overall"]["mean_alignment_nll"] == pytest.approx(1.5)
    assert aggregate["overall"]["token_weighted_alignment_nll"] == pytest.approx(8 / 5)
    assert aggregate["correct_responses_only"]["num_examples"] == 2
    assert aggregate["correct_responses_only"]["num_scored_examples"] == 1
    assert aggregate["incorrect_responses_only"]["mean_teacher_token_top1_rate"] == pytest.approx(1.0)
