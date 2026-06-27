import math
from types import SimpleNamespace

import pytest

from calibration_loaders import CalibrationExample
from evaluate_ce import _ce_metrics, _extract_vllm_token_logprob, _nll_from_prompt_logprobs, _prepare_vllm_ce_example


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_prepare_vllm_response_only_masks_prompt_and_first_token():
    item = _prepare_vllm_ce_example(TinyTokenizer(), CalibrationExample("ab", "cd", "abcd"), "response_only", 10)
    assert item["input_ids"] == [97, 98, 99, 100]
    assert item["label_mask"] == [False, False, True, True]


def test_prepare_vllm_full_trajectory_scores_all_except_first():
    item = _prepare_vllm_ce_example(TinyTokenizer(), CalibrationExample("ab", "cd", "abcd"), "full_trajectory", 10)
    assert item["input_ids"] == [97, 98, 99, 100]
    assert item["label_mask"] == [False, True, True, True]


def test_nll_from_prompt_logprobs_matches_selected_tokens():
    input_ids = [1, 2, 3, 4]
    prompt_logprobs = [
        None,
        {2: SimpleNamespace(logprob=-0.1)},
        {3: SimpleNamespace(logprob=-0.2)},
        {4: SimpleNamespace(logprob=-0.3)},
    ]
    nll, tokens = _nll_from_prompt_logprobs(prompt_logprobs, input_ids, [False, False, True, True])
    assert tokens == 2
    assert nll == pytest.approx(0.5)
    metrics = _ce_metrics(nll, tokens, examples=1)
    assert metrics["ce"] == pytest.approx(0.25)
    assert metrics["perplexity"] == pytest.approx(math.exp(0.25))


def test_extract_vllm_token_logprob_supports_float_values():
    assert _extract_vllm_token_logprob({7: -1.25}, 7) == pytest.approx(-1.25)
