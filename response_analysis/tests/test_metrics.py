from __future__ import annotations

import math

import torch

from response_analysis.metrics import answer_diversity, strategy_diversity, surface_diversity, token_entropy_from_logits


def test_one_hot_distribution_entropy_near_zero():
    logits = torch.tensor([[1000.0, -1000.0, -1000.0]])
    entropy = token_entropy_from_logits(logits)
    assert entropy.item() < 1e-6


def test_uniform_distribution_entropy_log_vocab():
    logits = torch.zeros(2, 7)
    entropy = token_entropy_from_logits(logits)
    assert torch.allclose(entropy, torch.full((2,), math.log(7)), atol=1e-6)


def test_identical_responses_zero_answer_and_strategy_entropy():
    answer_metrics = answer_diversity(["42", "42", "42"], [True, True, True])
    strategy_metrics = strategy_diversity([0, 0, 0], [True, True, True])
    assert answer_metrics["answer_entropy"] == 0.0
    assert answer_metrics["effective_num_answers"] == 1.0
    assert answer_metrics["pass_at_1"] == 1.0
    assert answer_metrics["pass_at_k"] == 1.0
    assert answer_metrics["avg_at_k"] == 1.0
    assert strategy_metrics["strategy_entropy"] == 0.0
    assert strategy_metrics["effective_num_strategies"] == 1.0


def test_empty_strategy_subset_has_zero_effective_count():
    metrics = strategy_diversity([0, 1], [False, False])
    assert metrics["num_strategy_clusters_correct"] == 0.0
    assert metrics["strategy_entropy_correct"] == 0.0
    assert metrics["effective_num_strategies_correct"] == 0.0
    assert metrics["num_strategy_clusters_incorrect"] == 2.0


def test_response_order_permutation_preserves_diversity_metrics():
    responses = ["solve by algebra", "solve by enumeration", "solve by algebra"]
    answers = ["1", "2", "1"]
    clusters = [0, 1, 0]
    perm = [2, 0, 1]
    assert surface_diversity(responses) == surface_diversity([responses[i] for i in perm])
    assert answer_diversity(answers) == answer_diversity([answers[i] for i in perm])
    assert strategy_diversity(clusters) == strategy_diversity([clusters[i] for i in perm])


def test_answer_diversity_reports_pass_and_avg_at_k():
    metrics = answer_diversity(["1", "2", "3"], [False, True, False])
    assert metrics["pass_at_1"] == 0.0
    assert metrics["pass_at_k"] == 1.0
    assert metrics["avg_at_k"] == 1 / 3
    assert metrics["accuracy"] == metrics["avg_at_k"]
