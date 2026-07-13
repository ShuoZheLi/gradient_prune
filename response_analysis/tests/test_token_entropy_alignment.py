from __future__ import annotations

import torch

from response_analysis.metrics import selected_logprobs_from_logits, token_entropy_from_logits, top1_stats_from_logits


def test_selected_logprobs_align_with_targets():
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    targets = torch.tensor([1, 0])
    selected = selected_logprobs_from_logits(logits, targets)
    assert torch.all(selected > -1e-3)


def test_top1_margin_and_probability():
    logits = torch.tensor([[3.0, 1.0, 0.0]])
    prob, margin = top1_stats_from_logits(logits)
    assert margin.item() == 2.0
    assert 0.0 < prob.item() < 1.0
    assert token_entropy_from_logits(logits).item() > 0.0
