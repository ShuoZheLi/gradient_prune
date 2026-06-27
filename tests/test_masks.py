import torch
import torch.nn as nn

from masks import apply_mask_to_module, compute_mask_sparsity, make_rowwise_mask


def test_rowwise_mask_prunes_lowest_per_row():
    score = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, -1.0, 9.0, 8.0]])
    mask = make_rowwise_mask(score, 0.5)
    expected = torch.tensor([[True, False, True, False], [False, False, True, True]])
    assert torch.equal(mask, expected)
    assert (~mask).sum(dim=1).tolist() == [2, 2]


def test_no_pruning_at_zero_sparsity():
    score = torch.randn(2, 4)
    mask = make_rowwise_mask(score, 0.0)
    assert mask.all()


def test_apply_rowwise_mask_50_percent():
    module = nn.Linear(4, 2, bias=False)
    module.weight.data.fill_(1.0)
    mask = torch.tensor([[True, False, True, False], [False, False, True, True]])
    apply_mask_to_module(module, mask)
    assert float((module.weight == 0).float().mean()) == 0.5


def test_rowwise_full_sparsity_prunes_all():
    score = torch.randn(3, 5)
    mask = make_rowwise_mask(score, 1.0)
    assert not mask.any()


def test_mask_sparsity_reports_pruned_by_mask_not_existing_zeros():
    masks = {"a": torch.tensor([[True, False], [True, True]])}
    stats = compute_mask_sparsity(masks)
    assert stats["num_pruned_weights"] == 1
    assert stats["num_total_prunable_weights"] == 4
    assert stats["actual_sparsity"] == 0.25
