from __future__ import annotations

import torch
import pytest

from recoverability_pruning.online_stats import OnlineMeanCovariance
from recoverability_pruning.scoring import (
    contiguous_probe_range,
    merge_stats_directory,
    save_stats_directory,
    validate_distributed_checkpoints,
)


def test_online_welford_covariance_matches_batch_formula():
    x_values = torch.tensor([[1.0, 2.0], [2.0, -1.0], [4.0, 3.0], [-2.0, 5.0]])
    y_values = torch.tensor([[0.5, 3.0], [1.0, 2.0], [-1.0, 4.0], [2.0, -2.0]])
    stats = OnlineMeanCovariance({"weight": torch.empty(2)})
    for x, y in zip(x_values, y_values, strict=True):
        stats.update({"weight": x}, {"weight": y})
    expected = ((x_values - x_values.mean(0)) * (y_values - y_values.mean(0))).sum(0) / (len(x_values) - 1)
    torch.testing.assert_close(stats.mean_x["weight"], x_values.mean(0))
    torch.testing.assert_close(stats.mean_y["weight"], y_values.mean(0))
    torch.testing.assert_close(stats.sample_covariance()["weight"], expected)


def test_hutchinson_diagonal_and_recovery_covariance_converge():
    torch.manual_seed(11)
    dimension = 5
    raw_ref = torch.randn(dimension, dimension)
    raw_kd = torch.randn(dimension, dimension)
    h_ref = (raw_ref + raw_ref.T) / 2
    h_kd = (raw_kd + raw_kd.T) / 2
    num_probes = 250_000
    probes = torch.randint(0, 2, (num_probes, dimension), dtype=torch.float32).mul_(2).sub_(1)
    x = probes * (probes @ h_ref.T)
    y = probes * (probes @ h_kd.T)
    estimated_diagonal = x.mean(0)
    estimated_covariance = ((x - x.mean(0)) * (y - y.mean(0))).sum(0) / (num_probes - 1)
    exact_covariance = torch.diag(h_ref @ h_kd) - torch.diag(h_ref) * torch.diag(h_kd)
    torch.testing.assert_close(estimated_diagonal, torch.diag(h_ref), rtol=0.02, atol=0.015)
    torch.testing.assert_close(estimated_covariance, exact_covariance, rtol=0.03, atol=0.02)


def test_distributed_welford_merge_matches_sequential_updates(tmp_path):
    torch.manual_seed(29)
    x_values = torch.randn(9, 3, 4)
    y_values = torch.randn(9, 3, 4)
    template = {"weight": torch.empty(3, 4)}
    sequential = OnlineMeanCovariance(template)
    first = OnlineMeanCovariance(template)
    second = OnlineMeanCovariance(template)
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        sequential.update({"weight": x_value}, {"weight": y_value})
        target = first if index < 4 else second
        target.update({"weight": x_value}, {"weight": y_value})

    state_dir = save_stats_directory(second, tmp_path / "rank_0001")
    merge_stats_directory(first, state_dir, delete_after=True)
    assert first.count == sequential.count == 9
    torch.testing.assert_close(first.mean_x["weight"], sequential.mean_x["weight"])
    torch.testing.assert_close(first.mean_y["weight"], sequential.mean_y["weight"])
    torch.testing.assert_close(first.cov_m2["weight"], sequential.cov_m2["weight"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first.sample_covariance()["weight"], sequential.sample_covariance()["weight"])
    assert not state_dir.exists()


def test_contiguous_probe_ranges_and_supported_checkpoints():
    assert [contiguous_probe_range(16, 4, rank) for rank in range(4)] == [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
    ]
    validate_distributed_checkpoints(16, 4, (2, 4, 8, 16))
    with pytest.raises(ValueError):
        validate_distributed_checkpoints(16, 4, (6,))
