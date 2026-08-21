from __future__ import annotations

import torch

from recoverability_pruning.online_stats import OnlineMeanCovariance


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
