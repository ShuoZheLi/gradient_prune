from __future__ import annotations

import torch
from scipy.stats import spearmanr


def test_single_weight_oracle_matches_first_order_recovery_prediction():
    torch.manual_seed(23)
    dimension = 64
    factor_ref = torch.randn(dimension, dimension)
    factor_kd = torch.randn(dimension, dimension)
    h_ref = factor_ref.T @ factor_ref / dimension + 0.2 * torch.eye(dimension)
    h_kd = factor_kd.T @ factor_kd / dimension + 0.2 * torch.eye(dimension)
    dense_weight = torch.randn(dimension)
    eta = 1e-4

    rho = torch.diag(h_ref @ h_kd) - torch.diag(h_ref) * torch.diag(h_kd)
    predicted = 0.5 * dense_weight.square() * torch.diag(h_ref) - eta * dense_weight.square() * rho
    actual = []
    for index in range(dimension):
        perturbation = torch.zeros(dimension)
        perturbation[index] = -dense_weight[index]
        kd_gradient = h_kd @ perturbation
        kd_gradient[index] = 0.0
        post_step_perturbation = perturbation - eta * kd_gradient
        actual.append(0.5 * post_step_perturbation @ h_ref @ post_step_perturbation)
    actual = torch.stack(actual)

    correlation = spearmanr(predicted.numpy(), actual.numpy()).statistic
    assert correlation > 0.999
    assert torch.max(torch.abs(predicted - actual)).item() < 2e-6
