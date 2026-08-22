from __future__ import annotations

import torch

from recoverability_pruning.factorized_factors import FactorPair
from recoverability_pruning.factorized_scoring import factorized_score_tensors


def _positive_semidefinite(size: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return matrix @ matrix.transpose(0, 1)


def test_factorized_score_matches_explicit_kronecker_hessians():
    generator = torch.Generator().manual_seed(17)
    d_out = 3
    d_in = 4
    activation_ref = _positive_semidefinite(d_in, generator)
    activation_kd = _positive_semidefinite(d_in, generator)
    gradient_ref = _positive_semidefinite(d_out, generator)
    gradient_kd = _positive_semidefinite(d_out, generator)
    weight = torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)
    eta = 0.07

    hessian_ref = torch.kron(gradient_ref, activation_ref)
    hessian_kd = torch.kron(gradient_kd, activation_kd)
    diagonal_ref = torch.diagonal(hessian_ref)
    product_diagonal = torch.diagonal(hessian_ref @ hessian_kd)
    rho_exact = product_diagonal - diagonal_ref * torch.diagonal(hessian_kd)
    score_exact = weight.reshape(-1).square() * (0.5 * diagonal_ref - eta * rho_exact)

    ref = FactorPair(activation_ref, gradient_ref, 1, 1)
    kd = FactorPair(activation_kd, gradient_kd, 1, 1)
    tensors = factorized_score_tensors(weight, ref, kd, eta=eta)

    assert torch.allclose(tensors["h_ref"].reshape(-1).double(), diagonal_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tensors["rho"].reshape(-1).double(), rho_exact, atol=2e-4, rtol=1e-5)
    assert torch.allclose(tensors["score"].reshape(-1).double(), score_exact, atol=2e-4, rtol=1e-5)


def test_cross_factor_diagonal_uses_transposed_second_factor():
    generator = torch.Generator().manual_seed(23)
    d_out = 2
    d_in = 3
    activation_ref = torch.randn(d_in, d_in, generator=generator, dtype=torch.float64)
    activation_kd = torch.randn(d_in, d_in, generator=generator, dtype=torch.float64)
    gradient_ref = torch.randn(d_out, d_out, generator=generator, dtype=torch.float64)
    gradient_kd = torch.randn(d_out, d_out, generator=generator, dtype=torch.float64)
    weight = torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)

    hessian_ref = torch.kron(gradient_ref, activation_ref)
    hessian_kd = torch.kron(gradient_kd, activation_kd)
    rho_exact = (
        torch.diagonal(hessian_ref @ hessian_kd)
        - torch.diagonal(hessian_ref) * torch.diagonal(hessian_kd)
    )

    ref = FactorPair(activation_ref, gradient_ref, 1, 1)
    kd = FactorPair(activation_kd, gradient_kd, 1, 1)
    tensors = factorized_score_tensors(weight, ref, kd, eta=0.1)

    assert torch.allclose(tensors["rho"].reshape(-1).double(), rho_exact, atol=1e-5, rtol=1e-5)
