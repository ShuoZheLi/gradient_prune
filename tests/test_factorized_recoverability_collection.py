from __future__ import annotations

import torch
import torch.nn as nn

from recoverability_pruning.factorized_factors import FactorCollector


def test_factor_collector_matches_explicit_masked_second_moments():
    torch.manual_seed(5)
    linear = nn.Linear(3, 2, bias=False)
    inputs = torch.randn(2, 4, 3)
    targets = torch.randn(2, 4, 2)
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    supervised_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    num_supervised = int(supervised_mask.sum().item())

    with FactorCollector({"linear": linear}, storage_device=torch.device("cpu"), chunk_size=2) as collector:
        collector.set_batch(attention_mask, loss_scale=float(num_supervised))
        output = linear(inputs)
        residual = output - targets
        loss = 0.5 * residual[supervised_mask].square().sum() / num_supervised
        loss.backward()
        collector.clear_batch()
        factors = collector.finalize()["linear"]

    valid_inputs = inputs[attention_mask]
    expected_activation = valid_inputs.transpose(0, 1) @ valid_inputs / valid_inputs.shape[0]
    loss_sum_output_gradient = torch.zeros_like(residual)
    loss_sum_output_gradient[supervised_mask] = residual.detach()[supervised_mask]
    valid_gradients = loss_sum_output_gradient[attention_mask]
    expected_gradient = valid_gradients.transpose(0, 1) @ valid_gradients / valid_gradients.shape[0]

    assert factors.activation_count == int(attention_mask.sum().item())
    assert factors.output_gradient_count == int(attention_mask.sum().item())
    assert torch.allclose(factors.activation, expected_activation, atol=1e-6, rtol=1e-6)
    assert torch.allclose(factors.output_gradient, expected_gradient, atol=1e-6, rtol=1e-6)


def test_loss_sum_gradient_rescaling_is_batch_partition_invariant():
    torch.manual_seed(9)
    linear = nn.Linear(3, 2, bias=False)
    inputs = torch.randn(2, 3, 3)
    targets = torch.randn(2, 3, 2)
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.bool)

    def collect(batch_slices):
        linear.zero_grad(set_to_none=True)
        with FactorCollector({"linear": linear}, storage_device=torch.device("cpu"), chunk_size=8) as collector:
            for batch_slice in batch_slices:
                batch_inputs = inputs[batch_slice]
                batch_targets = targets[batch_slice]
                batch_mask = attention_mask[batch_slice]
                num_tokens = int(batch_mask.sum().item())
                collector.set_batch(batch_mask, loss_scale=float(num_tokens))
                output = linear(batch_inputs)
                loss = 0.5 * (output - batch_targets)[batch_mask].square().sum() / num_tokens
                loss.backward()
                collector.clear_batch()
                linear.zero_grad(set_to_none=True)
            return collector.finalize()["linear"]

    combined = collect([slice(0, 2)])
    split = collect([slice(0, 1), slice(1, 2)])

    assert torch.allclose(combined.activation, split.activation, atol=1e-6, rtol=1e-6)
    assert torch.allclose(combined.output_gradient, split.output_gradient, atol=1e-6, rtol=1e-6)
