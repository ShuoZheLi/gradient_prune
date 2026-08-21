from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping

import torch

from .losses import NLLResult
from .params import ParameterSpace


LOGGER = logging.getLogger(__name__)


def _model_input_device(model) -> torch.device:
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def compute_batch_hvp(
    model,
    batch: Mapping[str, torch.Tensor],
    probe: Mapping[str, torch.Tensor],
    parameter_space: ParameterSpace,
    loss_fn: Callable[[object, dict[str, torch.Tensor]], NLLResult],
) -> tuple[dict[str, torch.Tensor], int]:
    model.zero_grad(set_to_none=True)
    moved_batch = _move_batch(batch, _model_input_device(model))
    result = loss_fn(model, moved_batch)
    first_gradients = torch.autograd.grad(
        result.loss,
        parameter_space.hvp_parameters,
        create_graph=True,
        allow_unused=True,
    )
    directional_derivative = result.loss.new_zeros(())
    for name, gradient in zip(parameter_space.hvp_names, first_gradients, strict=True):
        if gradient is None:
            continue
        probe_tensor = probe[name].to(device=gradient.device, dtype=gradient.dtype, non_blocking=True)
        directional_derivative = directional_derivative + (gradient * probe_tensor).sum().to(result.loss.device)
    if not directional_derivative.requires_grad:
        raise RuntimeError("Directional derivative has no gradient graph; second-order autodiff is unavailable")
    candidate_hvp = torch.autograd.grad(
        directional_derivative,
        parameter_space.candidate_parameters,
        allow_unused=True,
    )
    output = {}
    for name, parameter, value in zip(
        parameter_space.candidate_names,
        parameter_space.candidate_parameters,
        candidate_hvp,
        strict=True,
    ):
        if value is None:
            value = torch.zeros_like(parameter)
        output[name] = value.detach().to(device="cpu", dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    return output, result.num_tokens


def compute_dataset_hvp(
    model,
    dataloader,
    probe: Mapping[str, torch.Tensor],
    parameter_space: ParameterSpace,
    loss_fn: Callable[[object, dict[str, torch.Tensor]], NLLResult],
    *,
    objective_name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    accumulator = {
        name: torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
        for name, parameter in zip(
            parameter_space.candidate_names,
            parameter_space.candidate_parameters,
            strict=True,
        )
    }
    total_tokens = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(dataloader, start=1):
        batch_hvp, num_tokens = compute_batch_hvp(
            model,
            batch,
            probe,
            parameter_space,
            loss_fn,
        )
        for name in accumulator:
            accumulator[name].add_(batch_hvp[name], alpha=num_tokens)
        total_tokens += num_tokens
        LOGGER.info(
            "%s HVP batch=%d supervised_tokens=%d cumulative_tokens=%d",
            objective_name,
            batch_index,
            num_tokens,
            total_tokens,
        )
        del batch_hvp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if total_tokens == 0:
        raise ValueError(f"No supervised tokens found for {objective_name} objective")
    squared_norm = 0.0
    for value in accumulator.values():
        value.div_(total_tokens)
        squared_norm += float(value.double().square().sum().item())
    return accumulator, {
        "num_tokens": float(total_tokens),
        "candidate_hvp_norm": math.sqrt(squared_norm),
        "runtime_seconds": time.perf_counter() - started,
    }
