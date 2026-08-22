from __future__ import annotations

import logging
import math
import resource
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import IGNORE_INDEX, causal_sft_nll


LOGGER = logging.getLogger(__name__)


def model_input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def process_peak_cpu_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1024 if peak < 10**10 else 1))


def saved_tensor_context(
    activation_offload: str,
    *,
    pin_memory: bool,
    input_device: torch.device,
):
    if activation_offload == "none" or input_device.type != "cuda":
        return nullcontext()
    if activation_offload == "cpu":
        return torch.autograd.graph.save_on_cpu(pin_memory=pin_memory, device_type="cuda")
    raise ValueError(f"Unsupported activation_offload={activation_offload!r}")


@dataclass
class FactorPair:
    activation: torch.Tensor
    output_gradient: torch.Tensor
    activation_count: int
    output_gradient_count: int


@dataclass
class _FactorSums:
    activation: torch.Tensor | None = None
    output_gradient: torch.Tensor | None = None
    activation_count: int = 0
    output_gradient_count: int = 0


class FactorCollector:
    """Collect layer-local KFAC factors using ordinary first-order backward."""

    def __init__(
        self,
        modules: Mapping[str, nn.Linear],
        *,
        storage_device: torch.device,
        chunk_size: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.modules = dict(modules)
        self.storage_device = storage_device
        self.chunk_size = chunk_size
        self._sums = {name: _FactorSums() for name in self.modules}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_mask: torch.Tensor | None = None
        self._loss_scale: float | None = None

    def __enter__(self) -> FactorCollector:
        if self._handles:
            raise RuntimeError("FactorCollector hooks are already registered")
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_batch()

    def set_batch(self, attention_mask: torch.Tensor, *, loss_scale: float) -> None:
        if loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {loss_scale}")
        self._attention_mask = attention_mask.detach().bool()
        self._loss_scale = float(loss_scale)

    def clear_batch(self) -> None:
        self._attention_mask = None
        self._loss_scale = None

    def _make_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if self._attention_mask is None or self._loss_scale is None:
                raise RuntimeError("FactorCollector.set_batch must be called before model forward")
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError(f"Expected tensor input for {name}")
            if not torch.is_tensor(output):
                raise TypeError(f"Expected tensor output for linear module {name}")

            attention_mask = self._attention_mask
            loss_scale = self._loss_scale
            activation = inputs[0].detach()
            self._accumulate(name, "activation", activation, attention_mask, scale=1.0)

            if not output.requires_grad:
                raise RuntimeError(
                    f"Output of {name} does not require gradients. Enable input gradients and disable inference mode."
                )

            def capture_output_gradient(gradient: torch.Tensor) -> torch.Tensor:
                self._accumulate(
                    name,
                    "output_gradient",
                    gradient.detach(),
                    attention_mask,
                    scale=loss_scale,
                )
                return gradient

            output.register_hook(capture_output_gradient)

        return hook

    def _accumulate(
        self,
        name: str,
        factor_name: str,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        if values.ndim < 2:
            raise ValueError(f"Expected at least 2D values for {name}, got shape {tuple(values.shape)}")
        feature_size = values.shape[-1]
        prefix_shape = tuple(values.shape[:-1])
        if tuple(attention_mask.shape) != prefix_shape:
            raise ValueError(
                f"Attention-mask shape {tuple(attention_mask.shape)} does not match {name} prefix {prefix_shape}"
            )
        flat_values = values.reshape(-1, feature_size)
        valid = attention_mask.to(device=values.device).reshape(-1)
        selected = flat_values[valid]
        if selected.numel() == 0:
            return

        sums = self._sums[name]
        matrix = getattr(sums, factor_name)
        if matrix is None:
            matrix = torch.zeros(
                (feature_size, feature_size),
                dtype=torch.float32,
                device=self.storage_device,
            )
            setattr(sums, factor_name, matrix)

        for start in range(0, selected.shape[0], self.chunk_size):
            chunk = selected[start : start + self.chunk_size].float()
            if scale != 1.0:
                chunk = chunk * scale
            second_moment = chunk.transpose(0, 1).matmul(chunk)
            matrix.add_(second_moment.to(self.storage_device, non_blocking=False))
        count_name = f"{factor_name}_count"
        setattr(sums, count_name, getattr(sums, count_name) + selected.shape[0])

    def finalize(self) -> dict[str, FactorPair]:
        factors = {}
        for name, sums in self._sums.items():
            if sums.activation is None or sums.output_gradient is None:
                raise RuntimeError(f"No complete factors were collected for {name}")
            if sums.activation_count <= 0 or sums.output_gradient_count <= 0:
                raise RuntimeError(f"Invalid factor counts for {name}")
            factors[name] = FactorPair(
                activation=sums.activation.div(float(sums.activation_count)),
                output_gradient=sums.output_gradient.div(float(sums.output_gradient_count)),
                activation_count=sums.activation_count,
                output_gradient_count=sums.output_gradient_count,
            )
        return factors


def prepare_model_for_factor_collection(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "enable_input_require_grads"):
        raise ValueError("Model must support enable_input_require_grads for parameter-free factor collection")
    if not getattr(model, "_factorized_input_grads_enabled", False):
        model.enable_input_require_grads()
        model._factorized_input_grads_enabled = True
    model.eval()


def collect_dataset_factors(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    modules: Mapping[str, nn.Linear],
    *,
    storage_device: torch.device,
    chunk_size: int,
    dataset_name: str,
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
) -> tuple[dict[str, FactorPair], dict[str, float | int]]:
    prepare_model_for_factor_collection(model)
    total_loss_sum = 0.0
    total_supervised_tokens = 0
    total_examples = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with FactorCollector(modules, storage_device=storage_device, chunk_size=chunk_size) as collector:
        for batch_index, batch in enumerate(dataloader):
            model.zero_grad(set_to_none=True)
            moved_batch = move_batch(batch, model_input_device(model))
            supervised_tokens = int(moved_batch["labels"][:, 1:].ne(IGNORE_INDEX).sum().item())
            collector.set_batch(moved_batch["attention_mask"], loss_scale=float(supervised_tokens))
            with saved_tensor_context(
                activation_offload,
                pin_memory=activation_offload_pin_memory,
                input_device=model_input_device(model),
            ):
                result = causal_sft_nll(model, moved_batch)
                if result.num_tokens != supervised_tokens:
                    raise RuntimeError("Supervised-token count changed between hook setup and NLL computation")
                result.loss.backward()
            collector.clear_batch()
            model.zero_grad(set_to_none=True)

            total_loss_sum += float(result.loss_sum.detach().item())
            total_supervised_tokens += result.num_tokens
            total_examples += int(moved_batch["input_ids"].shape[0])
            if batch_index == 0 or (batch_index + 1) % 10 == 0:
                LOGGER.info(
                    "%s factors batch=%d examples=%d supervised_tokens=%d mean_nll=%.6f",
                    dataset_name,
                    batch_index + 1,
                    total_examples,
                    total_supervised_tokens,
                    total_loss_sum / total_supervised_tokens,
                )
        factors = collector.finalize()

    elapsed = time.perf_counter() - started
    diagnostics: dict[str, float | int] = {
        "loss": total_loss_sum / total_supervised_tokens,
        "loss_sum": total_loss_sum,
        "num_supervised_tokens": total_supervised_tokens,
        "num_examples": total_examples,
        "runtime_seconds": elapsed,
        "peak_cpu_memory_bytes": process_peak_cpu_memory_bytes(),
        "activation_offload": activation_offload,
        "activation_offload_pin_memory": activation_offload_pin_memory,
    }
    if torch.cuda.is_available():
        diagnostics["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    return factors, diagnostics


def compute_group_gradient_diagnostic(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    modules: Mapping[str, nn.Linear],
    *,
    max_batches: int,
    activation_offload: str = "none",
    activation_offload_pin_memory: bool = False,
) -> dict[str, float | int]:
    if max_batches <= 0:
        return {}
    selected_parameters = {f"{name}.weight": module.weight for name, module in modules.items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selected_parameters.values():
        parameter.requires_grad_(True)

    gradient_sums = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in selected_parameters.items()
    }
    total_loss_sum = 0.0
    total_tokens = 0
    total_examples = 0
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        moved_batch = move_batch(batch, model_input_device(model))
        with saved_tensor_context(
            activation_offload,
            pin_memory=activation_offload_pin_memory,
            input_device=model_input_device(model),
        ):
            result = causal_sft_nll(model, moved_batch)
            result.loss.backward()
        for name, parameter in selected_parameters.items():
            if parameter.grad is not None:
                gradient_sums[name].add_(parameter.grad.detach().float().cpu(), alpha=result.num_tokens)
        total_loss_sum += float(result.loss_sum.detach().item())
        total_tokens += result.num_tokens
        total_examples += int(moved_batch["input_ids"].shape[0])
    model.zero_grad(set_to_none=True)
    if total_tokens == 0:
        raise RuntimeError("Gradient diagnostic saw no supervised tokens")
    squared_norm = sum(float(gradient.square().sum().item()) for gradient in gradient_sums.values())
    return {
        "candidate_group_gradient_norm": math.sqrt(squared_norm) / total_tokens,
        "loss": total_loss_sum / total_tokens,
        "num_supervised_tokens": total_tokens,
        "num_examples": total_examples,
        "num_batches": min(max_batches, batch_index + 1),
    }
