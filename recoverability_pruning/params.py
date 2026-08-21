from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn


DEFAULT_CANDIDATE_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ParameterSpace:
    hvp_names: tuple[str, ...]
    hvp_parameters: tuple[nn.Parameter, ...]
    candidate_names: tuple[str, ...]
    candidate_parameters: tuple[nn.Parameter, ...]


def parse_module_patterns(patterns: Iterable[str] | str | None) -> tuple[str, ...]:
    if patterns is None:
        return DEFAULT_CANDIDATE_MODULES
    if isinstance(patterns, str):
        patterns = [patterns]
    parsed = []
    for raw in patterns:
        for item in str(raw).split(","):
            item = item.strip()
            if item and item not in parsed:
                parsed.append(item)
    if not parsed:
        raise ValueError("At least one candidate module pattern is required")
    return tuple(parsed)


def _matches_candidate(name: str, patterns: tuple[str, ...]) -> bool:
    if not name.endswith(".weight"):
        return False
    module_name = name[: -len(".weight")]
    return any(module_name == pattern or module_name.endswith(f".{pattern}") for pattern in patterns)


def _is_transformer_parameter(name: str) -> bool:
    lower = name.lower()
    if any(part in lower for part in ("embed", "embedding", "lm_head")):
        return False
    return any(marker in name for marker in (".layers.", ".h.", "gpt_neox.layers."))


def build_parameter_space(
    model: nn.Module,
    *,
    candidate_modules: Iterable[str] | str | None = None,
    hvp_parameter_scope: str = "all",
) -> ParameterSpace:
    patterns = parse_module_patterns(candidate_modules)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.is_floating_point()]
    candidates = [(name, parameter) for name, parameter in named_parameters if _matches_candidate(name, patterns)]
    if not candidates:
        available = [name for name, _ in named_parameters if name.endswith(".weight")]
        raise ValueError(
            f"No candidate parameters matched {patterns}. Example available weights: {available[:20]}"
        )

    if hvp_parameter_scope == "candidates":
        hvp = candidates
    elif hvp_parameter_scope == "transformer":
        hvp = [(name, parameter) for name, parameter in named_parameters if _is_transformer_parameter(name)]
    elif hvp_parameter_scope == "all":
        hvp = named_parameters
    else:
        raise ValueError(
            f"Unsupported hvp_parameter_scope={hvp_parameter_scope!r}; use 'transformer', 'candidates', or 'all'"
        )
    if not hvp:
        raise ValueError(f"HVP parameter scope {hvp_parameter_scope!r} selected no parameters")

    hvp_ids = {id(parameter) for _, parameter in hvp}
    missing_candidates = [name for name, parameter in candidates if id(parameter) not in hvp_ids]
    if missing_candidates:
        raise ValueError(f"Candidate parameters must be included in HVP space: {missing_candidates[:10]}")

    for _, parameter in named_parameters:
        parameter.requires_grad_(id(parameter) in hvp_ids)
    return ParameterSpace(
        hvp_names=tuple(name for name, _ in hvp),
        hvp_parameters=tuple(parameter for _, parameter in hvp),
        candidate_names=tuple(name for name, _ in candidates),
        candidate_parameters=tuple(parameter for _, parameter in candidates),
    )


def parameter_numel(parameters: Iterable[torch.Tensor]) -> int:
    return sum(parameter.numel() for parameter in parameters)
