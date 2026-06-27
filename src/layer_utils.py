from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import torch.nn as nn

PRUNABLE_OP_ALIASES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
    "down": "mlp.down_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.o_proj": "self_attn.o_proj",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.up_proj",
    "mlp.down_proj": "mlp.down_proj",
}
PRUNABLE_OPS = tuple(dict.fromkeys(PRUNABLE_OP_ALIASES.values()))
_SKIP_NAME_PARTS = ("embed", "embedding", "lm_head", "norm", "rotary")


def normalize_prune_ops(prune_ops: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if prune_ops is None:
        return None
    if isinstance(prune_ops, str):
        prune_ops = [prune_ops]
    normalized: list[str] = []
    for raw in prune_ops:
        for item in str(raw).split(","):
            key = item.strip()
            if not key:
                continue
            if key not in PRUNABLE_OP_ALIASES:
                raise ValueError(f"Unsupported prune op {key!r}. Choices: {sorted(PRUNABLE_OP_ALIASES)}")
            canonical = PRUNABLE_OP_ALIASES[key]
            if canonical not in normalized:
                normalized.append(canonical)
    return tuple(normalized) if normalized else None


def find_transformer_layers(model) -> list[nn.Module]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
            return list(module)
        except AttributeError:
            continue
    matches = []
    for name, module in model.named_modules():
        if re.search(r"(^|\.)(layers|h)\.\d+$", name):
            matches.append(module)
    if not matches:
        raise ValueError("Could not find transformer block list on model")
    return matches


def find_prunable_linears(block: nn.Module, prune_ops: Iterable[str] | str | None = None) -> dict[str, nn.Linear]:
    allowed = normalize_prune_ops(prune_ops)
    result: dict[str, nn.Linear] = {}
    for name, module in block.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(part in name for part in _SKIP_NAME_PARTS):
            continue
        if name in PRUNABLE_OPS and (allowed is None or name in allowed):
            result[name] = module
    return result


def iter_prunable_modules(model, prune_ops: Iterable[str] | str | None = None) -> Iterator[tuple[str, nn.Linear]]:
    for layer_idx, block in enumerate(find_transformer_layers(model)):
        for local_name, module in find_prunable_linears(block, prune_ops).items():
            yield f"model.layers.{layer_idx}.{local_name}", module
