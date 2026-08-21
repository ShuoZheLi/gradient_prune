from __future__ import annotations

import torch


def make_rademacher_probe(
    names: tuple[str, ...],
    parameters: tuple[torch.Tensor, ...],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probe = {}
    for name, parameter in zip(names, parameters, strict=True):
        values = torch.empty(parameter.shape, dtype=torch.int8, device="cpu")
        values.random_(0, 2, generator=generator)
        values.mul_(2).sub_(1)
        probe[name] = values
    return probe
