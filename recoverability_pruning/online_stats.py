from __future__ import annotations

from collections.abc import Mapping

import torch


class OnlineMeanCovariance:
    def __init__(self, templates: Mapping[str, torch.Tensor], *, dtype: torch.dtype = torch.float32):
        self.count = 0
        self.mean_x = {name: torch.zeros_like(tensor, dtype=dtype, device="cpu") for name, tensor in templates.items()}
        self.mean_y = {name: torch.zeros_like(tensor, dtype=dtype, device="cpu") for name, tensor in templates.items()}
        self.cov_m2 = {name: torch.zeros_like(tensor, dtype=dtype, device="cpu") for name, tensor in templates.items()}

    def update(self, x: Mapping[str, torch.Tensor], y: Mapping[str, torch.Tensor]) -> None:
        if x.keys() != self.mean_x.keys() or y.keys() != self.mean_y.keys():
            raise ValueError("Online covariance update keys do not match initialized parameter keys")
        self.count += 1
        for name in self.mean_x:
            x_value = x[name].detach().to(device="cpu", dtype=self.mean_x[name].dtype)
            y_value = y[name].detach().to(device="cpu", dtype=self.mean_y[name].dtype)
            delta_x = x_value - self.mean_x[name]
            self.mean_x[name].add_(delta_x, alpha=1.0 / self.count)
            delta_y = y_value - self.mean_y[name]
            self.mean_y[name].add_(delta_y, alpha=1.0 / self.count)
            self.cov_m2[name].addcmul_(delta_x, y_value - self.mean_y[name])

    def sample_covariance(self) -> dict[str, torch.Tensor]:
        if self.count < 2:
            raise ValueError("At least two probes are required for an unbiased sample covariance")
        return {name: value / (self.count - 1) for name, value in self.cov_m2.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "cov_m2": self.cov_m2,
        }
