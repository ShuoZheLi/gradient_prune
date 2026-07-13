from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from response_analysis.pruning import apply_score_pruning, build_masks_from_score_dir, infer_score_key


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.down_proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.model.layers[0].mlp.down_proj.weight.copy_(torch.arange(8, dtype=torch.float32).view(2, 4) + 1)


def write_score_dir(path: Path, score: torch.Tensor, key: str = "wanda") -> None:
    path.mkdir(parents=True, exist_ok=True)
    file_name = "model__layers__0__mlp__down_proj.pt"
    torch.save({key: score}, path / file_name)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "score_key": key,
                "modules": {"model.layers.0.mlp.down_proj": file_name},
                "num_modules": 1,
            }
        ),
        encoding="utf-8",
    )


def test_infer_score_key_from_wanda_metadata():
    assert infer_score_key({"score_key": "wanda"}, None) == "wanda"
    assert infer_score_key({"score_keys": ["magnitude"]}, None) == "magnitude"
    assert infer_score_key({"score_key": "wanda"}, "magnitude") == "magnitude"


def test_build_masks_from_score_dir_rowwise(tmp_path: Path):
    model = TinyModel()
    score = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, -1.0, 9.0, 8.0]])
    write_score_dir(tmp_path, score)
    masks, info = build_masks_from_score_dir(model, score_dir=tmp_path, sparsity=0.5)
    mask = masks["model.layers.0.mlp.down_proj"]
    expected = torch.tensor([[True, False, True, False], [False, False, True, True]])
    assert torch.equal(mask, expected)
    assert info["score_key"] == "wanda"
    assert info["actual_sparsity"] == 0.5


def test_apply_score_pruning_zeroes_lowest_scores(tmp_path: Path):
    model = TinyModel()
    score = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, -1.0, 9.0, 8.0]])
    write_score_dir(tmp_path, score)
    info = apply_score_pruning(model, score_dir=tmp_path, sparsity=0.5)
    weight = model.model.layers[0].mlp.down_proj.weight.detach()
    assert weight[0, 1].item() == 0.0
    assert weight[0, 3].item() == 0.0
    assert weight[1, 0].item() == 0.0
    assert weight[1, 1].item() == 0.0
    assert info["enabled"] is True
    assert info["model_actual_sparsity"] == 0.5
