from __future__ import annotations

import json

import torch
import torch.nn as nn

from recoverability_pruning.factorized_scoring import save_score_shard
from recoverability_pruning.generate_factorized_score_file import merge_manifests, write_rank_manifest
from verl.utils.sparse_update_mask import build_masks_from_wanda_scores


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Linear(3, 2, bias=False)


def test_pt_score_shards_are_consumed_by_existing_sparse_mask_loader(tmp_path):
    parameter_name = "model.layers.0.self_attn.q_proj.weight"
    score = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.4]])
    tensors = {
        "score": score,
        "damage": torch.zeros_like(score),
        "recovery": torch.zeros_like(score),
    }
    shard_path = save_score_shard(
        tmp_path,
        parameter_name,
        tensors,
        save_factor_diagnostics=False,
        shard_format="pt",
    )
    write_rank_manifest(
        tmp_path,
        0,
        {
            "rank": 0,
            "modules": {
                parameter_name: {
                    "shard": str(shard_path.relative_to(tmp_path)),
                    "shape": list(score.shape),
                }
            },
            "group_diagnostics": [],
        },
    )
    merge_manifests(
        tmp_path,
        {
            "method": "layerwise_factorized_recoverability",
            "model_path": "tiny",
            "ref_dataset": "ref",
            "kd_dataset": "kd",
            "score_shard_format": "pt",
        },
        1,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["score_key"] == "score"
    assert metadata["modules"] == {
        "model.layers.0.self_attn.q_proj": "model__layers__0__self_attn__q_proj.pt"
    }

    masks, mask_metadata = build_masks_from_wanda_scores(
        _TinyModel(),
        tmp_path,
        keep_fraction=0.5,
        config={"target_modules": ["q_proj"], "exclude_keywords": [], "score_key": None},
    )
    mask = masks[parameter_name]
    expected = torch.tensor([[False, True, False], [True, False, True]])
    assert torch.equal(mask, expected)
    assert mask_metadata["score_key"] == "score"
