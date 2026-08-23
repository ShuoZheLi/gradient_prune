from __future__ import annotations

import json
from collections import OrderedDict

import pytest
import torch.nn as nn

from recoverability_pruning.generate_factorized_score_file import load_resume_rank_manifest


def _group(*names: str) -> OrderedDict[str, nn.Linear]:
    return OrderedDict((name, nn.Linear(2, 2, bias=False)) for name in names)


def _write_manifest(tmp_path, payload: dict[str, object]) -> None:
    (tmp_path / "manifest.rank00000.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resume_skips_group_when_all_manifest_shards_exist(tmp_path):
    group = _group("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj")
    modules = {}
    for name in group:
        shard = tmp_path / f"{name.rsplit('.', 1)[-1]}.pt"
        shard.write_bytes(b"complete")
        modules[f"{name}.weight"] = {"shard": shard.name}
    diagnostic = {"modules": list(group), "reference_pass": {"examples": 1}}
    _write_manifest(
        tmp_path,
        {"rank": 0, "modules": modules, "group_diagnostics": [diagnostic]},
    )

    payload, completed_groups = load_resume_rank_manifest(tmp_path, 0, [group])

    assert completed_groups == {0}
    assert payload["modules"] == modules
    assert payload["group_diagnostics"] == [diagnostic]


def test_resume_recomputes_whole_group_when_one_shard_is_missing(tmp_path):
    first_group = _group("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj")
    second_group = _group("model.layers.1.self_attn.q_proj")
    existing_shard = tmp_path / "layer0_q.pt"
    existing_shard.write_bytes(b"complete")
    second_shard = tmp_path / "layer1_q.pt"
    second_shard.write_bytes(b"complete")
    first_diagnostic = {"modules": list(first_group)}
    second_diagnostic = {"modules": list(second_group)}
    _write_manifest(
        tmp_path,
        {
            "rank": 0,
            "modules": {
                "model.layers.0.self_attn.q_proj.weight": {"shard": existing_shard.name},
                "model.layers.0.self_attn.k_proj.weight": {"shard": "missing.pt"},
                "model.layers.1.self_attn.q_proj.weight": {"shard": second_shard.name},
            },
            "group_diagnostics": [first_diagnostic, second_diagnostic],
        },
    )

    payload, completed_groups = load_resume_rank_manifest(
        tmp_path,
        0,
        [first_group, second_group],
    )

    assert completed_groups == {1}
    assert payload["modules"] == {
        "model.layers.1.self_attn.q_proj.weight": {"shard": second_shard.name}
    }
    assert payload["group_diagnostics"] == [second_diagnostic]


def test_resume_without_rank_manifest_starts_clean(tmp_path):
    payload, completed_groups = load_resume_rank_manifest(
        tmp_path,
        0,
        [_group("model.layers.0.self_attn.q_proj")],
    )

    assert payload == {"rank": 0, "modules": {}, "group_diagnostics": []}
    assert completed_groups == set()


def test_resume_rejects_rank_assignment_changes(tmp_path):
    _write_manifest(
        tmp_path,
        {
            "rank": 0,
            "modules": {
                "model.layers.7.self_attn.q_proj.weight": {"shard": "old.pt"},
            },
            "group_diagnostics": [],
        },
    )

    with pytest.raises(ValueError, match="not assigned to rank 0"):
        load_resume_rank_manifest(
            tmp_path,
            0,
            [_group("model.layers.0.self_attn.q_proj")],
        )
