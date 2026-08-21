from __future__ import annotations

import torch

from recoverability_pruning.convergence import _sample_mapping_pair, compare_score_files


def test_convergence_reports_global_layer_and_matrix_spearman(tmp_path):
    first_path = tmp_path / "scores_probe_0002.pt"
    second_path = tmp_path / "scores_probe_0004.pt"
    first = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "model.layers.1.mlp.up_proj.weight": torch.tensor([[4.0, 1.0], [2.0, 3.0]]),
    }
    second = {name: value * 3.0 + 2.0 for name, value in first.items()}
    torch.save({"num_probes": 2, "scores": first}, first_path)
    torch.save({"num_probes": 4, "scores": second}, second_path)
    result = compare_score_files(first_path, second_path, max_elements_per_matrix=100)
    assert result["global"]["spearman"] == 1.0
    assert set(result["per_layer"]) == {"model.layers.0", "model.layers.1"}
    assert all(entry["spearman"] == 1.0 for entry in result["per_matrix"].values())


def test_aggregate_sampling_is_uniform_over_weight_coordinates():
    first = {
        "small": torch.zeros(1_000),
        "large": torch.ones(9_000),
    }
    second = {name: value.clone() for name, value in first.items()}
    sampled_first, sampled_second, exact = _sample_mapping_pair(
        first,
        second,
        ["small", "large"],
        max_elements=5_000,
        seed=123,
    )
    assert exact is False
    assert 0.88 < sampled_first.mean() < 0.92
    assert (sampled_first == sampled_second).all()
