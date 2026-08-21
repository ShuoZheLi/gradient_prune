from __future__ import annotations

import pandas as pd
import pytest

from recoverability_pruning.data import assert_disjoint_datasets, load_trajectory_dataset
from recoverability_pruning.losses import IGNORE_INDEX


def _write_parquet(tmp_path, name, records):
    path = tmp_path / name
    pd.DataFrame(records).to_parquet(path)
    return path


def test_full_trajectory_uses_provided_token_ids(tmp_path):
    path = _write_parquet(tmp_path, "full.parquet", [{"prompt_generated_trajectory_ids": [1, 2, 3, 4]}])
    dataset = load_trajectory_dataset(path, loss_on="full_trajectory", max_length=3, truncation_side="right")
    assert dataset.examples[0].input_ids == (1, 2, 3)
    assert dataset.examples[0].labels == (1, 2, 3)


def test_response_only_refuses_to_infer_prompt_boundary(tmp_path):
    path = _write_parquet(tmp_path, "missing_boundary.parquet", [{"prompt_generated_trajectory_ids": [1, 2, 3]}])
    with pytest.raises(ValueError, match="will not be inferred"):
        load_trajectory_dataset(path, loss_on="response_only")


def test_response_only_and_loss_mask_are_explicit(tmp_path):
    response_path = _write_parquet(
        tmp_path,
        "response.parquet",
        [{"prompt_generated_trajectory_ids": [1, 2, 3, 4], "prompt_length": 2}],
    )
    response_dataset = load_trajectory_dataset(response_path, loss_on="response_only")
    assert response_dataset.examples[0].labels == (IGNORE_INDEX, IGNORE_INDEX, 3, 4)

    mask_path = _write_parquet(
        tmp_path,
        "mask.parquet",
        [{"prompt_generated_trajectory_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}],
    )
    mask_dataset = load_trajectory_dataset(mask_path, loss_on="loss_mask")
    assert mask_dataset.examples[0].labels == (IGNORE_INDEX, 2, IGNORE_INDEX, 4)


def test_disjoint_check_uses_exact_full_token_trajectory(tmp_path):
    first_path = _write_parquet(tmp_path, "first.parquet", [{"prompt_generated_trajectory_ids": [1, 2, 3]}])
    same_path = _write_parquet(tmp_path, "same.parquet", [{"prompt_generated_trajectory_ids": [1, 2, 3]}])
    other_path = _write_parquet(tmp_path, "other.parquet", [{"prompt_generated_trajectory_ids": [1, 2, 4]}])
    first = load_trajectory_dataset(first_path, loss_on="full_trajectory")
    same = load_trajectory_dataset(same_path, loss_on="full_trajectory")
    other = load_trajectory_dataset(other_path, loss_on="full_trajectory")
    with pytest.raises(ValueError, match="not disjoint"):
        assert_disjoint_datasets(first, same)
    assert_disjoint_datasets(first, other)


def test_disjoint_check_rejects_same_prompt_with_different_trajectory(tmp_path):
    first_path = _write_parquet(
        tmp_path,
        "first_prompt.parquet",
        [{"prompt": "same prompt", "prompt_generated_trajectory_ids": [1, 2, 3]}],
    )
    second_path = _write_parquet(
        tmp_path,
        "second_prompt.parquet",
        [{"prompt": "same prompt", "prompt_generated_trajectory_ids": [1, 2, 4]}],
    )
    first = load_trajectory_dataset(first_path, loss_on="full_trajectory")
    second = load_trajectory_dataset(second_path, loss_on="full_trajectory")
    with pytest.raises(ValueError, match="explicit disjoint keys"):
        assert_disjoint_datasets(first, second)
