from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from response_analysis.io_utils import write_jsonl
from response_analysis.process_existing_results import (
    condition_model_id,
    discover_inputs,
    fallback_judgment_for_request,
    normalize_entropy_file,
    normalize_generation_file,
    parse_condition_file,
    request_payload_for_group,
    semantic_metrics_from_judgments,
    sparsity_label_to_float,
)


def test_sparsity_label_parsing_and_model_id() -> None:
    assert sparsity_label_to_float("0") == pytest.approx(0.0)
    assert sparsity_label_to_float("0d1") == pytest.approx(0.1)
    assert sparsity_label_to_float("0d25") == pytest.approx(0.25)
    assert condition_model_id("qwen3_8b_wanda", "0d3") == "qwen3_8b_wanda_s0d3"


def test_parse_and_discover_downloaded_result_files(tmp_path: Path) -> None:
    generation = tmp_path / "slurm-828229_qwen3_8b_resp_analysis_sparsity_0d1_generation.jsonl"
    on_policy = tmp_path / "slurm-828229_qwen3_8b_resp_analysis_sparsity_0d1_on_policy_entropy.parquet"
    fixed = tmp_path / "slurm-828229_qwen3_8b_resp_analysis_sparsity_0d1_fixed_prefix_entropy.parquet"
    ignored = tmp_path / "slurm-828229_qwen3_8b_resp_analysis_sparsity_0d1.out"
    generation.write_text("", encoding="utf-8")
    on_policy.write_bytes(b"")
    fixed.write_bytes(b"")
    ignored.write_text("", encoding="utf-8")

    info = parse_condition_file(generation)
    assert info is not None
    assert info["job_id"] == "828229"
    assert info["sparsity_label"] == "0d1"
    assert info["sparsity"] == pytest.approx(0.1)
    assert info["kind"] == "generation"

    discovered = discover_inputs(tmp_path)
    assert list(discovered) == ["slurm-828229_sparsity_0d1"]
    condition = discovered["slurm-828229_sparsity_0d1"]
    assert condition["generation"] == generation
    assert condition["on_policy_entropy"] == on_policy
    assert condition["fixed_prefix_entropy"] == fixed


def test_normalize_generation_preserves_original_fields(tmp_path: Path) -> None:
    generation = tmp_path / "slurm-1_qwen3_8b_resp_analysis_sparsity_0d2_generation.jsonl"
    rows = [
        {
            "model_id": "qwen3_8b_dense",
            "pruning_sparsity": 0.0,
            "prompt_id": 7,
            "sample_id": 0,
            "generated_text": "A",
            "parsed_final_answer": "1",
            "correctness": True,
            "response_length": 3,
        }
    ]
    write_jsonl(generation, rows)
    condition = {
        "generation": generation,
        "condition_key": "slurm-1_sparsity_0d2",
        "sparsity_label": "0d2",
        "sparsity": 0.2,
    }
    output = tmp_path / "normalized.jsonl"

    summary = normalize_generation_file(
        condition,
        output,
        model_prefix="qwen3_8b_wanda",
        preserve_record_model_id=False,
        allow_metadata_mismatch=True,
    )
    normalized = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["generation_rows"] == 1
    assert normalized[0]["model_id"] == "qwen3_8b_wanda_s0d2"
    assert normalized[0]["pruning_sparsity"] == pytest.approx(0.2)
    assert normalized[0]["original_model_id"] == "qwen3_8b_dense"
    assert normalized[0]["original_pruning_sparsity"] == pytest.approx(0.0)
    assert normalized[0]["condition_sparsity_label"] == "0d2"


def test_normalize_entropy_overrides_condition_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "slurm-2_qwen3_8b_resp_analysis_sparsity_0d3_on_policy_entropy.parquet"
    pd.DataFrame(
        [
            {
                "model_id": "qwen3_8b_dense",
                "prompt_id": 1,
                "sample_id": 0,
                "pruning_sparsity": 0.0,
                "token_entropy_mean": 1.5,
            }
        ]
    ).to_parquet(input_path, index=False)
    condition = {"condition_key": "slurm-2_sparsity_0d3", "sparsity_label": "0d3", "sparsity": 0.3}
    output_path = tmp_path / "token_metrics.parquet"

    result = normalize_entropy_file(
        input_path,
        output_path,
        condition,
        model_prefix="qwen3_8b_wanda",
        preserve_record_model_id=False,
        allow_metadata_mismatch=True,
    )
    df = pd.read_parquet(result)

    assert df.loc[0, "model_id"] == "qwen3_8b_wanda_s0d3"
    assert df.loc[0, "pruning_sparsity"] == pytest.approx(0.3)
    assert df.loc[0, "original_model_id"] == "qwen3_8b_dense"
    assert df.loc[0, "original_pruning_sparsity"] == pytest.approx(0.0)


def test_nonzero_condition_rejects_dense_generation_metadata(tmp_path: Path) -> None:
    generation = tmp_path / "slurm-1_qwen3_8b_resp_analysis_sparsity_0d5_generation.jsonl"
    write_jsonl(
        generation,
        [
            {
                "model_id": "qwen3_8b_dense",
                "pruning_sparsity": 0.0,
                "prompt_id": 0,
                "sample_id": 0,
                "generated_text": "A",
            }
        ],
    )
    condition = {
        "generation": generation,
        "condition_key": "slurm-1_sparsity_0d5",
        "sparsity_label": "0d5",
        "sparsity": 0.5,
    }

    with pytest.raises(ValueError, match="Refusing to relabel dense/zero-sparsity records"):
        normalize_generation_file(condition, tmp_path / "normalized.jsonl", model_prefix="qwen3_8b_wanda", preserve_record_model_id=False)


def test_nonzero_condition_rejects_dense_entropy_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "slurm-2_qwen3_8b_resp_analysis_sparsity_0d5_on_policy_entropy.parquet"
    pd.DataFrame([{"model_id": "qwen3_8b_dense", "prompt_id": 0, "sample_id": 0, "pruning_sparsity": 0.0}]).to_parquet(input_path, index=False)
    condition = {"condition_key": "slurm-2_sparsity_0d5", "sparsity_label": "0d5", "sparsity": 0.5}

    with pytest.raises(ValueError, match="Refusing to relabel dense/zero-sparsity records"):
        normalize_entropy_file(input_path, tmp_path / "token_metrics.parquet", condition, model_prefix="qwen3_8b_wanda", preserve_record_model_id=False)


def test_request_payload_shuffle_keeps_response_ids() -> None:
    group = [
        {"sample_id": idx, "prompt": "2+2?", "ground_truth": "4", "generated_text": str(idx)}
        for idx in range(4)
    ]
    args = SimpleNamespace(judge_seed=123, judge_model="judge")
    payload = request_payload_for_group(group, "model", 5, 1, args)

    assert payload["model_id"] == "model"
    assert payload["prompt_id"] == 5
    assert payload["expected_response_ids"] == ["r0", "r1", "r2", "r3"]
    assert sorted(payload["order"]) == [0, 1, 2, 3]


def test_fallback_judgment_preserves_all_response_ids() -> None:
    request_payload = {
        "model_id": "m",
        "prompt_id": 0,
        "repeat": 1,
        "order": [1, 0],
        "expected_response_ids": ["r0", "r1"],
    }

    fallback = fallback_judgment_for_request(request_payload, RuntimeError("missing response ids"))

    assert fallback["judge_failed"] is True
    assert fallback["model_id"] == "m"
    assert [row["response_id"] for row in fallback["judgment"]["responses"]] == ["r0", "r1"]
    assert all(row["cluster_id"] == 0 for row in fallback["judgment"]["responses"])
    assert all(row["valid_reasoning"] is False for row in fallback["judgment"]["responses"])


def test_semantic_metrics_detects_label_invariant_consistency() -> None:
    records = [
        {"model_id": "m", "prompt_id": 0, "sample_id": 0, "correctness": True},
        {"model_id": "m", "prompt_id": 0, "sample_id": 1, "correctness": False},
    ]
    judgments = [
        {
            "model_id": "m",
            "prompt_id": 0,
            "repeat": 0,
            "judgment": {
                "responses": [
                    {"response_id": "r0", "cluster_id": 10, "valid_reasoning": True},
                    {"response_id": "r1", "cluster_id": 20, "valid_reasoning": True},
                ]
            },
        },
        {
            "model_id": "m",
            "prompt_id": 0,
            "repeat": 1,
            "judgment": {
                "responses": [
                    {"response_id": "r0", "cluster_id": 0, "valid_reasoning": True},
                    {"response_id": "r1", "cluster_id": 1, "valid_reasoning": True},
                ]
            },
        },
    ]

    metrics = semantic_metrics_from_judgments(records, judgments)
    assert bool(metrics.loc[0, "order_inconsistent_across_repeats"]) is False
    assert metrics.loc[0, "num_strategy_clusters"] == pytest.approx(2.0)
    assert metrics.loc[0, "num_strategy_clusters_correct"] == pytest.approx(1.0)
    assert metrics.loc[0, "num_strategy_clusters_incorrect"] == pytest.approx(1.0)
