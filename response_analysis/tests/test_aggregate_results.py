from __future__ import annotations

import pandas as pd
import pytest

from response_analysis import aggregate_results
from response_analysis.aggregate_results import paired_comparisons


def test_paired_comparisons_handles_boolean_metrics() -> None:
    per_prompt = pd.DataFrame(
        {
            "model_id": ["base", "base", "pruned", "pruned"],
            "prompt_id": [1, 2, 1, 2],
            "accuracy": [0.0, 1.0, 1.0, 1.0],
            "order_inconsistent_across_repeats": [False, True, True, True],
        }
    )

    paired = paired_comparisons(per_prompt, "base", samples=25, seed=0)

    bool_row = paired[paired["metric"] == "order_inconsistent_across_repeats"].iloc[0]
    assert bool_row["mean_difference"] == pytest.approx(0.5)
    assert bool_row["num_prompts"] == 2


def test_aggregate_outputs_pass_and_avg_at_k(tmp_path, monkeypatch) -> None:
    response_metrics = tmp_path / "response_metrics.parquet"
    per_prompt_output = tmp_path / "per_prompt_metrics.csv"
    aggregate_output = tmp_path / "aggregate_metrics.csv"
    paired_output = tmp_path / "paired_comparisons.csv"
    figures_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "model_id": ["dense", "dense"],
            "prompt_id": [0, 1],
            "pass_at_1": [0.0, 1.0],
            "pass_at_k": [1.0, 1.0],
            "avg_at_k": [0.5, 0.75],
            "accuracy": [0.5, 0.75],
        }
    ).to_parquet(response_metrics, index=False)
    monkeypatch.setattr(aggregate_results, "write_figures", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_results.py",
            "--response_metrics",
            str(response_metrics),
            "--per_prompt_output",
            str(per_prompt_output),
            "--aggregate_output",
            str(aggregate_output),
            "--paired_output",
            str(paired_output),
            "--figures_dir",
            str(figures_dir),
        ],
    )

    aggregate_results.main()

    aggregate = pd.read_csv(aggregate_output)
    assert list(aggregate.columns[:5]) == ["model_id", "pass_at_1", "pass_at_k", "avg_at_k", "accuracy"]
    assert aggregate.loc[0, "pass_at_1"] == pytest.approx(0.5)
    assert aggregate.loc[0, "pass_at_k"] == pytest.approx(1.0)
    assert aggregate.loc[0, "avg_at_k"] == pytest.approx(0.625)
    assert aggregate.loc[0, "accuracy"] == pytest.approx(0.625)
