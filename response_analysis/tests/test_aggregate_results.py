from __future__ import annotations

import pandas as pd
import pytest

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
