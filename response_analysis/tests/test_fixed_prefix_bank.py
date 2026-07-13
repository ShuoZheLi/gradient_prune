from __future__ import annotations

import json
from pathlib import Path

from response_analysis.io_utils import read_jsonl, write_jsonl


def test_fixed_prefix_records_are_model_independent(tmp_path: Path):
    bank = [
        {"prompt_id": 0, "prompt": "p0", "prefix_id": 0, "prefix_token_ids": [1, 2, 3]},
        {"prompt_id": 1, "prompt": "p1", "prefix_id": 1, "prefix_token_ids": [4, 5]},
    ]
    path = tmp_path / "fixed_prefix_bank.jsonl"
    write_jsonl(path, bank)
    loaded_a = read_jsonl(path)
    loaded_b = read_jsonl(path)
    assert [row["prefix_token_ids"] for row in loaded_a] == [row["prefix_token_ids"] for row in loaded_b]
    assert all("model_id" not in row for row in loaded_a)
