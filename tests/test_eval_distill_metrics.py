import json

from eval_distill_metrics import aggregate_eval_metrics, count_response_tokens


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_count_response_tokens_uses_tokenizer_not_characters():
    tokenizer = WhitespaceTokenizer()
    assert count_response_tokens("one two three", tokenizer) == 3
    assert len("one two three") == 13


def test_aggregate_eval_metrics_reports_token_lengths(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "metrics_shard_0.json").write_text(
        json.dumps({"num_examples": 2, "num_generations": 3, "num_scored": 3, "num_unscored": 0}),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"
    rows = [
        {"example_id": 0, "response": "one two three", "task_score": 1.0, "is_correct": True},
        {"example_id": 1, "response": "four five", "task_score": 0.0, "is_correct": False},
        {"example_id": 1, "response": "six", "task_score": None},
    ]
    (tmp_path / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    metrics = aggregate_eval_metrics(
        metrics_path=metrics_path,
        shard_dir=shard_dir,
        max_examples=2,
        num_responses=1,
        tokenizer=WhitespaceTokenizer(),
    )

    assert metrics["avg_response_length_tokens"] == 2.0
    assert metrics["avg_correct_response_length_tokens"] == 3.0
    assert metrics["avg_wrong_response_length_tokens"] == 2.0
    assert "avg_response_length_chars" not in metrics
    assert metrics["accuracy"] == 0.5
