from __future__ import annotations

from response_analysis import generate_responses


def test_score_response_unpacks_task_scoring_tuple(monkeypatch):
    monkeypatch.setattr(generate_responses, "score_task_response", lambda response, ground_truth, data_source="": (1.0, True))
    assert generate_responses.score_response("math", "Answer: 2", "2") == 1.0


def test_score_response_falls_back_to_zero_on_scoring_exception(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("bad scorer")

    monkeypatch.setattr(generate_responses, "score_task_response", raise_error)
    assert generate_responses.score_response("math", "Answer: 2", "2") == 0.0
