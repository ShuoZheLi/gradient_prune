from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from response_analysis.judge_strategy_diversity import (
    call_openai,
    canonical_cluster_signature,
    cache_path,
    parse_json_content,
    validate_judgment,
)


def valid_judgment():
    return {
        "responses": [
            {"response_id": "r0", "normalized_answer": "1", "strategy_label": "direct algebra", "cluster_id": 0, "valid_reasoning": True},
            {"response_id": "r1", "normalized_answer": "1", "strategy_label": "direct algebra", "cluster_id": 0, "valid_reasoning": True},
        ],
        "clusters": [{"cluster_id": 0, "description": "direct algebra"}],
    }


def test_cached_api_results_are_reused(tmp_path: Path):
    request_payload = {
        "model": "judge",
        "messages": [{"role": "user", "content": "x"}],
        "expected_response_ids": ["r0", "r1"],
    }
    cache_file = cache_path(tmp_path, request_payload)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"request": request_payload, "judgment": valid_judgment()}), encoding="utf-8")

    class Args:
        cache_dir = tmp_path
        disable_api = True
        api_key = None
        base_url = None
        model = "judge"
        temperature = 0.0
        json_mode = "schema"
        max_retries = 1

    assert call_openai(request_payload, Args()) == valid_judgment()


def test_malformed_evaluator_json_is_rejected():
    malformed = {"responses": [{"response_id": "r0", "cluster_id": "bad"}], "clusters": []}
    with pytest.raises(ValueError):
        validate_judgment(malformed, {"r0"})


def test_nullable_text_fields_are_normalized_for_invalid_responses():
    judgment = {
        "responses": [
            {"response_id": "r0", "normalized_answer": None, "strategy_label": None, "cluster_id": 0, "valid_reasoning": False}
        ],
        "clusters": [{"cluster_id": 0, "description": "invalid or incoherent"}],
    }
    validated = validate_judgment(judgment, {"r0"})
    assert validated["responses"][0]["normalized_answer"] == ""
    assert validated["responses"][0]["strategy_label"] == ""


def test_malformed_evaluator_json_is_retried_and_validated(tmp_path: Path, monkeypatch):
    calls = {"count": 0}

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, content: str):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return FakeResponse('{"responses": [], "clusters": []}')
            return FakeResponse(json.dumps(valid_judgment()))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr("response_analysis.judge_strategy_diversity.time.sleep", lambda _: None)

    class Args:
        cache_dir = tmp_path
        disable_api = False
        api_key = "test"
        base_url = None
        model = "judge"
        temperature = 0.0
        json_mode = "object"
        max_retries = 2

    request_payload = {
        "model": "judge",
        "messages": [{"role": "user", "content": "x"}],
        "expected_response_ids": ["r0", "r1"],
    }
    assert call_openai(request_payload, Args()) == valid_judgment()
    assert calls["count"] == 2


def test_unsupported_temperature_is_retried_without_temperature(tmp_path: Path, monkeypatch):
    calls = []

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, content: str):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported value: 'temperature' does not support 0 with this model")
            return FakeResponse(json.dumps(valid_judgment()))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr("response_analysis.judge_strategy_diversity.time.sleep", lambda _: None)

    class Args:
        cache_dir = tmp_path
        disable_api = False
        api_key = "test"
        base_url = None
        model = "judge"
        temperature = 0.0
        json_mode = "object"
        max_retries = 2

    request_payload = {
        "model": "judge",
        "messages": [{"role": "user", "content": "x"}],
        "expected_response_ids": ["r0", "r1"],
    }
    assert call_openai(request_payload, Args()) == valid_judgment()
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_markdown_json_block_is_parsed_and_validated():
    content = "```json\n" + json.dumps(valid_judgment()) + "\n```"
    parsed = parse_json_content(content)
    assert validate_judgment(parsed, {"r0", "r1"}) == valid_judgment()


def test_cluster_signature_ignores_cluster_description_wording():
    first = valid_judgment()
    second = {
        "responses": [
            {"response_id": "r1", "normalized_answer": "one", "strategy_label": "same", "cluster_id": 7, "valid_reasoning": True},
            {"response_id": "r0", "normalized_answer": "one", "strategy_label": "same", "cluster_id": 7, "valid_reasoning": True},
        ],
        "clusters": [{"cluster_id": 7, "description": "different wording"}],
    }
    assert canonical_cluster_signature(first) == canonical_cluster_signature(second)
