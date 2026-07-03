import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from task_scoring import compute_score_with_reward_module, extract_data_source, score_task_response


def _install_fake_verl(monkeypatch, default_compute_score=None, math_verify_score=None):
    verl_module = types.ModuleType("verl")
    utils_module = types.ModuleType("verl.utils")
    reward_score_module = types.ModuleType("verl.utils.reward_score")

    if default_compute_score is not None:
        reward_score_module.default_compute_score = default_compute_score

    monkeypatch.setitem(sys.modules, "verl", verl_module)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_module)
    monkeypatch.setitem(sys.modules, "verl.utils.reward_score", reward_score_module)

    if math_verify_score is not None:
        math_verify_module = types.ModuleType("verl.utils.reward_score.math_verify")
        math_verify_module.compute_score = math_verify_score
        monkeypatch.setitem(sys.modules, "verl.utils.reward_score.math_verify", math_verify_module)


def test_local_metamath_paths_infer_math_500_data_source():
    assert extract_data_source({}, "/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet") == "math_500"
    assert extract_data_source({}, "/data/shuozhe/saved_dataset/MetaMathQA-math-500/math7500.parquet") == "math_500"


def test_explicit_data_source_column_takes_precedence():
    assert extract_data_source({"data_source": "custom_math"}, "/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet") == "custom_math"


def test_default_scoring_uses_verl_math_reward_compute_score(monkeypatch):
    calls = []

    def fake_default_compute_score(data_source, solution_str, ground_truth, **kwargs):
        calls.append((data_source, solution_str, ground_truth, kwargs))
        return 1.0

    _install_fake_verl(monkeypatch, default_compute_score=fake_default_compute_score)
    monkeypatch.delenv("TASK_SCORER_BACKEND", raising=False)
    monkeypatch.delenv("MATH_SCORER", raising=False)

    assert compute_score_with_reward_module("math_500", "Answer: \\boxed{2}", "2") == 1.0
    assert calls == [("math_500", "Answer: \\boxed{2}", "2", {"math_dapo_binary_reward": True})]


def test_verl_default_passes_reference_math_dapo_binary_reward(monkeypatch):
    calls = []

    def fake_default_compute_score(data_source, solution_str, ground_truth, **kwargs):
        calls.append((data_source, kwargs))
        return 1.0

    _install_fake_verl(monkeypatch, default_compute_score=fake_default_compute_score)
    monkeypatch.setenv("TASK_SCORER_BACKEND", "verl_math_reward")
    monkeypatch.delenv("MATH_DAPO_BINARY_REWARD", raising=False)

    assert compute_score_with_reward_module("math", "Answer: 1", "1") == 1.0
    assert calls == [("math", {"math_dapo_binary_reward": True})]


def test_verl_default_aliases_reference_math_reward_backend(monkeypatch):
    calls = []

    def fake_default_compute_score(data_source, solution_str, ground_truth, **kwargs):
        calls.append((data_source, solution_str, ground_truth, kwargs))
        return 1.0

    _install_fake_verl(monkeypatch, default_compute_score=fake_default_compute_score)
    monkeypatch.setenv("TASK_SCORER_BACKEND", "verl_default")

    assert compute_score_with_reward_module("lighteval/MATH", "Answer: \\boxed{2}", "2") == 1.0
    assert calls == [("lighteval/MATH", "Answer: \\boxed{2}", "2", {"math_dapo_binary_reward": True})]


def test_score_task_response_correctness_comes_from_verl(monkeypatch):
    def fake_default_compute_score(data_source, solution_str, ground_truth, **kwargs):
        return float(data_source == "math_500" and solution_str.endswith("\\boxed{7}") and ground_truth == "7")

    _install_fake_verl(monkeypatch, default_compute_score=fake_default_compute_score)
    monkeypatch.setenv("TASK_SCORER_BACKEND", "verl_math_reward")

    assert score_task_response("Reasoning... \\boxed{7}", "7", data_source="math_500") == (1.0, True)
    assert score_task_response("Reasoning... \\boxed{8}", "7", data_source="math_500") == (0.0, False)


def test_verl_math_verify_backend_uses_verl_math_verify_for_math(monkeypatch):
    calls = []

    def fake_default_compute_score(data_source, solution_str, ground_truth, **kwargs):
        raise AssertionError("math data should use math_verify backend")

    def fake_math_verify_score(model_output, ground_truth):
        calls.append((model_output, ground_truth))
        return 1.0

    _install_fake_verl(
        monkeypatch,
        default_compute_score=fake_default_compute_score,
        math_verify_score=fake_math_verify_score,
    )
    monkeypatch.setenv("TASK_SCORER_BACKEND", "verl_math_verify")

    assert compute_score_with_reward_module("math_500", "x=\\boxed{1/2}", "0.5") == 1.0
    assert calls == [("x=\\boxed{1/2}", "0.5")]


def test_missing_verl_fails_fast_with_actionable_message(monkeypatch):
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "verl.utils.reward_score":
            raise ImportError("simulated missing verl")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setenv("TASK_SCORER_BACKEND", "verl_math_reward")

    with pytest.raises(ImportError, match="Activate the verl environment"):
        compute_score_with_reward_module("math_500", "\\boxed{1}", "1")
