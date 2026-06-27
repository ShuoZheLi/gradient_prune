from types import SimpleNamespace

import experiment_runner
from config import load_config


def test_calibration_ce_config_is_separate_from_stat_microbatch_size():
    cfg = load_config("configs/qwen25_1p5b_math.yaml")
    assert cfg.calibration.microbatch_size == 32
    assert cfg.calibration_ce.batch_size == 32
    assert cfg.heldout_ce.batch_size == 32


def test_evaluate_all_uses_calibration_ce_batch_size(monkeypatch):
    calls = []

    def fake_evaluate_ce(*args, **kwargs):
        calls.append(kwargs)
        return {"ce": 1.0, "perplexity": 1.0, "num_tokens": 1, "num_examples": 1}

    monkeypatch.setattr(experiment_runner, "evaluate_ce", fake_evaluate_ce)
    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        calibration=SimpleNamespace(
            path="stats-path",
            type="prompt_response",
            only_correct=True,
            loss_on="response_only",
            max_samples=10,
            microbatch_size=99,
            max_length=100,
            text_key=None,
            prompt_key="prompt",
            response_key="response",
        ),
        calibration_ce=SimpleNamespace(
            enabled=True,
            backend="transformers",
            path="ce-path",
            type="prompt_response",
            only_correct=False,
            loss_on="full_trajectory",
            max_samples=7,
            batch_size=5,
            max_length=80,
            text_key=None,
            prompt_key="prompt",
            response_key="response",
        ),
        heldout_ce=SimpleNamespace(
            enabled=True,
            backend="transformers",
            path="held-path",
            loss_on="response_only",
            max_samples=6,
            batch_size=4,
            max_length=70,
            text_key=None,
            prompt_key="prompt",
            response_key="response",
        ),
        text_ppl=SimpleNamespace(enabled=False),
        task_accuracy=SimpleNamespace(enabled=False, dataset_path=None),
    )

    metrics = experiment_runner._evaluate_all(None, None, cfg, root="unused", method="m", sparsity=0.0, lambda_value=None)

    assert metrics["calibration_ce"] == 1.0
    assert calls[0]["path"] == "ce-path"
    assert calls[0]["batch_size"] == 5
    assert calls[0]["max_samples"] == 7
    assert calls[0]["loss_on"] == "full_trajectory"
    assert calls[0]["only_correct"] is False
    assert calls[1]["path"] == "held-path"
    assert calls[1]["batch_size"] == 4


def test_evaluate_all_can_disable_calibration_and_heldout_ce(monkeypatch):
    def fail_evaluate_ce(*args, **kwargs):
        raise AssertionError("evaluate_ce should not be called when CE evals are disabled")

    monkeypatch.setattr(experiment_runner, "evaluate_ce", fail_evaluate_ce)
    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        calibration=SimpleNamespace(path="stats-path", type="prompt_response", only_correct=True, loss_on="response_only", max_samples=10, microbatch_size=99, max_length=100, text_key=None, prompt_key="prompt", response_key="response"),
        calibration_ce=SimpleNamespace(enabled=False, backend="transformers"),
        heldout_ce=SimpleNamespace(enabled=False, backend="transformers", path="held-path"),
        text_ppl=SimpleNamespace(enabled=False),
        task_accuracy=SimpleNamespace(enabled=False, dataset_path=None),
    )

    metrics = experiment_runner._evaluate_all(None, None, cfg, root="unused", method="m", sparsity=0.0, lambda_value=None)
    assert metrics["calibration_ce"] is None
    assert metrics["heldout_ce"] is None


def test_evaluate_all_uses_vllm_ce_backend(monkeypatch, tmp_path):
    calls = []

    def fake_save(model, tokenizer, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}")
        (output_dir / "model.safetensors").write_text("stub")

    def fake_vllm_ce(**kwargs):
        calls.append(kwargs)
        return {"ce": 2.0, "perplexity": 1.0, "num_tokens": 2, "num_examples": 1}

    monkeypatch.setattr(experiment_runner, "save_pruned_model", fake_save)
    monkeypatch.setattr(experiment_runner, "evaluate_ce_vllm", fake_vllm_ce)
    cfg = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        calibration=SimpleNamespace(path="stats-path", type="prompt_response", only_correct=True, loss_on="response_only", max_samples=10, microbatch_size=99, max_length=100, text_key=None, prompt_key="prompt", response_key="response"),
        calibration_ce=SimpleNamespace(enabled=True, backend="vllm", path="ce-path", type="prompt_response", only_correct=True, loss_on="response_only", max_samples=5, batch_size=3, data_parallel_size=4, tensor_parallel_size=1, gpu_memory_utilization=0.8, dtype="auto", enforce_eager=True, trust_remote_code=False, max_length=77, text_key=None, prompt_key="prompt", response_key="response"),
        heldout_ce=SimpleNamespace(enabled=False, backend="transformers", path=None),
        text_ppl=SimpleNamespace(enabled=False),
        task_accuracy=SimpleNamespace(enabled=False, dataset_path=None, backend="transformers"),
    )

    metrics = experiment_runner._evaluate_all(None, None, cfg, root=tmp_path, method="m", sparsity=0.5, lambda_value=None)

    assert metrics["calibration_ce"] == 2.0
    assert calls[0]["model_path"] == tmp_path / "eval_models" / "method=m" / "sparsity=0.5" / "lambda=none"
    assert calls[0]["data_parallel_size"] == 4
    assert calls[0]["batch_size"] == 3
