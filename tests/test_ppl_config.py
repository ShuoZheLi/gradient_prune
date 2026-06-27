from types import SimpleNamespace

import experiment_runner
from config import load_config


def test_text_ppl_config_can_use_vllm_and_custom_dataset_fields():
    cfg = load_config("configs/qwen25_1p5b_math.yaml")
    assert cfg.text_ppl.enabled is True
    assert cfg.text_ppl.backend == "vllm"
    assert cfg.text_ppl.dataset_name == "wikitext"
    assert cfg.text_ppl.dataset_config == "wikitext-2-raw-v1"
    assert cfg.text_ppl.split == "validation"
    assert cfg.text_ppl.text_key == "text"
    assert cfg.text_ppl.batch_size == 32
    assert cfg.text_ppl.data_parallel_size == 4
    assert cfg.text_ppl.tensor_parallel_size == 1


def test_evaluate_all_uses_vllm_text_ppl_backend(monkeypatch, tmp_path):
    calls = []

    def fake_save(model, tokenizer, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}")
        (output_dir / "model.safetensors").write_text("stub")

    def fake_ppl(**kwargs):
        calls.append(kwargs)
        return {"ce": 0.5, "perplexity": 1.65, "num_tokens": 10, "num_examples": 2}

    monkeypatch.setattr(experiment_runner, "save_pruned_model", fake_save)
    monkeypatch.setattr(experiment_runner, "evaluate_text_ppl_vllm", fake_ppl)
    cfg = SimpleNamespace(
        seed=7,
        model=SimpleNamespace(device="cpu"),
        calibration=SimpleNamespace(path="stats-path", type="prompt_response", only_correct=True, loss_on="response_only", max_samples=10, microbatch_size=99, max_length=100, text_key=None, prompt_key="prompt", response_key="response"),
        calibration_ce=SimpleNamespace(enabled=False, backend="transformers"),
        heldout_ce=SimpleNamespace(enabled=False, backend="transformers", path=None),
        text_ppl=SimpleNamespace(enabled=True, backend="vllm", dataset_name="my_dataset", dataset_config=None, split="test", text_key="content", max_samples=3, batch_size=2, data_parallel_size=4, tensor_parallel_size=1, gpu_memory_utilization=0.8, dtype="auto", enforce_eager=True, trust_remote_code=False, max_length=99),
        task_accuracy=SimpleNamespace(enabled=False, dataset_path=None, backend="transformers"),
    )

    metrics = experiment_runner._evaluate_all(None, None, cfg, root=tmp_path, method="m", sparsity=0.0, lambda_value=None)

    assert metrics["wikitext_ppl"] == 1.65
    assert calls[0]["model_path"] == tmp_path / "eval_models" / "method=m" / "sparsity=0.0" / "lambda=none"
    assert calls[0]["dataset_name"] == "my_dataset"
    assert calls[0]["dataset_config"] is None
    assert calls[0]["split"] == "test"
    assert calls[0]["text_key"] == "content"
    assert calls[0]["data_parallel_size"] == 4
