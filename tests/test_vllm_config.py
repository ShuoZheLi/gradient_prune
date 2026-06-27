import sys
import types

import pandas as pd

from config import load_config
from evaluate_accuracy import evaluate_task_accuracy


class FakeOutputText:
    text = "The answer is \\boxed{4}."


class FakeRequestOutput:
    outputs = [FakeOutputText()]


def test_main_config_uses_four_one_gpu_vllm_workers():
    cfg = load_config("configs/qwen25_1p5b_math.yaml")
    assert cfg.task_accuracy.backend == "vllm"
    assert cfg.task_accuracy.data_parallel_size == 4
    assert cfg.task_accuracy.tensor_parallel_size == 1
    assert cfg.task_accuracy.batch_size == 32
    assert cfg.task_accuracy.gpu_memory_utilization == 0.9


def test_vllm_kwargs_are_forwarded(monkeypatch, tmp_path):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, prompts, sampling, use_tqdm=False):
            captured.setdefault("generate_calls", []).append(list(prompts))
            captured["prompts"] = prompts
            captured["sampling"] = sampling
            captured["use_tqdm"] = use_tqdm
            return [FakeRequestOutput() for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    data_path = tmp_path / "eval.parquet"
    pd.DataFrame([{"prompt": "2+2?", "answer": "\\boxed{4}"}]).to_parquet(data_path)

    metrics = evaluate_task_accuracy(
        tmp_path / "model",
        tokenizer=None,
        dataset_path=str(data_path),
        backend="vllm",
        output_jsonl=tmp_path / "responses.jsonl",
        metrics_json=tmp_path / "metrics.json",
        prompt_key="prompt",
        response_key=None,
        reward_score_dir=None,
        max_examples=1,
        max_prompt_length=128,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        batch_size=1,
        seed=123,
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        dtype="auto",
        enforce_eager=True,
        trust_remote_code=False,
    )

    assert metrics["accuracy"] == 1.0
    assert captured["model"] == str(tmp_path / "model")
    assert captured["tensor_parallel_size"] == 1
    assert captured["gpu_memory_utilization"] == 0.75
    assert captured["dtype"] == "auto"
    assert captured["enforce_eager"] is True
    assert captured["max_model_len"] == 144
    assert captured["sampling"].kwargs["seed"] == 123
    assert captured["use_tqdm"] is False
    assert captured["generate_calls"] == [["2+2?"]]

from evaluate_accuracy import _split_round_robin


def test_round_robin_sharding_balances_examples():
    shards = _split_round_robin([{"example_id": i} for i in range(10)], 4)
    assert [[item["example_id"] for item in shard] for shard in shards] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6],
        [3, 7],
    ]


def test_vllm_empty_dataset_writes_empty_metrics(tmp_path):
    data_path = tmp_path / "empty.parquet"
    pd.DataFrame([], columns=["prompt", "answer"]).to_parquet(data_path)
    metrics = evaluate_task_accuracy(
        tmp_path / "model",
        dataset_path=str(data_path),
        backend="vllm",
        output_jsonl=tmp_path / "responses.jsonl",
        metrics_json=tmp_path / "metrics.json",
        max_examples=0,
        data_parallel_size=4,
    )
    assert metrics["num_examples"] == 0
    assert metrics["num_correct"] == 0
    assert (tmp_path / "responses.jsonl").read_text() == ""

from experiment_runner import _hf_checkpoint_exists


def test_hf_checkpoint_exists_requires_config_and_weights(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert not _hf_checkpoint_exists(model_dir)
    (model_dir / "config.json").write_text("{}")
    assert not _hf_checkpoint_exists(model_dir)
    (model_dir / "model.safetensors").write_text("stub")
    assert _hf_checkpoint_exists(model_dir)
