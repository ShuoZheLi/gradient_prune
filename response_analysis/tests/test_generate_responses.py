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


def test_vllm_dtype_mapping():
    assert generate_responses.resolve_vllm_dtype("auto") == "auto"
    assert generate_responses.resolve_vllm_dtype("bf16") == "bfloat16"
    assert generate_responses.resolve_vllm_dtype("fp16") == "float16"
    assert generate_responses.resolve_vllm_dtype("fp32") == "float32"


def test_vllm_sampling_kwargs_uses_single_completion_and_seed():
    class Args:
        temperature = 1.0
        top_p = 0.9
        top_k = 50
        max_new_tokens = 128

    kwargs = generate_responses.vllm_sampling_kwargs(Args(), seed=123)
    assert kwargs == {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "max_tokens": 128, "n": 1, "seed": 123}


def test_default_vllm_pruned_model_dir_uses_output_parent():
    class Args:
        output = "outputs/run/generations.jsonl"
        model_path = "/models/Qwen3-8B"
        pruning_sparsity = 0.5

    assert str(generate_responses.default_vllm_pruned_model_dir(Args())) == "outputs/run/vllm_pruned_Qwen3-8B_s0.5"
