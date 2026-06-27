from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from calibration_loaders import CalibrationExample, load_calibration_examples, make_calibration_dataloader
from evaluate_accuracy import _split_round_robin
from model_utils import temporarily_disable_cache


def evaluate_ce(model, tokenizer, *, path: str, calibration_type: str = "prompt_response", loss_on: str = "response_only", max_samples: int | None = None, max_length: int = 4096, batch_size: int = 1, device: str | None = None, only_correct: bool = False, text_key: str | None = None, prompt_key: str = "prompt", response_key: str | None = "response") -> dict[str, float | int]:
    examples = load_calibration_examples(path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_samples, text_key=text_key, prompt_key=prompt_key, response_key=response_key)
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on=loss_on, microbatch_size=batch_size)
    return evaluate_ce_dataloader(model, dataloader, device=device)


def evaluate_ce_dataloader(model, dataloader, *, device: str | None = None) -> dict[str, float | int]:
    model.eval()
    if device is None:
        device = str(next(model.parameters()).device)
    total_nll = 0.0
    total_tokens = 0
    examples = 0
    with torch.no_grad(), temporarily_disable_cache(model):
        for batch in tqdm(dataloader, desc="CE", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            token_count = int((labels[:, 1:] != -100).sum().item())
            total_nll += float(outputs.loss.item()) * max(token_count, 1)
            total_tokens += token_count
            examples += labels.shape[0]
    return _ce_metrics(total_nll, total_tokens, examples)


def evaluate_ce_vllm(
    *,
    model_path: str | Path,
    path: str,
    calibration_type: str = "prompt_response",
    loss_on: str = "response_only",
    max_samples: int | None = None,
    max_length: int = 4096,
    batch_size: int = 32,
    only_correct: bool = False,
    text_key: str | None = None,
    prompt_key: str = "prompt",
    response_key: str | None = "response",
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    enforce_eager: bool = True,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> dict[str, float | int]:
    examples = load_calibration_examples(path, calibration_type=calibration_type, only_correct=only_correct, max_samples=max_samples, text_key=text_key, prompt_key=prompt_key, response_key=response_key)
    return evaluate_ce_vllm_examples(model_path=model_path, examples=examples, loss_on=loss_on, max_length=max_length, batch_size=batch_size, data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, enforce_eager=enforce_eager, trust_remote_code=trust_remote_code, seed=seed)


def evaluate_ce_vllm_examples(
    *,
    model_path: str | Path,
    examples: list[CalibrationExample],
    loss_on: str = "response_only",
    max_length: int = 4096,
    batch_size: int = 32,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    enforce_eager: bool = True,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> dict[str, float | int]:
    if not examples:
        return _ce_metrics(0.0, 0, 0)
    worker_count = min(max(int(data_parallel_size), 1), len(examples))
    tensor_parallel_size = max(int(tensor_parallel_size), 1)
    if worker_count == 1:
        total_nll, total_tokens, total_examples = _run_vllm_ce_worker(
            worker_id=0,
            gpu_ids=None,
            model_path=str(model_path),
            examples=examples,
            loss_on=loss_on,
            max_length=max_length,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
        return _ce_metrics(total_nll, total_tokens, total_examples)
    if tensor_parallel_size != 1:
        raise ValueError("For data-parallel vLLM CE, set tensor_parallel_size: 1 so each worker uses one GPU")
    available_gpus = torch.cuda.device_count()
    if available_gpus and worker_count > available_gpus:
        raise ValueError(f"Requested data_parallel_size={worker_count}, but only {available_gpus} CUDA devices are visible")
    shards = [shard for shard in _split_round_robin(examples, worker_count) if shard]
    worker_count = len(shards)
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for worker_id, shard in enumerate(shards):
        process = ctx.Process(
            target=_vllm_ce_worker_entrypoint,
            kwargs={
                "progress_queue": progress_queue,
                "worker_id": worker_id,
                "gpu_ids": str(worker_id),
                "model_path": str(model_path),
                "examples": shard,
                "loss_on": loss_on,
                "max_length": max_length,
                "batch_size": batch_size,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": dtype,
                "enforce_eager": enforce_eager,
                "trust_remote_code": trust_remote_code,
                "seed": seed + worker_id,
            },
        )
        process.start()
        processes.append(process)
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    finished = 0
    with tqdm(total=len(examples), desc=f"vLLM CE ({worker_count} GPUs)") as progress:
        while finished < worker_count:
            try:
                message_type, payload = progress_queue.get(timeout=5.0)
            except queue.Empty:
                failed = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
                if failed:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    raise RuntimeError(f"vLLM CE worker exited unexpectedly with code(s): {failed}")
                continue
            if message_type == "progress":
                progress.update(int(payload))
            elif message_type == "result":
                worker_nll, worker_tokens, worker_examples = payload
                total_nll += float(worker_nll)
                total_tokens += int(worker_tokens)
                total_examples += int(worker_examples)
            elif message_type == "done":
                finished += 1
            elif message_type == "error":
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                raise RuntimeError(str(payload))
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"vLLM CE worker exited with code {process.exitcode}")
    return _ce_metrics(total_nll, total_tokens, total_examples)


def _vllm_ce_worker_entrypoint(progress_queue, **kwargs):
    try:
        result = _run_vllm_ce_worker(progress_queue=progress_queue, **kwargs)
        progress_queue.put(("result", result))
        progress_queue.put(("done", kwargs["worker_id"]))
    except Exception as exc:
        progress_queue.put(("error", f"CE worker {kwargs.get('worker_id')} failed: {exc!r}"))


def _run_vllm_ce_worker(
    *,
    worker_id: int,
    gpu_ids: str | None,
    model_path: str,
    examples: list[CalibrationExample],
    loss_on: str,
    max_length: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    enforce_eager: bool,
    trust_remote_code: bool,
    seed: int,
    progress_queue=None,
) -> tuple[float, int, int]:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("Install vllm to use backend='vllm' for CE evaluation") from exc
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        dtype=dtype,
        trust_remote_code=bool(trust_remote_code),
        enforce_eager=bool(enforce_eager),
        max_model_len=max_length,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=seed)
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    iterator = range(0, len(examples), max(int(batch_size), 1))
    if progress_queue is None:
        iterator = tqdm(iterator, desc="vLLM CE")
    for start in iterator:
        batch = examples[start : start + max(int(batch_size), 1)]
        prepared = [_prepare_vllm_ce_example(tokenizer, example, loss_on, max_length) for example in batch]
        prompts = [{"prompt_token_ids": item["input_ids"]} for item in prepared]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for item, output in zip(prepared, outputs):
            nll, tokens = _nll_from_prompt_logprobs(output.prompt_logprobs, item["input_ids"], item["label_mask"])
            total_nll += nll
            total_tokens += tokens
            total_examples += 1
        if progress_queue is not None:
            progress_queue.put(("progress", len(batch)))
    return total_nll, total_tokens, total_examples


def _prepare_vllm_ce_example(tokenizer, example: CalibrationExample, loss_on: str, max_length: int) -> dict[str, Any]:
    if loss_on == "response_only" and example.prompt is not None and example.response is not None:
        prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(example.response, add_special_tokens=False)
        input_ids = (prompt_ids + response_ids)[-max_length:]
        kept_prompt = max(0, len(input_ids) - len(response_ids))
        label_mask = [False] * kept_prompt + [True] * (len(input_ids) - kept_prompt)
    elif loss_on in {"full_trajectory", "full_text"}:
        input_ids = tokenizer.encode(example.text, add_special_tokens=False)[:max_length]
        label_mask = [True] * len(input_ids)
    else:
        raise ValueError(f"Unsupported loss_on for vLLM CE: {loss_on}")
    if len(input_ids) < 2:
        label_mask = [False] * len(input_ids)
    else:
        label_mask[0] = False
    return {"input_ids": input_ids, "label_mask": label_mask}


def _nll_from_prompt_logprobs(prompt_logprobs, input_ids: list[int], label_mask: list[bool]) -> tuple[float, int]:
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs")
    if len(prompt_logprobs) != len(input_ids):
        raise RuntimeError(f"prompt_logprobs length {len(prompt_logprobs)} != input length {len(input_ids)}")
    total_nll = 0.0
    total_tokens = 0
    for pos, should_score in enumerate(label_mask):
        if not should_score:
            continue
        entry = prompt_logprobs[pos]
        if entry is None:
            continue
        token_id = input_ids[pos]
        logprob = _extract_vllm_token_logprob(entry, token_id)
        total_nll -= logprob
        total_tokens += 1
    return total_nll, total_tokens


def _extract_vllm_token_logprob(prompt_logprob_entry, token_id: int) -> float:
    if token_id not in prompt_logprob_entry:
        available = list(prompt_logprob_entry.keys())[:10]
        raise KeyError(f"Token id {token_id} missing from vLLM prompt_logprobs; available token ids include {available}")
    value = prompt_logprob_entry[token_id]
    return float(value.logprob) if hasattr(value, "logprob") else float(value)


def _ce_metrics(total_nll: float, total_tokens: int, examples: int) -> dict[str, float | int]:
    ce = float(total_nll) / max(int(total_tokens), 1)
    return {"ce": ce, "perplexity": math.exp(min(ce, 50.0)), "num_tokens": int(total_tokens), "num_examples": int(examples)}
