from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from data_utils import first_present, load_table
from model_utils import load_model_and_tokenizer
from task_scoring import (
    dataframe_to_task_examples,
    extract_data_source,
    extract_ground_truth,
    extract_prompt,
    score_math_response,
    score_task_response,
    task_example_to_dict,
)


def _extract_prompt(row, prompt_key: str, tokenizer=None) -> str:
    return extract_prompt(row, prompt_key, tokenizer)


def _extract_data_source(row) -> str:
    return extract_data_source(row)


def _extract_ground_truth(row, response_key: str | None) -> Any:
    return extract_ground_truth(row, response_key)


def evaluate_task_accuracy(
    model_or_model_path,
    tokenizer=None,
    dataset_path: str | None = None,
    backend: str = "transformers",
    output_jsonl: str | Path | None = None,
    metrics_json: str | Path | None = None,
    prompt_key: str = "prompt",
    response_key: str | None = None,
    reward_score_dir: str | None = None,
    max_examples: int | None = None,
    max_prompt_length: int = 2048,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    enforce_eager: bool = True,
    trust_remote_code: bool = False,
) -> dict[str, float | int]:
    if backend == "vllm":
        if not isinstance(model_or_model_path, (str, Path)):
            raise TypeError("backend='vllm' requires a saved model path, not an in-memory transformers model")
        return _evaluate_vllm(
            model_or_model_path,
            dataset_path,
            output_jsonl,
            metrics_json,
            prompt_key,
            response_key,
            max_examples,
            max_prompt_length,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            data_parallel_size,
            tensor_parallel_size,
            gpu_memory_utilization,
            dtype,
            enforce_eager,
            trust_remote_code,
            reward_score_dir,
            batch_size=batch_size,
        )
    if backend != "transformers":
        raise ValueError(f"Unsupported backend: {backend}")
    if dataset_path is None:
        raise ValueError("dataset_path is required")
    if isinstance(model_or_model_path, str):
        model, tokenizer = load_model_and_tokenizer(model_or_model_path, dtype="auto", device="cuda:0" if torch.cuda.is_available() else "cpu", trust_remote_code=False)
    else:
        model = model_or_model_path
        if tokenizer is None:
            raise ValueError("tokenizer is required when passing a model object")
    torch.manual_seed(seed)
    df = load_table(dataset_path)
    if max_examples is not None:
        df = df.head(max_examples)
    rows = []
    device = next(model.parameters()).device
    model.eval()
    for start in tqdm(range(0, len(df), batch_size), desc="task accuracy"):
        chunk = df.iloc[start : start + batch_size]
        prompts = [_extract_prompt(row, prompt_key, tokenizer) for _, row in chunk.iterrows()]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
        do_sample = temperature and temperature > 0
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": bool(do_sample), "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
        with torch.no_grad():
            output_ids = model.generate(**encoded, **gen_kwargs)
        new_ids = output_ids[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for offset, ((_, row), prompt, response) in enumerate(zip(chunk.iterrows(), prompts, responses)):
            ground_truth = _extract_ground_truth(row, response_key)
            data_source = _extract_data_source(row)
            task_score, is_correct = score_task_response(response, ground_truth, data_source, reward_score_dir)
            task_score_value = None if ground_truth is None else task_score
            rows.append({"example_id": int(first_present(row, ["example_id", "id"], start + offset)), "prompt": prompt, "response": response, "ground_truth": ground_truth, "data_source": data_source, "task_score": task_score_value, "is_correct": is_correct})
    metrics = _write_accuracy_outputs(rows, output_jsonl, metrics_json)
    return metrics


def _evaluate_vllm(*args, **kwargs):
    (
        model_path,
        dataset_path,
        output_jsonl,
        metrics_json,
        prompt_key,
        response_key,
        max_examples,
        max_prompt_length,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        data_parallel_size,
        tensor_parallel_size,
        gpu_memory_utilization,
        dtype,
        enforce_eager,
        trust_remote_code,
        reward_score_dir,
    ) = args
    df = load_table(dataset_path)
    if max_examples is not None:
        df = df.head(max_examples)
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False, trust_remote_code=bool(trust_remote_code))
    except Exception:
        tokenizer = None
    examples = _dataframe_to_accuracy_examples(df, prompt_key, response_key, tokenizer)
    if not examples:
        return _write_accuracy_outputs([], output_jsonl, metrics_json)
    worker_count = min(max(int(data_parallel_size), 1), len(examples))
    tensor_parallel_size = max(int(tensor_parallel_size), 1)
    if worker_count == 1:
        rows = _run_vllm_worker(
            worker_id=0,
            gpu_ids=None,
            model_path=str(model_path),
            examples=examples,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            batch_size=batch_size_from_kwargs(kwargs),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            reward_score_dir=reward_score_dir,
        )
        return _write_accuracy_outputs(rows, output_jsonl, metrics_json)

    if tensor_parallel_size != 1:
        raise ValueError("For data-parallel vLLM evaluation, set tensor_parallel_size: 1 so each worker uses one GPU")
    available_gpus = torch.cuda.device_count()
    if available_gpus and worker_count > available_gpus:
        raise ValueError(f"Requested data_parallel_size={worker_count}, but only {available_gpus} CUDA devices are visible")
    shards = [shard for shard in _split_round_robin(examples, worker_count) if shard]
    worker_count = len(shards)
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for worker_id, shard in enumerate(shards):
        gpu_id = _worker_cuda_visible_devices(worker_id)
        process = ctx.Process(
            target=_vllm_worker_entrypoint,
            kwargs={
                "progress_queue": progress_queue,
                "worker_id": worker_id,
                "gpu_ids": gpu_id,
                "model_path": str(model_path),
                "examples": shard,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed + worker_id,
                "batch_size": batch_size_from_kwargs(kwargs),
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": dtype,
                "enforce_eager": enforce_eager,
                "trust_remote_code": trust_remote_code,
                "reward_score_dir": reward_score_dir,
            },
        )
        process.start()
        processes.append(process)

    rows = []
    finished = 0
    with tqdm(total=len(examples), desc=f"vLLM accuracy ({worker_count} GPUs)") as progress:
        while finished < worker_count:
            try:
                message_type, payload = progress_queue.get(timeout=5.0)
            except queue.Empty:
                failed = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
                if failed:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    raise RuntimeError(f"vLLM worker exited unexpectedly with code(s): {failed}")
                continue
            if message_type == "progress":
                progress.update(int(payload))
            elif message_type == "result":
                rows.extend(payload)
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
            raise RuntimeError(f"vLLM worker exited with code {process.exitcode}")
    rows.sort(key=lambda item: item["example_id"])
    return _write_accuracy_outputs(rows, output_jsonl, metrics_json)


def batch_size_from_kwargs(kwargs: dict) -> int:
    return max(int(kwargs.get("batch_size", 1)), 1)


def _dataframe_to_accuracy_examples(df, prompt_key: str, response_key: str | None, tokenizer=None) -> list[dict[str, Any]]:
    return [task_example_to_dict(example) for example in dataframe_to_task_examples(df, prompt_key, response_key, tokenizer)]


def _split_round_robin(items: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        shards[idx % num_shards].append(item)
    return shards


def _worker_cuda_visible_devices(worker_id: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(worker_id)
    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    if not devices:
        return str(worker_id)
    if worker_id >= len(devices):
        raise ValueError(f"Worker {worker_id} requested, but CUDA_VISIBLE_DEVICES only has {len(devices)} device(s): {visible_devices}")
    return devices[worker_id]


def _prepare_vllm_worker_environment(gpu_ids: str | None) -> None:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    for key in (
        "LOCAL_RANK",
        "RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ):
        os.environ.pop(key, None)
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


def _vllm_worker_entrypoint(progress_queue, **kwargs):
    try:
        rows = _run_vllm_worker(progress_queue=progress_queue, **kwargs)
        progress_queue.put(("result", rows))
        progress_queue.put(("done", kwargs["worker_id"]))
    except Exception as exc:
        progress_queue.put(("error", f"worker {kwargs.get('worker_id')} failed: {exc!r}"))


def _run_vllm_worker(
    *,
    worker_id: int,
    gpu_ids: str | None,
    model_path: str,
    examples: list[dict[str, Any]],
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    enforce_eager: bool,
    trust_remote_code: bool,
    reward_score_dir: str | Path | None,
    progress_queue=None,
) -> list[dict[str, Any]]:
    _prepare_vllm_worker_environment(gpu_ids)
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("Install vllm or use backend='transformers'") from exc
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        dtype=dtype,
        trust_remote_code=bool(trust_remote_code),
        enforce_eager=bool(enforce_eager),
        max_model_len=max_prompt_length + max_new_tokens,
    )
    sampling_kwargs = {"max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "seed": seed}
    if top_k and top_k > 0:
        sampling_kwargs["top_k"] = top_k
    sampling = SamplingParams(**sampling_kwargs)
    rows = []
    iterator = range(0, len(examples), max(int(batch_size), 1))
    if progress_queue is None:
        iterator = tqdm(iterator, desc="vLLM accuracy")
    for start in iterator:
        batch = examples[start : start + max(int(batch_size), 1)]
        prompts = [example["prompt"] for example in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for example, output in zip(batch, outputs):
            response = output.outputs[0].text
            task_score, is_correct = score_task_response(response, example["ground_truth"], example.get("data_source", ""), reward_score_dir)
            task_score_value = None if example["ground_truth"] is None else task_score
            rows.append({"example_id": example["example_id"], "prompt": example["prompt"], "response": response, "ground_truth": example["ground_truth"], "data_source": example.get("data_source", ""), "task_score": task_score_value, "is_correct": is_correct})
        if progress_queue is not None:
            progress_queue.put(("progress", len(batch)))
    return rows


def _write_accuracy_outputs(rows, output_jsonl, metrics_json):
    if output_jsonl is not None:
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    num_scored = sum(1 for row in rows if row.get("task_score") is not None)
    num_correct = sum(1 for row in rows if row.get("task_score") is not None and row["is_correct"])
    accuracy = num_correct / max(num_scored, 1)
    metrics = {"accuracy": accuracy, "pass@1": accuracy, "num_examples": len(rows), "num_scored": num_scored, "num_unscored": len(rows) - num_scored, "num_correct": num_correct}
    if metrics_json is not None:
        metrics_json = Path(metrics_json)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
