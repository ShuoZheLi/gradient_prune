from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from data_utils import first_present, load_table
from model_utils import load_model_and_tokenizer


def extract_math_answer(text: Any) -> str | None:
    if text is None:
        return None
    text = str(text)
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return _normalize_answer(boxed[-1])
    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes:
        return _normalize_answer(hashes[-1])
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text.replace(",", ""))
    return _normalize_answer(numbers[-1]) if numbers else _normalize_answer(text.splitlines()[-1] if text else "")


def _normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.strip(".。")
    return answer


def score_math_response(response: str, ground_truth: Any) -> tuple[float, bool]:
    pred = extract_math_answer(response)
    gold = extract_math_answer(ground_truth)
    correct = pred is not None and gold is not None and pred == gold
    return (1.0 if correct else 0.0), correct


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
        prompts = [str(first_present(row, [prompt_key, "prompt", "query", "question", "problem"], "")) for _, row in chunk.iterrows()]
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
            ground_truth = first_present(row, [response_key] if response_key else [], None)
            if ground_truth is None:
                ground_truth = first_present(row, ["ground_truth", "answer", "solution", "response"], None)
            task_score, is_correct = score_math_response(response, ground_truth)
            rows.append({"example_id": int(first_present(row, ["example_id", "id"], start + offset)), "prompt": prompt, "response": response, "ground_truth": ground_truth, "task_score": task_score, "is_correct": is_correct})
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
    ) = args
    df = load_table(dataset_path)
    if max_examples is not None:
        df = df.head(max_examples)
    examples = _dataframe_to_accuracy_examples(df, prompt_key, response_key)
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
        gpu_id = str(worker_id)
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


def _dataframe_to_accuracy_examples(df, prompt_key: str, response_key: str | None) -> list[dict[str, Any]]:
    examples = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt = str(first_present(row, [prompt_key, "prompt", "query", "question", "problem"], ""))
        ground_truth = first_present(row, [response_key] if response_key else [], None)
        if ground_truth is None:
            ground_truth = first_present(row, ["ground_truth", "answer", "solution", "response"], None)
        examples.append({"example_id": int(first_present(row, ["example_id", "id"], idx)), "prompt": prompt, "ground_truth": ground_truth})
    return examples


def _split_round_robin(items: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        shards[idx % num_shards].append(item)
    return shards


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
    progress_queue=None,
) -> list[dict[str, Any]]:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
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
            task_score, is_correct = score_math_response(response, example["ground_truth"])
            rows.append({"example_id": example["example_id"], "prompt": example["prompt"], "response": response, "ground_truth": example["ground_truth"], "task_score": task_score, "is_correct": is_correct})
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
    num_scored = len(rows)
    num_correct = sum(1 for row in rows if row["is_correct"])
    metrics = {"accuracy": num_correct / max(num_scored, 1), "pass@1": num_correct / max(num_scored, 1), "num_examples": len(rows), "num_scored": num_scored, "num_correct": num_correct}
    if metrics_json is not None:
        metrics_json = Path(metrics_json)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
