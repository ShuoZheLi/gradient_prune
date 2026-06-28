from __future__ import annotations

import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from calibration_loaders import CalibrationExample
from evaluate_accuracy import _dataframe_to_accuracy_examples, _split_round_robin, _worker_cuda_visible_devices, _write_accuracy_outputs, score_task_response
from evaluate_ce import _ce_metrics, _nll_from_prompt_logprobs, _prepare_vllm_ce_example


class SharedVLLMEvaluator:
    def __init__(
        self,
        *,
        model_path: str | Path,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        enforce_eager: bool = True,
        trust_remote_code: bool = False,
        max_model_len: int = 2048,
        seed: int = 42,
    ):
        self.model_path = str(model_path)
        self.trust_remote_code = bool(trust_remote_code)
        self._tokenizer = None
        self.worker_count = max(int(data_parallel_size), 1)
        self.tensor_parallel_size = max(int(tensor_parallel_size), 1)
        if self.tensor_parallel_size != 1 and self.worker_count > 1:
            raise ValueError("For shared data-parallel vLLM evaluation, set tensor_parallel_size: 1 so each worker uses one GPU")
        available_gpus = torch.cuda.device_count()
        if available_gpus and self.worker_count > available_gpus:
            raise ValueError(f"Requested data_parallel_size={self.worker_count}, but only {available_gpus} CUDA devices are visible")
        self.ctx = mp.get_context("spawn")
        self.request_queues = []
        self.response_queue = self.ctx.Queue()
        self.processes = []
        for worker_id in range(self.worker_count):
            request_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=_shared_vllm_worker_entrypoint,
                kwargs={
                    "request_queue": request_queue,
                    "response_queue": self.response_queue,
                    "worker_id": worker_id,
                    "gpu_ids": _worker_cuda_visible_devices(worker_id) if self.worker_count > 1 else None,
                    "model_path": self.model_path,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "dtype": dtype,
                    "enforce_eager": enforce_eager,
                    "trust_remote_code": trust_remote_code,
                    "max_model_len": max_model_len,
                    "seed": seed + worker_id,
                },
            )
            process.start()
            self.request_queues.append(request_queue)
            self.processes.append(process)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for request_queue in self.request_queues:
            request_queue.put({"type": "shutdown"})
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for request_queue in self.request_queues:
            request_queue.close()
        self.response_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=self.trust_remote_code)
        return self._tokenizer

    def evaluate_ce_examples(self, *, examples: list[CalibrationExample], loss_on: str, max_length: int, batch_size: int, desc: str = "vLLM CE") -> dict[str, float | int]:
        if not examples:
            return _ce_metrics(0.0, 0, 0)
        shards = [shard for shard in _split_round_robin(examples, min(self.worker_count, len(examples))) if shard]
        return self._run_ce_shards(shards=shards, loss_on=loss_on, max_length=max_length, batch_size=batch_size, desc=desc, total=len(examples))

    def evaluate_accuracy_examples(
        self,
        *,
        examples: list[dict[str, Any]],
        output_jsonl: str | Path | None,
        metrics_json: str | Path | None,
        max_prompt_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        batch_size: int,
        reward_score_dir: str | Path | None = None,
        desc: str = "vLLM accuracy",
    ) -> dict[str, float | int]:
        if not examples:
            return _write_accuracy_outputs([], output_jsonl, metrics_json)
        shards = [shard for shard in _split_round_robin(examples, min(self.worker_count, len(examples))) if shard]
        for shard_id, shard in enumerate(shards):
            self.request_queues[shard_id].put(
                {
                    "type": "accuracy",
                    "job_id": shard_id,
                    "examples": shard,
                    "max_prompt_length": max_prompt_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "batch_size": batch_size,
                    "reward_score_dir": reward_score_dir,
                }
            )
        rows = []
        finished = 0
        with tqdm(total=len(examples), desc=f"{desc} ({len(shards)} GPUs)") as progress:
            while finished < len(shards):
                message_type, payload = self._get_message()
                if message_type == "progress":
                    progress.update(int(payload))
                elif message_type == "accuracy_result":
                    rows.extend(payload)
                elif message_type == "done":
                    finished += 1
                elif message_type == "error":
                    raise RuntimeError(str(payload))
        rows.sort(key=lambda item: item["example_id"])
        return _write_accuracy_outputs(rows, output_jsonl, metrics_json)

    def _run_ce_shards(self, *, shards, loss_on: str, max_length: int, batch_size: int, desc: str, total: int) -> dict[str, float | int]:
        for shard_id, shard in enumerate(shards):
            self.request_queues[shard_id].put(
                {
                    "type": "ce",
                    "job_id": shard_id,
                    "examples": shard,
                    "loss_on": loss_on,
                    "max_length": max_length,
                    "batch_size": batch_size,
                }
            )
        total_nll = 0.0
        total_tokens = 0
        total_examples = 0
        finished = 0
        with tqdm(total=total, desc=f"{desc} ({len(shards)} GPUs)") as progress:
            while finished < len(shards):
                message_type, payload = self._get_message()
                if message_type == "progress":
                    progress.update(int(payload))
                elif message_type == "ce_result":
                    worker_nll, worker_tokens, worker_examples = payload
                    total_nll += float(worker_nll)
                    total_tokens += int(worker_tokens)
                    total_examples += int(worker_examples)
                elif message_type == "done":
                    finished += 1
                elif message_type == "error":
                    raise RuntimeError(str(payload))
        return _ce_metrics(total_nll, total_tokens, total_examples)

    def _get_message(self):
        while True:
            try:
                return self.response_queue.get(timeout=5.0)
            except queue.Empty:
                failed = [process.exitcode for process in self.processes if process.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Shared vLLM worker exited unexpectedly with code(s): {failed}")


def load_accuracy_examples(dataset_path, prompt_key, response_key, max_examples, tokenizer=None):
    from data_utils import load_table

    df = load_table(dataset_path)
    if max_examples is not None:
        df = df.head(max_examples)
    return _dataframe_to_accuracy_examples(df, prompt_key, response_key, tokenizer)


def _shared_vllm_worker_entrypoint(request_queue, response_queue, **kwargs):
    worker_id = kwargs["worker_id"]
    try:
        _run_shared_vllm_worker(request_queue=request_queue, response_queue=response_queue, **kwargs)
    except Exception as exc:
        response_queue.put(("error", f"shared vLLM worker {worker_id} failed: {exc!r}"))


def _run_shared_vllm_worker(
    *,
    request_queue,
    response_queue,
    worker_id: int,
    gpu_ids: str | None,
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    enforce_eager: bool,
    trust_remote_code: bool,
    max_model_len: int,
    seed: int,
):
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("Install vllm to use shared vLLM evaluation") from exc
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        dtype=dtype,
        trust_remote_code=bool(trust_remote_code),
        enforce_eager=bool(enforce_eager),
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    while True:
        request = request_queue.get()
        request_type = request.get("type")
        if request_type == "shutdown":
            break
        try:
            if request_type == "ce":
                result = _run_loaded_llm_ce_job(
                    llm=llm,
                    tokenizer=tokenizer,
                    examples=request["examples"],
                    loss_on=request["loss_on"],
                    max_length=request["max_length"],
                    batch_size=request["batch_size"],
                    seed=seed,
                    response_queue=response_queue,
                )
                response_queue.put(("ce_result", result))
                response_queue.put(("done", (worker_id, request.get("job_id"))))
            elif request_type == "accuracy":
                rows = _run_loaded_llm_accuracy_job(
                    llm=llm,
                    examples=request["examples"],
                    max_prompt_length=request["max_prompt_length"],
                    max_new_tokens=request["max_new_tokens"],
                    temperature=request["temperature"],
                    top_p=request["top_p"],
                    top_k=request["top_k"],
                    batch_size=request["batch_size"],
                    reward_score_dir=request.get("reward_score_dir"),
                    seed=seed,
                    response_queue=response_queue,
                )
                response_queue.put(("accuracy_result", rows))
                response_queue.put(("done", (worker_id, request.get("job_id"))))
            else:
                raise ValueError(f"Unsupported shared vLLM request type: {request_type}")
        except Exception as exc:
            response_queue.put(("error", f"shared vLLM worker {worker_id} job failed: {exc!r}"))


def _run_loaded_llm_ce_job(*, llm, tokenizer, examples, loss_on, max_length, batch_size, seed, response_queue):
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=seed)
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    for start in range(0, len(examples), max(int(batch_size), 1)):
        batch = examples[start : start + max(int(batch_size), 1)]
        prepared = [_prepare_vllm_ce_example(tokenizer, example, loss_on, max_length) for example in batch]
        prompts = [{"prompt_token_ids": item["input_ids"]} for item in prepared]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for item, output in zip(prepared, outputs):
            nll, tokens = _nll_from_prompt_logprobs(output.prompt_logprobs, item["input_ids"], item["label_mask"])
            total_nll += nll
            total_tokens += tokens
            total_examples += 1
        response_queue.put(("progress", len(batch)))
    return total_nll, total_tokens, total_examples


def _run_loaded_llm_accuracy_job(*, llm, examples, max_prompt_length, max_new_tokens, temperature, top_p, top_k, batch_size, reward_score_dir, seed, response_queue):
    from vllm import SamplingParams

    sampling_kwargs = {"max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "seed": seed}
    if top_k and top_k > 0:
        sampling_kwargs["top_k"] = top_k
    sampling = SamplingParams(**sampling_kwargs)
    rows = []
    for start in range(0, len(examples), max(int(batch_size), 1)):
        batch = examples[start : start + max(int(batch_size), 1)]
        prompts = [example["prompt"] for example in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for example, output in zip(batch, outputs):
            response = output.outputs[0].text
            task_score, is_correct = score_task_response(response, example["ground_truth"], example.get("data_source", ""), reward_score_dir)
            task_score_value = None if example["ground_truth"] is None else task_score
            rows.append({"example_id": example["example_id"], "prompt": example["prompt"], "response": response, "ground_truth": example["ground_truth"], "data_source": example.get("data_source", ""), "task_score": task_score_value, "is_correct": is_correct})
        response_queue.put(("progress", len(batch)))
    return rows
