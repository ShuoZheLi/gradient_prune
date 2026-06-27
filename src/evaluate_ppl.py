from __future__ import annotations

from datasets import load_dataset

from calibration_loaders import CalibrationExample, make_calibration_dataloader
from evaluate_ce import _ce_metrics, evaluate_ce_dataloader, evaluate_ce_vllm_examples


def load_text_examples_from_dataset(*, dataset_name: str = "wikitext", dataset_config: str | None = "wikitext-2-raw-v1", split: str = "validation", text_key: str = "text", max_samples: int | None = 256) -> list[CalibrationExample]:
    load_kwargs = {"path": dataset_name, "split": split}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    ds = load_dataset(**load_kwargs)
    texts = [str(row[text_key]) for row in ds if str(row[text_key]).strip()]
    if max_samples is not None:
        texts = texts[:max_samples]
    return [CalibrationExample(None, None, text) for text in texts]


def evaluate_text_ppl(model, tokenizer, *, dataset_name: str = "wikitext", dataset_config: str | None = "wikitext-2-raw-v1", split: str = "validation", text_key: str = "text", max_samples: int | None = 256, max_length: int = 2048, batch_size: int = 1, device: str | None = None) -> dict[str, float | int]:
    examples = load_text_examples_from_dataset(dataset_name=dataset_name, dataset_config=dataset_config, split=split, text_key=text_key, max_samples=max_samples)
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on="full_text", microbatch_size=batch_size)
    return evaluate_ce_dataloader(model, dataloader, device=device)


def evaluate_text_ppl_vllm(*, model_path, dataset_name: str = "wikitext", dataset_config: str | None = "wikitext-2-raw-v1", split: str = "validation", text_key: str = "text", max_samples: int | None = 256, max_length: int = 2048, batch_size: int = 32, data_parallel_size: int = 1, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9, dtype: str = "auto", enforce_eager: bool = True, trust_remote_code: bool = False, seed: int = 42) -> dict[str, float | int]:
    examples = load_text_examples_from_dataset(dataset_name=dataset_name, dataset_config=dataset_config, split=split, text_key=text_key, max_samples=max_samples)
    return evaluate_ce_vllm_examples(model_path=model_path, examples=examples, loss_on="full_text", max_length=max_length, batch_size=batch_size, data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, enforce_eager=enforce_eager, trust_remote_code=trust_remote_code, seed=seed)


def evaluate_wikitext_ppl(model, tokenizer, *, dataset_name: str = "wikitext", dataset_config: str | None = "wikitext-2-raw-v1", split: str = "validation", text_key: str = "text", max_samples: int | None = 256, max_length: int = 2048, batch_size: int = 1, device: str | None = None) -> dict[str, float | int]:
    return evaluate_text_ppl(model, tokenizer, dataset_name=dataset_name, dataset_config=dataset_config, split=split, text_key=text_key, max_samples=max_samples, max_length=max_length, batch_size=batch_size, device=device)
