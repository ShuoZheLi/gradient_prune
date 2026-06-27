from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import first_present, load_table

LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100


@dataclass
class CalibrationExample:
    prompt: str | None
    response: str | None
    text: str


class CalibrationDataset(Dataset):
    def __init__(self, examples: list[CalibrationExample], tokenizer, max_length: int, loss_on: str):
        if loss_on not in {"full_trajectory", "response_only", "full_text"}:
            raise ValueError(f"Unsupported loss_on: {loss_on}")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_on = loss_on

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.loss_on == "response_only" and example.prompt is not None and example.response is not None:
            prompt_ids = self.tokenizer(example.prompt, add_special_tokens=False).input_ids
            response_ids = self.tokenizer(example.response, add_special_tokens=False).input_ids
            input_ids = (prompt_ids + response_ids)[-self.max_length :]
            kept_prompt = max(0, len(input_ids) - len(response_ids))
            labels = [IGNORE_INDEX] * kept_prompt + input_ids[kept_prompt:]
        else:
            input_ids = self.tokenizer(example.text, add_special_tokens=False, truncation=True, max_length=self.max_length).input_ids
            labels = list(input_ids)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}


def collate_lm_batch(features: list[dict[str, torch.Tensor]], tokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(item["input_ids"].numel() for item in features)
    input_ids, labels, attention_mask = [], [], []
    for item in features:
        ids = item["input_ids"]
        labs = item["labels"]
        pad = max_len - ids.numel()
        input_ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), ids]))
        labels.append(torch.cat([torch.full((pad,), IGNORE_INDEX, dtype=torch.long), labs]))
        attention_mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(ids.numel(), dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}


def load_calibration_examples(
    path: str,
    *,
    calibration_type: str = "prompt_response",
    only_correct: bool = False,
    max_samples: int | None = None,
    text_key: str | None = None,
    prompt_key: str = "prompt",
    response_key: str | None = "response",
    shuffle: bool = False,
    seed: int = 42,
) -> list[CalibrationExample]:
    df = load_table(path)
    if only_correct and "is_correct" in df.columns:
        df = df[df["is_correct"] == True]
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed)
    if max_samples is not None:
        df = df.head(max_samples)
    examples = []
    for _, row in df.iterrows():
        if calibration_type == "text":
            text = str(first_present(row, [text_key or "text", "prompt_generated_trajectory", "text", "content"]))
            examples.append(CalibrationExample(None, None, text))
        else:
            prompt = str(first_present(row, [prompt_key, "prompt", "query", "question"], ""))
            response = first_present(row, [response_key] if response_key else [], None)
            if response is None:
                response = first_present(row, ["response", "answer", "solution"], "")
            response = str(response)
            text = str(first_present(row, ["prompt_generated_trajectory", "trajectory", "text"], prompt + response))
            examples.append(CalibrationExample(prompt, response, text))
    LOGGER.info("Loaded %d calibration examples from %s", len(examples), path)
    return examples


def make_calibration_dataloader(examples: list[CalibrationExample], tokenizer, *, max_length: int, loss_on: str, microbatch_size: int, shuffle: bool = False):
    dataset = CalibrationDataset(examples, tokenizer, max_length=max_length, loss_on=loss_on)
    return DataLoader(dataset, batch_size=microbatch_size, shuffle=shuffle, collate_fn=lambda batch: collate_lm_batch(batch, tokenizer))
