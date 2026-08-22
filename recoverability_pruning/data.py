from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import IGNORE_INDEX


@dataclass(frozen=True)
class TrajectoryExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    trajectory_identity: str
    disjoint_identity: str | None


class TrajectoryDataset(Dataset):
    def __init__(self, examples: list[TrajectoryExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
        }


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, bool) else False


def _available_parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _load_dataframe(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    dataset_path = Path(path).expanduser()
    if dataset_path.is_dir():
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if parquet_files:
            frames = []
            for item in parquet_files:
                selected = [column for column in columns or () if column in _available_parquet_columns(item)]
                frames.append(pd.read_parquet(item, columns=selected or None))
            return pd.concat(frames, ignore_index=True)
        jsonl_files = sorted(dataset_path.glob("*.jsonl"))
        if jsonl_files:
            records = []
            for item in jsonl_files:
                with item.open("r", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle if line.strip())
            return pd.DataFrame(records)
        raise ValueError(f"No parquet or jsonl files found in dataset directory: {dataset_path}")
    if dataset_path.suffix == ".parquet":
        selected = [column for column in columns or () if column in _available_parquet_columns(dataset_path)]
        return pd.read_parquet(dataset_path, columns=selected or None)
    if dataset_path.suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            return pd.DataFrame(json.loads(line) for line in handle if line.strip())
    raise ValueError(f"Unsupported dataset path: {dataset_path}")


def _as_int_list(value: Any, column: str, row_index: int) -> list[int]:
    if _is_missing(value):
        raise ValueError(f"Missing {column!r} at row {row_index}")
    if isinstance(value, str):
        value = json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a token sequence in {column!r} at row {row_index}, got {type(value).__name__}")
    return [int(item) for item in value]


def _trajectory_identity(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _value_identity(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_labels(
    row: pd.Series,
    token_ids: list[int],
    *,
    loss_on: str,
    prompt_length_column: str,
    loss_mask_column: str,
    prompt_text_column: str,
    derive_prompt_length_from_prompt: bool,
    prompt_tokenizer: Callable[[str], Sequence[int]] | None,
    prompt_token_cache: dict[str, tuple[int, ...]],
    row_index: int,
) -> list[int]:
    if loss_on == "full_trajectory":
        return list(token_ids)
    if loss_on == "response_only":
        if prompt_length_column in row and not _is_missing(row[prompt_length_column]):
            prompt_length = int(row[prompt_length_column])
        elif derive_prompt_length_from_prompt:
            if prompt_tokenizer is None:
                raise ValueError("Prompt-length derivation requires a tokenizer")
            if prompt_text_column not in row or _is_missing(row[prompt_text_column]):
                raise ValueError(
                    f"Prompt-length derivation requires explicit text column {prompt_text_column!r}"
                )
            prompt_text = str(row[prompt_text_column])
            prompt_ids = prompt_token_cache.get(prompt_text)
            if prompt_ids is None:
                prompt_ids = tuple(int(token_id) for token_id in prompt_tokenizer(prompt_text))
                prompt_token_cache[prompt_text] = prompt_ids
            prompt_length = len(prompt_ids)
            if token_ids[:prompt_length] != list(prompt_ids):
                mismatch = next(
                    (
                        index
                        for index, (trajectory_id, prompt_id) in enumerate(
                            zip(token_ids, prompt_ids, strict=False)
                        )
                        if trajectory_id != prompt_id
                    ),
                    min(len(token_ids), len(prompt_ids)),
                )
                raise ValueError(
                    f"Tokenized {prompt_text_column!r} is not an exact prefix of {len(token_ids)} stored "
                    f"trajectory IDs at row {row_index}; first mismatch index={mismatch}. "
                    "Refusing to infer a response boundary."
                )
        else:
            raise ValueError(
                f"loss_on='response_only' requires explicit column {prompt_length_column!r}; "
                "or explicitly enable verified derivation from prompt text"
            )
        if not 0 <= prompt_length <= len(token_ids):
            raise ValueError(
                f"Invalid prompt length {prompt_length} for {len(token_ids)} tokens at row {row_index}"
            )
        return [IGNORE_INDEX] * prompt_length + token_ids[prompt_length:]
    if loss_on == "loss_mask":
        if loss_mask_column not in row or _is_missing(row[loss_mask_column]):
            raise ValueError(f"loss_on='loss_mask' requires explicit column {loss_mask_column!r}")
        mask = _as_int_list(row[loss_mask_column], loss_mask_column, row_index)
        if len(mask) != len(token_ids):
            raise ValueError(
                f"Loss mask length {len(mask)} does not match token length {len(token_ids)} at row {row_index}"
            )
        invalid = sorted(set(mask) - {0, 1})
        if invalid:
            raise ValueError(f"Loss mask must contain only 0/1 values, found {invalid} at row {row_index}")
        return [token_id if include else IGNORE_INDEX for token_id, include in zip(token_ids, mask, strict=True)]
    raise ValueError(f"Unsupported loss_on={loss_on!r}")


def _truncate(token_ids: list[int], labels: list[int], max_length: int, truncation_side: str) -> tuple[list[int], list[int]]:
    if max_length <= 1:
        raise ValueError(f"max_length must be at least 2, got {max_length}")
    if len(token_ids) <= max_length:
        return token_ids, labels
    if truncation_side == "right":
        return token_ids[:max_length], labels[:max_length]
    if truncation_side == "left":
        return token_ids[-max_length:], labels[-max_length:]
    raise ValueError(f"Unsupported truncation_side={truncation_side!r}")


def load_trajectory_dataset(
    path: str | Path,
    *,
    token_ids_column: str = "prompt_generated_trajectory_ids",
    loss_on: str = "full_trajectory",
    prompt_length_column: str = "prompt_length",
    loss_mask_column: str = "loss_mask",
    prompt_text_column: str = "prompt",
    derive_prompt_length_from_prompt: bool = False,
    prompt_tokenizer: Callable[[str], Sequence[int]] | None = None,
    disjoint_key_column: str | None = "prompt",
    max_length: int = 4096,
    truncation_side: str = "right",
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> TrajectoryDataset:
    requested_columns = [token_ids_column]
    if loss_on == "response_only":
        requested_columns.append(prompt_length_column)
        if derive_prompt_length_from_prompt:
            requested_columns.append(prompt_text_column)
    elif loss_on == "loss_mask":
        requested_columns.append(loss_mask_column)
    if disjoint_key_column:
        requested_columns.append(disjoint_key_column)
    requested_columns = list(dict.fromkeys(requested_columns))
    frame = _load_dataframe(path, columns=requested_columns)
    if token_ids_column not in frame.columns:
        raise ValueError(f"Dataset {path} does not contain required column {token_ids_column!r}")
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=seed)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        frame = frame.head(max_samples)

    examples = []
    prompt_token_cache: dict[str, tuple[int, ...]] = {}
    for row_index, (_, row) in enumerate(frame.iterrows()):
        full_token_ids = _as_int_list(row[token_ids_column], token_ids_column, row_index)
        if len(full_token_ids) < 2:
            raise ValueError(f"Trajectory at row {row_index} has fewer than two tokens")
        labels = _build_labels(
            row,
            full_token_ids,
            loss_on=loss_on,
            prompt_length_column=prompt_length_column,
            loss_mask_column=loss_mask_column,
            prompt_text_column=prompt_text_column,
            derive_prompt_length_from_prompt=derive_prompt_length_from_prompt,
            prompt_tokenizer=prompt_tokenizer,
            prompt_token_cache=prompt_token_cache,
            row_index=row_index,
        )
        token_ids, labels = _truncate(full_token_ids, labels, max_length, truncation_side)
        supervised_after_shift = sum(label != IGNORE_INDEX for label in labels[1:])
        if supervised_after_shift == 0:
            raise ValueError(f"Trajectory at row {row_index} has no supervised targets after truncation and causal shift")
        examples.append(
            TrajectoryExample(
                input_ids=tuple(token_ids),
                labels=tuple(labels),
                trajectory_identity=_trajectory_identity(full_token_ids),
                disjoint_identity=(
                    _value_identity(row[disjoint_key_column])
                    if disjoint_key_column
                    and disjoint_key_column in row
                    and not _is_missing(row[disjoint_key_column])
                    else None
                ),
            )
        )
    if not examples:
        raise ValueError(f"Dataset is empty after selection: {path}")
    return TrajectoryDataset(examples)


def assert_disjoint_datasets(reference: TrajectoryDataset, kd: TrajectoryDataset) -> None:
    reference_trajectories = {example.trajectory_identity for example in reference.examples}
    kd_trajectories = {example.trajectory_identity for example in kd.examples}
    trajectory_overlap = reference_trajectories & kd_trajectories
    if trajectory_overlap:
        raise ValueError(
            "Reference and KD datasets are not disjoint: "
            f"found {len(trajectory_overlap)} identical token trajectories"
        )
    reference_keys = {example.disjoint_identity for example in reference.examples if example.disjoint_identity}
    kd_keys = {example.disjoint_identity for example in kd.examples if example.disjoint_identity}
    key_overlap = reference_keys & kd_keys
    if key_overlap:
        raise ValueError(
            "Reference and KD datasets are not disjoint: "
            f"found {len(key_overlap)} overlapping explicit disjoint keys"
        )


def collate_trajectory_batch(features: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(feature["input_ids"].numel() for feature in features)
    input_ids = []
    labels = []
    attention_mask = []
    for feature in features:
        length = feature["input_ids"].numel()
        padding = max_length - length
        input_ids.append(
            torch.cat([torch.full((padding,), pad_token_id, dtype=torch.long), feature["input_ids"]])
        )
        labels.append(
            torch.cat([torch.full((padding,), IGNORE_INDEX, dtype=torch.long), feature["labels"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(padding, dtype=torch.long), torch.ones(length, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def make_trajectory_dataloader(
    dataset: TrajectoryDataset,
    *,
    batch_size: int,
    pad_token_id: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda features: collate_trajectory_batch(features, pad_token_id),
    )
