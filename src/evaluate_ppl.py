from __future__ import annotations

from datasets import load_dataset

from calibration_loaders import CalibrationExample, make_calibration_dataloader
from evaluate_ce import evaluate_ce_dataloader


def evaluate_wikitext_ppl(model, tokenizer, *, dataset_name: str = "wikitext", dataset_config: str = "wikitext-2-raw-v1", split: str = "validation", text_key: str = "text", max_samples: int | None = 256, max_length: int = 2048, batch_size: int = 1, device: str | None = None) -> dict[str, float | int]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    texts = [str(row[text_key]) for row in ds if str(row[text_key]).strip()]
    if max_samples is not None:
        texts = texts[:max_samples]
    examples = [CalibrationExample(None, None, text) for text in texts]
    dataloader = make_calibration_dataloader(examples, tokenizer, max_length=max_length, loss_on="full_text", microbatch_size=batch_size)
    return evaluate_ce_dataloader(model, dataloader, device=device)
