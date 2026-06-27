from __future__ import annotations

import math

import torch
from tqdm import tqdm

from calibration_loaders import load_calibration_examples, make_calibration_dataloader
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
    ce = total_nll / max(total_tokens, 1)
    return {"ce": ce, "perplexity": math.exp(min(ce, 50.0)), "num_tokens": total_tokens, "num_examples": examples}
