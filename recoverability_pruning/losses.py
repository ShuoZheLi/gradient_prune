from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass(frozen=True)
class NLLResult:
    loss: torch.Tensor
    loss_sum: torch.Tensor
    num_tokens: int


def causal_sft_nll(model, batch: dict[str, torch.Tensor]) -> NLLResult:
    """Compute token-normalized shifted causal NLL using the batch labels."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        use_cache=False,
    )
    logits = outputs.logits
    labels = batch["labels"].to(logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(IGNORE_INDEX)
    num_tokens = int(valid.sum().item())
    if num_tokens == 0:
        raise ValueError("Batch contains no supervised causal-LM target tokens after shifting")
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return NLLResult(loss=loss_sum / num_tokens, loss_sum=loss_sum, num_tokens=num_tokens)
