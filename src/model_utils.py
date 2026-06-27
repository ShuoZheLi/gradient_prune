from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def resolve_dtype(dtype: str | torch.dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "full"}:
        return torch.float32
    if name in {"auto", "none"}:
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model_and_tokenizer(model_name_or_path: str, dtype: str = "bf16", device: str = "cuda:0", trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = resolve_dtype(dtype)
    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.to(torch.device(device))
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    LOGGER.info("Loaded model=%s dtype=%s device=%s", model_name_or_path, dtype, device)
    return model, tokenizer


@contextmanager
def temporarily_disable_cache(model):
    old_value = getattr(model.config, "use_cache", None)
    if old_value is not None:
        model.config.use_cache = False
    try:
        yield
    finally:
        if old_value is not None:
            model.config.use_cache = old_value
