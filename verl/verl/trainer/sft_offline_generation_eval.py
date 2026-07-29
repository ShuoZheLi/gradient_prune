import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl.trainer.sft_trainer import (
    build_generation_reward_metrics,
    create_generation_eval_dataset,
    evaluate_generation_reward_batches,
    merge_generation_reward_results,
    prefix_generation_reward_metrics,
)
from verl.utils.dataset.rl_dataset import collate_fn as rlhf_collate_fn
from verl.utils.tokenizer import hf_processor, hf_tokenizer


class OfflineVLLMModel:
    def __init__(self, model_path):
        self.config = SimpleNamespace(_name_or_path=model_path)
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    tmp_path.replace(path)


def touch_status(output_dir, name, payload=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    if payload is None:
        path.write_text("")
    else:
        atomic_write_json(path, payload)


def load_model_for_backend(model_path, tokenizer, generation_config, device):
    backend = generation_config.get("backend", "vllm")
    if backend == "vllm":
        generation_config["vllm_sync_weights"] = False
        generation_config.setdefault("vllm_model_path", model_path)
        generation_config.setdefault("vllm_tokenizer_path", model_path)
        return OfflineVLLMModel(model_path)
    if backend == "hf":
        from transformers import AutoModelForCausalLM

        dtype = generation_config.get("dtype", None)
        torch_dtype = None
        if dtype is not None and str(dtype).lower() not in {"", "none", "null", "auto"}:
            dtype_name = str(dtype).lower()
            torch_dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }[dtype_name]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(generation_config.get("trust_remote_code", False)),
        ).to(device)
        model.eval()
        return model
    raise ValueError(f"Unsupported offline generation backend {backend!r}")


def run_manifest(manifest):
    model_path = manifest["model_path"]
    eval_specs = manifest["eval_specs"]
    output_dir = Path(manifest["output_dir"])
    metrics_path = Path(manifest["metrics_path"])
    data_config = OmegaConf.create(manifest["data_config"])
    generation_config = OmegaConf.create(manifest["generation_config"])
    device = manifest.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_samples = int(manifest.get("max_samples", -1))
    batch_size = int(manifest.get("batch_size", 1))
    num_workers = int(manifest.get("num_workers", 0))
    prefix_metrics = bool(manifest.get("prefix_metrics", len(eval_specs) > 1))

    tokenizer = hf_tokenizer(model_path, trust_remote_code=bool(generation_config.get("trust_remote_code", False)))
    try:
        processor = hf_processor(model_path, trust_remote_code=bool(generation_config.get("trust_remote_code", False)))
    except Exception:
        processor = None

    model = load_model_for_backend(model_path, tokenizer, generation_config, device)
    config = OmegaConf.create({"data": data_config, "trainer": {"generation_eval": generation_config}})

    combined_metrics = {}
    summaries = {}
    for spec in eval_specs:
        dataset = create_generation_eval_dataset(
            spec["files"],
            data_config,
            tokenizer,
            processor,
            max_samples=max_samples,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=rlhf_collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )
        result = evaluate_generation_reward_batches(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            config=config,
            sync_version=manifest.get("global_step", None),
        )
        merged = merge_generation_reward_results([result])
        metrics = build_generation_reward_metrics(merged)
        sample_count = len(merged["sample_scores"])
        if sample_count > 0:
            metrics["val-aux/reward/num_samples"] = sample_count
            metrics["val-aux/response_length/mean"] = float(np.mean(merged["sample_response_lengths"]))
            metrics["val-aux/response_length/max"] = int(np.max(merged["sample_response_lengths"]))
        if prefix_metrics:
            metrics = prefix_generation_reward_metrics(metrics, spec["name"])
        combined_metrics.update(metrics)
        summaries[spec["name"]] = {"num_samples": sample_count, "files": spec["files"]}

    atomic_write_json(metrics_path, combined_metrics)
    atomic_write_json(output_dir / "summary.json", summaries)
    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description="Offline SFT generation_reward evaluator.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    output_dir = Path(manifest["output_dir"])
    touch_status(output_dir, "status.running", {"manifest": str(manifest_path)})
    for stale in ["status.done", "status.failed"]:
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    try:
        metrics = run_manifest(manifest)
        touch_status(output_dir, "status.done", {"metric_count": len(metrics)})
    except Exception as exc:
        failure = {"error": repr(exc), "traceback": traceback.format_exc()}
        touch_status(output_dir, "status.failed", failure)
        print(f"Offline SFT generation eval failed: {exc}", file=sys.stderr)
        print(failure["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
