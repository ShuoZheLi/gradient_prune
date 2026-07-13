from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CREATE_DIR = REPO_ROOT / "create_calibration_dataset"
for path in (SRC_DIR, CREATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_accuracy_test import load_examples  # noqa: E402
from task_scoring import normalize_enable_thinking  # noqa: E402

from response_analysis.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared teacher-forced response prefix bank.")
    parser.add_argument("--output", default="outputs/fixed_prefix_bank.jsonl")
    parser.add_argument("--source", choices=["dataset_reference", "generations", "shared_file"], default="dataset_reference")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--generations", default=None)
    parser.add_argument("--shared_file", default=None)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", default="true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def source_records(args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    if args.source == "dataset_reference":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required for dataset_reference source")
        examples = load_examples(
            args.dataset_path,
            tokenizer,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            start_index=args.start_index,
            max_examples=args.max_examples,
            shuffle=False,
            seed=args.seed,
            enable_thinking=normalize_enable_thinking(args.enable_thinking),
        )
        return [
            {
                "prompt_id": example.example_id,
                "prompt": example.prompt_text,
                "reference_text": "" if example.ground_truth is None else str(example.ground_truth),
                "ground_truth": example.ground_truth,
                "source": "dataset_reference",
            }
            for example in examples
        ]
    if args.source == "generations":
        if not args.generations:
            raise ValueError("--generations is required for generations source")
        seen: set[Any] = set()
        records = []
        for record in read_jsonl(args.generations):
            prompt_id = record.get("prompt_id")
            if prompt_id in seen:
                continue
            seen.add(prompt_id)
            records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": record.get("prompt", ""),
                    "reference_text": record.get("generated_text", ""),
                    "ground_truth": record.get("ground_truth"),
                    "source": "generations",
                    "source_model_id": record.get("model_id"),
                    "source_sample_id": record.get("sample_id"),
                }
            )
        return records[: args.max_examples] if args.max_examples >= 0 else records
    if not args.shared_file:
        raise ValueError("--shared_file is required for shared_file source")
    records = read_jsonl(args.shared_file)
    return records[: args.max_examples] if args.max_examples >= 0 else records


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bank = []
    for prefix_id, record in enumerate(source_records(args, tokenizer)):
        text = str(record.get("reference_text", record.get("generated_text", record.get("response", ""))))
        token_ids = tokenizer(text, add_special_tokens=False, return_token_type_ids=False)["input_ids"]
        if not token_ids:
            continue
        bank.append(
            {
                **record,
                "prefix_id": prefix_id,
                "prefix_text": text,
                "prefix_token_ids": [int(token_id) for token_id in token_ids],
                "prefix_length": len(token_ids),
            }
        )
    write_jsonl(args.output, bank)


if __name__ == "__main__":
    main()
