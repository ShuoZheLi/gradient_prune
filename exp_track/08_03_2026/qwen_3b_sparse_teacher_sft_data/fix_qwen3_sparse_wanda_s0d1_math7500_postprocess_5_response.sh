#!/bin/bash
#SBATCH --job-name=fix_qwen3_sparse_s0d1_n5
#SBATCH --account=ASC26008
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j_fix_qwen3_sparse_s0d1_n5.out
#SBATCH --error=slurm-%j_fix_qwen3_sparse_s0d1_n5.err

set -euo pipefail

# Recover the final parquet files from an already-completed raw response file.
# This script intentionally does not launch generation and does not delete shards
# or raw outputs, so it is safe to run after the original collection job timed out
# during final postprocessing.

if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

find_repo_root() {
  local start_dir="$1"
  local dir
  dir="$(CDPATH= cd -- "$start_dir" 2>/dev/null && pwd)" || return 1
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" && -d "$dir/create_calibration_dataset" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
if [[ -z "$repo_root" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work/09576/shuozhe/gradient_prune" "/work2/09576/shuozhe/gradient_prune" "/data/shuozhe/gradient_prune"; do
    [[ -z "$candidate" ]] && continue
    if repo_root="$(find_repo_root "$candidate")"; then
      break
    fi
  done
fi
if [[ -z "$repo_root" || ! -d "$repo_root" ]]; then
  echo "Could not locate gradient_prune repo. Set WORK_DIR=/path/to/gradient_prune when submitting." >&2
  exit 1
fi
cd "$repo_root"
export PYTHONPATH="${repo_root}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python3}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/tmp/${USER:-shuozhe}}}"
cache_root="${CACHE_ROOT:-${SCRATCH_ROOT}/gradient_prune_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${cache_root}/tiktoken}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

SPARSITY_LABEL="${SPARSITY_LABEL:-s0d1}"
PRUNE_GRANULARITY="${PRUNE_GRANULARITY:-layerwise}"
GRANULARITY_LABEL="${GRANULARITY_LABEL:-${PRUNE_GRANULARITY}}"
TEACHER_ID="${TEACHER_ID:-qwen3_8b_wanda_${SPARSITY_LABEL}_${GRANULARITY_LABEL}}"
PRUNED_TEACHER_DIR="${PRUNED_TEACHER_DIR:-${SCRATCH_ROOT}/gradient_prune/pruned_teachers/${TEACHER_ID}}"
MODEL_PATH="${MODEL_PATH:-$PRUNED_TEACHER_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/saved_calibration_dataset/${TEACHER_ID}_math7500_correct_5_response}"
RAW_JSONL="${RAW_JSONL:-${OUTPUT_DIR}/raw_actor_responses.jsonl}"
ALL_TRAJECTORIES_JSONL="${ALL_TRAJECTORIES_JSONL:-${OUTPUT_DIR}/all_actor_trajectories.jsonl}"
ALL_TRAJECTORIES_PARQUET="${ALL_TRAJECTORIES_PARQUET:-${OUTPUT_DIR}/all_actor_trajectories.parquet}"
CORRECT_JSONL="${CORRECT_JSONL:-${OUTPUT_DIR}/correct_actor_responses.jsonl}"
CALIB_PARQUET="${CALIB_PARQUET:-${OUTPUT_DIR}/${TEACHER_ID}_math7500_correct_5_response.parquet}"
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/metrics.json}"

NUM_RESPONSES_PER_PROMPT="${NUM_RESPONSES_PER_PROMPT:-5}"
MAX_EXAMPLES="${MAX_EXAMPLES:-7500}"
EXPECTED_RAW_LINES="${EXPECTED_RAW_LINES:-$((MAX_EXAMPLES * NUM_RESPONSES_PER_PROMPT))}"
SKIP_MERGE="${SKIP_MERGE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

echo "[fix] repo_root=$repo_root"
echo "[fix] model_path=$MODEL_PATH"
echo "[fix] raw_jsonl=$RAW_JSONL"
echo "[fix] all_trajectories_parquet=$ALL_TRAJECTORIES_PARQUET"
echo "[fix] calib_parquet=$CALIB_PARQUET"

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$RAW_JSONL" ]]; then
  echo "Raw response file not found: $RAW_JSONL" >&2
  exit 1
fi

raw_lines="$(wc -l < "$RAW_JSONL" | tr -d ' ')"
echo "[fix] raw rows: $raw_lines"
if [[ "$EXPECTED_RAW_LINES" != "0" && "$raw_lines" -ne "$EXPECTED_RAW_LINES" ]]; then
  echo "Expected $EXPECTED_RAW_LINES raw responses, got $raw_lines. Set EXPECTED_RAW_LINES=0 to bypass this check." >&2
  exit 1
fi

"$python_bin" - \
  --model_path "$MODEL_PATH" \
  --skip_merge "$SKIP_MERGE" \
  --trust_remote_code "$TRUST_REMOTE_CODE" \
  --raw_jsonl "$RAW_JSONL" \
  --all_trajectories_jsonl "$ALL_TRAJECTORIES_JSONL" \
  --all_trajectories_parquet "$ALL_TRAJECTORIES_PARQUET" \
  --correct_jsonl "$CORRECT_JSONL" \
  --calib_parquet "$CALIB_PARQUET" \
  --metrics_json "$METRICS_JSON" \
  --num_responses_per_prompt "$NUM_RESPONSES_PER_PROMPT" <<'PY'
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from create_calibration_dataset.generate_actor_responses_minimal import resolve_actor_hf_dir


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--skip_merge", required=True)
parser.add_argument("--trust_remote_code", required=True)
parser.add_argument("--raw_jsonl", required=True)
parser.add_argument("--all_trajectories_jsonl", required=True)
parser.add_argument("--all_trajectories_parquet", required=True)
parser.add_argument("--correct_jsonl", required=True)
parser.add_argument("--calib_parquet", required=True)
parser.add_argument("--metrics_json", required=True)
parser.add_argument("--num_responses_per_prompt", type=int, default=1)
args = parser.parse_args()

raw_path = Path(args.raw_jsonl).expanduser()
all_jsonl_path = Path(args.all_trajectories_jsonl).expanduser()
all_parquet_path = Path(args.all_trajectories_parquet).expanduser()
correct_path = Path(args.correct_jsonl).expanduser()
parquet_path = Path(args.calib_parquet).expanduser()
metrics_path = Path(args.metrics_json).expanduser()
for path in (all_jsonl_path, all_parquet_path, correct_path, parquet_path, metrics_path):
    path.parent.mkdir(parents=True, exist_ok=True)

actor_dir = resolve_actor_hf_dir(
    args.model_path,
    skip_merge=args.skip_merge.lower() in {"1", "true", "yes"},
)
tokenizer = AutoTokenizer.from_pretrained(
    actor_dir,
    trust_remote_code=args.trust_remote_code.lower() in {"1", "true", "yes"},
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

all_rows = []
correct_rows = []
prompt_correct = {}
num_total = 0
num_scored = 0
all_jsonl_tmp = Path(f"{all_jsonl_path}.tmp")
correct_tmp = Path(f"{correct_path}.tmp")

with raw_path.open("r", encoding="utf-8") as input_file, all_jsonl_tmp.open("w", encoding="utf-8") as all_file, correct_tmp.open("w", encoding="utf-8") as correct_file:
    for line in input_file:
        if not line.strip():
            continue
        num_total += 1
        row = json.loads(line)
        if "is_correct" in row:
            num_scored += 1

        prompt = row.get("prompt", "")
        response = row.get("response", "")
        trajectory = f"{prompt}{response}"
        trajectory_ids = tokenizer(
            trajectory,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        out_row = {
            "example_id": row.get("example_id"),
            "response_index": row.get("response_index", 0),
            "num_responses_per_prompt": row.get("num_responses_per_prompt", args.num_responses_per_prompt),
            "prompt": prompt,
            "response": response,
            "task_score": row.get("task_score"),
            "is_correct": bool(row.get("is_correct", False)),
            "prompt_generated_trajectory": trajectory,
            "prompt_generated_trajectory_ids": trajectory_ids,
        }
        all_rows.append(out_row)
        all_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        if out_row["is_correct"]:
            prompt_correct[row.get("example_id")] = True
            correct_rows.append(out_row)
            correct_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")

if not correct_rows:
    raise SystemExit("No correct trajectories were collected; inspect raw responses before using this dataset.")

all_parquet_tmp = all_parquet_path.with_name(f".{all_parquet_path.name}.tmp.parquet")
calib_parquet_tmp = parquet_path.with_name(f".{parquet_path.name}.tmp.parquet")
metrics_tmp = Path(f"{metrics_path}.tmp")

pd.DataFrame(all_rows).to_parquet(all_parquet_tmp, index=False)
pd.DataFrame(correct_rows).to_parquet(calib_parquet_tmp, index=False)
prompt_count = len({row.get("example_id") for row in all_rows})
metrics = {
    "num_total": num_total,
    "num_prompts": prompt_count,
    "num_responses_per_prompt": args.num_responses_per_prompt,
    "num_prompts_with_correct_response": sum(1 for value in prompt_correct.values() if value),
    "prompt_pass_rate": (sum(1 for value in prompt_correct.values() if value) / prompt_count) if prompt_count else None,
    "num_scored": num_scored,
    "num_correct": len(correct_rows),
    "accuracy": (len(correct_rows) / num_scored) if num_scored else None,
    "response_accuracy": (len(correct_rows) / num_scored) if num_scored else None,
    "raw_jsonl": str(raw_path),
    "all_trajectories_jsonl": str(all_jsonl_path),
    "all_trajectories_parquet": str(all_parquet_path),
    "correct_jsonl": str(correct_path),
    "calib_parquet": str(parquet_path),
}
metrics_tmp.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

os.replace(all_jsonl_tmp, all_jsonl_path)
os.replace(correct_tmp, correct_path)
os.replace(all_parquet_tmp, all_parquet_path)
os.replace(calib_parquet_tmp, parquet_path)
os.replace(metrics_tmp, metrics_path)
print(json.dumps(metrics, indent=2))
PY

echo "[done] all trajectories parquet: $ALL_TRAJECTORIES_PARQUET"
echo "[done] correct trajectories parquet: $CALIB_PARQUET"
echo "[done] use with PUNE as calib_data=$CALIB_PARQUET"
