#!/bin/bash
#SBATCH --job-name=recover_nemotron_nano_9b_v2_shard1
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j_recover_nemotron_nano_9b_v2_shard1.out
#SBATCH --error=slurm-%j_recover_nemotron_nano_9b_v2_shard1.err

set -euo pipefail

experiment_start_epoch=$(date +%s)
format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

on_exit() {
  local status="$?"
  local experiment_end_epoch elapsed
  experiment_end_epoch=$(date +%s)
  elapsed=$((experiment_end_epoch - experiment_start_epoch))
  echo "[recover] total_elapsed_seconds=$elapsed"
  echo "[recover] total_elapsed_time=$(format_duration "$elapsed")"
  exit "$status"
}
trap on_exit EXIT

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
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work/09576/shuozhe/gradient_prune" "/data/shuozhe/gradient_prune"; do
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
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python3}"

RUN_NAME="${RUN_NAME:-recover_nvidia_nemotron_nano_9b_v2_math7500_shard_1}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}}"

cache_root="${CACHE_ROOT:-${SCRATCH:-/tmp}/${USER:-shuozhe}/gradient_prune_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules/${RUN_ID}}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${cache_root}/tiktoken}"
export PYTHONUNBUFFERED=1
export TASK_SCORER_BACKEND="${TASK_SCORER_BACKEND:-verl_math_reward}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

model_path="${MODEL_PATH:-/work/09576/shuozhe/saved_model/NVIDIA-Nemotron-Nano-9B-v2}"
dataset_path="${DATASET_PATH:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/math7500.parquet}"
output_dir="${OUTPUT_DIR:-$repo_root/saved_calibration_dataset/nvidia-nemotron-nano-9b-v2_math7500_correct_5_response}"
raw_jsonl="${RAW_JSONL:-$output_dir/raw_actor_responses.jsonl}"
shard_dir="${SHARD_DIR:-$output_dir/shards}"
log_dir="${LOG_DIR:-$output_dir/logs/${RUN_ID}}"
all_trajectories_jsonl="${ALL_TRAJECTORIES_JSONL:-$output_dir/all_actor_trajectories.jsonl}"
all_trajectories_parquet="${ALL_TRAJECTORIES_PARQUET:-$output_dir/all_actor_trajectories.parquet}"
correct_jsonl="${CORRECT_JSONL:-$output_dir/correct_actor_responses.jsonl}"
calib_parquet="${CALIB_PARQUET:-$output_dir/nvidia-nemotron-nano-9b-v2_math7500_correct_5_response.parquet}"
metrics_json="${METRICS_JSON:-$output_dir/metrics.json}"

max_examples="${MAX_EXAMPLES:-7500}"
start_index="${START_INDEX:-0}"
num_shards="${NUM_SHARDS:-8}"
target_shard_id="${TARGET_SHARD_ID:-1}"
seed="${SEED:-42}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${MAX_NEW_TOKENS:-16384}"
batch_size="${BATCH_SIZE:-64}"
generation_max_batch_tokens="${GENERATION_MAX_BATCH_TOKENS:-0}"
response_log_max="${RESPONSE_LOG_MAX:--1}"
multi_response_temperature="${MULTI_RESPONSE_TEMPERATURE:-0.7}"
num_responses_per_prompt="${NUM_RESPONSES_PER_PROMPT:-5}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-0.95}"
top_k="${TOP_K:-0}"
dtype="${DTYPE:-auto}"
local_device_spec="${LOCAL_DEVICE:-${DEVICES:-0}}"
read -r local_device _ <<< "$local_device_spec"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
enforce_eager="${ENFORCE_EAGER:-1}"
enable_thinking="${ENABLE_THINKING:-true}"
response_key="${RESPONSE_KEY:-}"
reward_score_dir="${REWARD_SCORE_DIR:-}"
skip_generation="${SKIP_GENERATION:-0}"
force_rerun="${FORCE_RERUN:-0}"

if [[ "$max_examples" -lt 0 ]]; then
  echo "MAX_EXAMPLES must be >= 0; got $max_examples" >&2
  exit 2
fi
if [[ "$num_shards" -lt 1 ]]; then
  echo "NUM_SHARDS must be >= 1; got $num_shards" >&2
  exit 2
fi
if [[ "$target_shard_id" -lt 0 || "$target_shard_id" -ge "$num_shards" ]]; then
  echo "TARGET_SHARD_ID must be in [0, $((num_shards - 1))]; got $target_shard_id" >&2
  exit 2
fi

mkdir -p "$output_dir" "$shard_dir" "$log_dir"

base_count=$((max_examples / num_shards))
remainder=$((max_examples % num_shards))
expected_raw_lines=$((max_examples * num_responses_per_prompt))

shard_count_for() {
  local shard_id="$1"
  local count="$base_count"
  if [[ "$shard_id" -lt "$remainder" ]]; then
    count=$((count + 1))
  fi
  printf '%s' "$count"
}

shard_start_for() {
  local shard_id="$1"
  if [[ "$shard_id" -lt "$remainder" ]]; then
    printf '%s' $((start_index + shard_id * (base_count + 1)))
  else
    printf '%s' $((start_index + remainder * (base_count + 1) + (shard_id - remainder) * base_count))
  fi
}

line_count_or_zero() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path" | tr -d ' '
  else
    printf '0'
  fi
}

echo "[recover] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[recover] Run ID: $RUN_ID"
echo "[recover] repo_root=$repo_root"
echo "[recover] model_path=$model_path"
echo "[recover] dataset_path=$dataset_path"
echo "[recover] output_dir=$output_dir"
echo "[recover] shard_dir=$shard_dir"
echo "[recover] target_shard_id=$target_shard_id num_shards=$num_shards"
echo "[recover] cache_root=$cache_root"
echo "[recover] HF_MODULES_CACHE=$HF_MODULES_CACHE"

if [[ ! -e "$model_path" ]]; then
  echo "Model path not found: $model_path" >&2
  exit 1
fi
if [[ ! -f "$dataset_path" ]]; then
  echo "Dataset path not found: $dataset_path" >&2
  exit 1
fi

echo "[recover] existing shard line counts:"
for ((idx = 0; idx < num_shards; idx++)); do
  shard_output="$shard_dir/raw_actor_responses_shard_${idx}.jsonl"
  shard_count="$(shard_count_for "$idx")"
  expected_shard_lines=$((shard_count * num_responses_per_prompt))
  printf '[recover] shard_%s lines=%s expected=%s file=%s\n' "$idx" "$(line_count_or_zero "$shard_output")" "$expected_shard_lines" "$shard_output"
done

target_shard_count="$(shard_count_for "$target_shard_id")"
target_shard_start="$(shard_start_for "$target_shard_id")"
target_expected_lines=$((target_shard_count * num_responses_per_prompt))
target_output="$shard_dir/raw_actor_responses_shard_${target_shard_id}.jsonl"
target_metrics="$shard_dir/metrics_shard_${target_shard_id}.json"
target_current_lines="$(line_count_or_zero "$target_output")"

if [[ "$skip_generation" != "1" ]]; then
  if [[ "$target_current_lines" -eq "$target_expected_lines" && "$force_rerun" != "1" ]]; then
    echo "[recover] target shard already has expected line count; set FORCE_RERUN=1 to regenerate it."
  else
    tmp_output="$target_output.recovery_${RUN_ID}.tmp"
    tmp_metrics="$target_metrics.recovery_${RUN_ID}.tmp"
    rm -f "$tmp_output" "$tmp_metrics"

    echo "[recover] regenerating shard $target_shard_id start=$target_shard_start count=$target_shard_count expected_lines=$target_expected_lines"
    cmd=(
      create_calibration_dataset/vllm_accuracy_runner.py
      --model_path "$model_path"
      --dataset_path "$dataset_path"
      --output_path "$tmp_output"
      --metrics_path "$tmp_metrics"
      --prompt_key prompt
      --start_index "$target_shard_start"
      --max_examples "$target_shard_count"
      --seed "$seed"
      --max_prompt_length "$max_prompt_length"
      --max_new_tokens "$max_new_tokens"
      --batch_size "$batch_size"
      --generation_max_batch_tokens "$generation_max_batch_tokens"
      --response_log_max "$response_log_max"
      --num-responses-per-prompt "$num_responses_per_prompt"
      --multi-response-temperature "$multi_response_temperature"
      --temperature "$temperature"
      --top_p "$top_p"
      --top_k "$top_k"
      --tensor_parallel_size "$tensor_parallel_size"
      --gpu_memory_utilization "$gpu_memory_utilization"
      --dtype "$dtype"
      --enable-thinking "$enable_thinking"
    )
    if [[ "$enforce_eager" == "1" ]]; then
      cmd+=(--enforce_eager)
    else
      cmd+=(--no_enforce_eager)
    fi
    [[ -n "$response_key" ]] && cmd+=(--response_key "$response_key")
    [[ -n "$reward_score_dir" ]] && cmd+=(--reward_score_dir "$reward_score_dir")

    printf '[recover] command:'
    printf ' %q' env "CUDA_VISIBLE_DEVICES=$local_device" "$python_bin" "${cmd[@]}"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$local_device" "$python_bin" "${cmd[@]}" 2>&1 | tee "$log_dir/shard_${target_shard_id}_recovery.log"

    tmp_lines="$(line_count_or_zero "$tmp_output")"
    if [[ "$tmp_lines" -ne "$target_expected_lines" ]]; then
      echo "Recovered shard has $tmp_lines lines, expected $target_expected_lines. Keeping temp file for inspection: $tmp_output" >&2
      exit 1
    fi
    mv -f "$tmp_output" "$target_output"
    mv -f "$tmp_metrics" "$target_metrics"
    echo "[recover] replaced $target_output with $tmp_lines recovered rows"
  fi
else
  echo "[recover] SKIP_GENERATION=1; only validating and merging existing shards."
fi

echo "[recover] validating all shard line counts before merge"
for ((idx = 0; idx < num_shards; idx++)); do
  shard_output="$shard_dir/raw_actor_responses_shard_${idx}.jsonl"
  shard_count="$(shard_count_for "$idx")"
  expected_shard_lines=$((shard_count * num_responses_per_prompt))
  shard_lines="$(line_count_or_zero "$shard_output")"
  if [[ "$shard_lines" -ne "$expected_shard_lines" ]]; then
    echo "Shard $idx has $shard_lines lines, expected $expected_shard_lines: $shard_output" >&2
    exit 1
  fi
done

tmp_raw="$raw_jsonl.recovery_${RUN_ID}.tmp"
: > "$tmp_raw"
for ((idx = 0; idx < num_shards; idx++)); do
  cat "$shard_dir/raw_actor_responses_shard_${idx}.jsonl" >> "$tmp_raw"
done
raw_lines="$(line_count_or_zero "$tmp_raw")"
if [[ "$raw_lines" -ne "$expected_raw_lines" ]]; then
  echo "Merged raw file has $raw_lines lines, expected $expected_raw_lines. Keeping temp file: $tmp_raw" >&2
  exit 1
fi
mv -f "$tmp_raw" "$raw_jsonl"
echo "[recover] merged raw responses: $raw_lines rows -> $raw_jsonl"

"$python_bin" - \
  --model_path "$model_path" \
  --raw_jsonl "$raw_jsonl" \
  --all_trajectories_jsonl "$all_trajectories_jsonl" \
  --all_trajectories_parquet "$all_trajectories_parquet" \
  --correct_jsonl "$correct_jsonl" \
  --calib_parquet "$calib_parquet" \
  --metrics_json "$metrics_json" \
  --num_responses_per_prompt "$num_responses_per_prompt" \
  --start_index "$start_index" \
  --max_examples "$max_examples" <<'PY'
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from create_calibration_dataset.vllm_accuracy_runner import _ensure_pad_token, _load_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--raw_jsonl", required=True)
parser.add_argument("--all_trajectories_jsonl", required=True)
parser.add_argument("--all_trajectories_parquet", required=True)
parser.add_argument("--correct_jsonl", required=True)
parser.add_argument("--calib_parquet", required=True)
parser.add_argument("--metrics_json", required=True)
parser.add_argument("--num_responses_per_prompt", type=int, required=True)
parser.add_argument("--start_index", type=int, required=True)
parser.add_argument("--max_examples", type=int, required=True)
args = parser.parse_args()

raw_path = Path(args.raw_jsonl).expanduser()
all_jsonl_path = Path(args.all_trajectories_jsonl).expanduser()
all_parquet_path = Path(args.all_trajectories_parquet).expanduser()
correct_path = Path(args.correct_jsonl).expanduser()
parquet_path = Path(args.calib_parquet).expanduser()
metrics_path = Path(args.metrics_json).expanduser()
for path in (all_jsonl_path, all_parquet_path, correct_path, parquet_path, metrics_path):
    path.parent.mkdir(parents=True, exist_ok=True)

raw_rows = []
pair_counts = Counter()
example_counts = Counter()
with raw_path.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row.get("example_id")
        response_index = row.get("response_index", 0)
        pair_counts[(example_id, response_index)] += 1
        example_counts[example_id] += 1
        raw_rows.append(row)

expected_total = args.max_examples * args.num_responses_per_prompt
if len(raw_rows) != expected_total:
    raise SystemExit(f"Expected {expected_total} raw rows, got {len(raw_rows)}")
expected_ids = set(range(args.start_index, args.start_index + args.max_examples))
actual_ids = set(example_counts)
missing_ids = sorted(expected_ids - actual_ids)
extra_ids = sorted(actual_ids - expected_ids)
bad_counts = sorted((example_id, count) for example_id, count in example_counts.items() if count != args.num_responses_per_prompt)
duplicate_pairs = sorted((example_id, response_index, count) for (example_id, response_index), count in pair_counts.items() if count != 1)
if missing_ids or extra_ids or bad_counts or duplicate_pairs:
    summary = {
        "missing_ids_head": missing_ids[:20],
        "num_missing_ids": len(missing_ids),
        "extra_ids_head": extra_ids[:20],
        "num_extra_ids": len(extra_ids),
        "bad_counts_head": bad_counts[:20],
        "num_bad_counts": len(bad_counts),
        "duplicate_pairs_head": duplicate_pairs[:20],
        "num_duplicate_pairs": len(duplicate_pairs),
    }
    raise SystemExit("Raw shard validation failed: " + json.dumps(summary, indent=2))

tokenizer = _load_tokenizer(args.model_path)
_ensure_pad_token(tokenizer, args.model_path)

all_rows = []
correct_rows = []
prompt_correct = {}
num_scored = 0
with all_jsonl_path.open("w", encoding="utf-8") as all_file, correct_path.open("w", encoding="utf-8") as correct_file:
    for row in raw_rows:
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

pd.DataFrame(all_rows).to_parquet(all_parquet_path, index=False)
pd.DataFrame(correct_rows).to_parquet(parquet_path, index=False)
num_prompts_with_correct_response = sum(1 for value in prompt_correct.values() if value)
metrics = {
    "num_total": len(raw_rows),
    "num_prompts": len(example_counts),
    "num_responses_per_prompt": args.num_responses_per_prompt,
    "num_prompts_with_correct_response": num_prompts_with_correct_response,
    "prompt_pass_rate": num_prompts_with_correct_response / len(example_counts) if example_counts else None,
    "num_scored": num_scored,
    "num_correct": len(correct_rows),
    "accuracy": len(correct_rows) / num_scored if num_scored else None,
    "response_accuracy": len(correct_rows) / num_scored if num_scored else None,
    "raw_jsonl": str(raw_path),
    "all_trajectories_jsonl": str(all_jsonl_path),
    "all_trajectories_parquet": str(all_parquet_path),
    "correct_jsonl": str(correct_path),
    "calib_parquet": str(parquet_path),
}
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
if not correct_rows:
    raise SystemExit("No correct trajectories were collected; try sampling or inspect raw responses.")
PY

echo "[done] raw responses: $raw_jsonl"
echo "[done] all trajectories parquet: $all_trajectories_parquet"
echo "[done] correct trajectories parquet: $calib_parquet"
echo "[done] use with PUNE as calib_data=$calib_parquet"
