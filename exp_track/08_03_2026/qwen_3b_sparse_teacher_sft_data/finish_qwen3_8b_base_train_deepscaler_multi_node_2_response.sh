#!/bin/bash
#SBATCH --job-name=finish_qwen3_8b_base_deepscaler_n2
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
# For multi-GPU nodes, set --ntasks-per-node to the number of LOCAL_DEVICES.
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j_finish_qwen3_8b_base_deepscaler_n2.out
#SBATCH --error=slurm-%j_finish_qwen3_8b_base_deepscaler_n2.err

set -euo pipefail

experiment_start_epoch=$(date +%s)
format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

# Resume/finalize collector for 2-response trajectories from the unpruned Qwen3-8B base model
# on MetaMathQA train_deepscaler. This script does not apply pruning, does not call the older 08/01 wrapper
# script, and preserves completed shard JSONL files from an interrupted run.

# -----------------------------
# Environment setup
# -----------------------------
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

# Slurm may execute a spooled copy under /var/spool. Prefer explicit overrides,
# then discover the repo by walking up from the submit/current/script directory.
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

cache_root="${CACHE_ROOT:-${SCRATCH:-/tmp}/${USER:-shuozhe}/gradient_prune_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
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
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

# -----------------------------
# Paths and collection config
# -----------------------------
RUN_NAME="${RUN_NAME:-finish_qwen3_8b_base_train_deepscaler_2_response}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}}"

model_path="${MODEL_PATH:-${BASE_MODEL_PATH:-${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-8B}}}"
dataset_path="${DATASET_PATH:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/train_deepscaler.parquet}"
output_dir="${OUTPUT_DIR:-$repo_root/saved_calibration_dataset/qwen3_8b_base_train_deepscaler_correct_2_response}"
raw_jsonl="${RAW_JSONL:-$output_dir/raw_actor_responses.jsonl}"
shard_dir="${SHARD_DIR:-$output_dir/shards}"
log_dir="${LOG_DIR:-$output_dir/logs/${RUN_ID}}"
all_trajectories_jsonl="${ALL_TRAJECTORIES_JSONL:-$output_dir/all_actor_trajectories.jsonl}"
all_trajectories_parquet="${ALL_TRAJECTORIES_PARQUET:-$output_dir/all_actor_trajectories.parquet}"
correct_jsonl="${CORRECT_JSONL:-$output_dir/correct_actor_responses.jsonl}"
calib_parquet="${CALIB_PARQUET:-$output_dir/qwen3_8b_base_train_deepscaler_correct_2_response.parquet}"
metrics_json="${METRICS_JSON:-$output_dir/metrics.json}"

max_examples="${MAX_EXAMPLES:--1}"
start_index="${START_INDEX:-0}"
seed="${SEED:-42}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${MAX_NEW_TOKENS:-16384}"
generation_backend="${GENERATION_BACKEND:-vllm}"
batch_size="${BATCH_SIZE:-64}"
generation_max_batch_tokens="${GENERATION_MAX_BATCH_TOKENS:-0}"
response_log_max="${RESPONSE_LOG_MAX:--1}"
multi_response_temperature="${MULTI_RESPONSE_TEMPERATURE:-0.7}"
num_responses_per_prompt="${NUM_RESPONSES_PER_PROMPT:-2}"
use_cache="${USE_CACHE:-0}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-0.95}"
top_k="${TOP_K:-0}"
dtype="${DTYPE:-auto}"
local_devices="${LOCAL_DEVICES:-${DEVICES:-0}}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
enforce_eager="${ENFORCE_EAGER:-1}"
response_key="${RESPONSE_KEY:-}"
reward_score_dir="${REWARD_SCORE_DIR:-}"
trust_remote_code="${TRUST_REMOTE_CODE:-0}"
enable_thinking="${ENABLE_THINKING:-true}"
skip_merge="${SKIP_MERGE:-1}"
progress_interval="${PROGRESS_INTERVAL:-5}"
dry_run="${DRY_RUN:-0}"

if [[ "$num_responses_per_prompt" -lt 1 ]]; then
  echo "NUM_RESPONSES_PER_PROMPT must be >= 1; got $num_responses_per_prompt" >&2
  exit 2
fi
if [[ "$num_responses_per_prompt" -gt 1 && "$generation_backend" != "vllm" ]]; then
  echo "NUM_RESPONSES_PER_PROMPT > 1 requires GENERATION_BACKEND=vllm; got $generation_backend" >&2
  exit 2
fi
if [[ "$start_index" -lt 0 ]]; then
  echo "START_INDEX must be >= 0; got $start_index" >&2
  exit 2
fi
if [[ "$max_examples" -lt -1 ]]; then
  echo "MAX_EXAMPLES must be -1 for all examples, or >= 0; got $max_examples" >&2
  exit 2
fi
if [[ "$max_examples" -eq -1 ]]; then
  if [[ "$dry_run" != "1" && ! -f "$dataset_path" ]]; then
    echo "Dataset path does not exist: $dataset_path" >&2
    exit 2
  fi
  dataset_total_examples=$("$python_bin" - "$dataset_path" <<'PY_COUNT_ROWS'
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import pyarrow.parquet as pq
    print(pq.ParquetFile(path).metadata.num_rows)
except Exception:
    import pandas as pd
    print(len(pd.read_parquet(path, columns=[])))
PY_COUNT_ROWS
)
  max_examples=$((dataset_total_examples - start_index))
  if [[ "$max_examples" -lt 0 ]]; then
    max_examples=0
  fi
  echo "[collect] MAX_EXAMPLES=-1 resolved to $max_examples examples from $dataset_total_examples dataset rows starting at START_INDEX=$start_index"
fi
expected_raw_lines=$((max_examples * num_responses_per_prompt))

mkdir -p "$output_dir" "$shard_dir" "$log_dir"

if [[ "$batch_size" -lt 1 ]]; then
  echo "BATCH_SIZE must be >= 1; got $batch_size" >&2
  exit 2
fi
if [[ "$generation_backend" != "vllm" && "$generation_backend" != "transformers" ]]; then
  echo "GENERATION_BACKEND must be 'vllm' or 'transformers'; got $generation_backend" >&2
  exit 2
fi

read -r -a local_devices_array <<< "$local_devices"
if [[ ${#local_devices_array[@]} -lt 1 ]]; then
  echo "LOCAL_DEVICES/DEVICES must contain at least one GPU id per node." >&2
  exit 2
fi

# -----------------------------
# Slurm node discovery
# -----------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
  nodes_array=("$(hostname)")
fi
if [[ ${#nodes_array[@]} -lt 1 ]]; then
  echo "Could not discover any Slurm nodes." >&2
  exit 1
fi

num_nodes=${#nodes_array[@]}
gpus_per_node=${#local_devices_array[@]}
num_shards=$((num_nodes * gpus_per_node))
if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "$gpus_per_node" -gt 1 ]]; then
  ntasks_per_node_value="${SLURM_NTASKS_PER_NODE%%(*}"
  if [[ "$ntasks_per_node_value" =~ ^[0-9]+$ && "$ntasks_per_node_value" -lt "$gpus_per_node" ]]; then
    echo "LOCAL_DEVICES requests $gpus_per_node shards per node, but Slurm allocated only $SLURM_NTASKS_PER_NODE task(s) per node." >&2
    echo "Set #SBATCH --ntasks-per-node=$gpus_per_node or reduce LOCAL_DEVICES." >&2
    exit 2
  fi
fi

# Keep model merge products out of WORK/repo. If an FSDP checkpoint needs merging,
# copy/merge from a scratch-local path instead of writing under MODEL_PATH.
if [[ "$skip_merge" != "1" ]]; then
  source_model_path="$model_path"
  merged_root="${MERGED_MODEL_ROOT:-${SCRATCH:-/tmp}/${USER:-shuozhe}/gradient_prune_merged/${RUN_ID}}"
  model_path="${merged_root}/model_checkpoint"
  if [[ ! -e "$model_path" ]]; then
    mkdir -p "$(dirname "$model_path")"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude='merged_hf/' "${source_model_path%/}/" "${model_path}/"
    else
      cp -a "$source_model_path" "$model_path"
    fi
  fi
fi

# -----------------------------
# Helpers
# -----------------------------
run_generation_shard() {
  local shard_id="$1"
  local shard_start="$2"
  local shard_count="$3"
  local shard_device="$4"
  local shard_output="$5"

  local cmd=()
  if [[ "$generation_backend" == "vllm" ]]; then
    local actor_dir="$model_path"
    if [[ "$skip_merge" != "1" ]]; then
      actor_dir=$("$python_bin" - "$model_path" <<'PYRESOLVE'
import sys
from create_calibration_dataset.generate_actor_responses_minimal import resolve_actor_hf_dir
print(resolve_actor_hf_dir(sys.argv[1], skip_merge=False))
PYRESOLVE
)
    fi
    local shard_metrics="$shard_dir/metrics_shard_${shard_id}.json"
    cmd=(
      create_calibration_dataset/vllm_accuracy_runner.py
      --model_path "$actor_dir"
      --dataset_path "$dataset_path"
      --output_path "$shard_output"
      --metrics_path "$shard_metrics"
      --prompt_key prompt
      --start_index "$shard_start"
      --max_examples "$shard_count"
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
  else
    cmd=(
      create_calibration_dataset/generate_actor_responses_minimal.py
      --checkpoint_dir "$model_path"
      --dataset_path "$dataset_path"
      --output_path "$shard_output"
      --prompt_key prompt
      --start_index "$shard_start"
      --max_examples "$shard_count"
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
      --dtype "${TRANSFORMERS_DTYPE:-bf16}"
      --device cuda:0
      --enable-thinking "$enable_thinking"
    )
    [[ "$use_cache" == "1" ]] && cmd+=(--use_cache)
    [[ "$trust_remote_code" == "1" ]] && cmd+=(--trust_remote_code)
    [[ "$skip_merge" == "1" ]] && cmd+=(--skip_merge)
  fi

  [[ -n "$response_key" ]] && cmd+=(--response_key "$response_key")
  [[ -n "$reward_score_dir" ]] && cmd+=(--reward_score_dir "$reward_score_dir")

  echo "[collect][shard $shard_id] backend=$generation_backend gpu=$shard_device start=$shard_start count=$shard_count output=$shard_output"
  if [[ "$dry_run" == "1" ]]; then
    printf '[collect][shard %s] command:' "$shard_id"
    printf ' %q' env "CUDA_VISIBLE_DEVICES=$shard_device" "$python_bin" "${cmd[@]}"
    printf '\n'
    return 0
  fi
  env CUDA_VISIBLE_DEVICES="$shard_device" "$python_bin" "${cmd[@]}"
}

launch_on_node() {
  local node="$1"
  local inner_cmd="$2"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --nodes=1 --ntasks=1 -w "$node" bash -lc "$inner_cmd"
  else
    bash -lc "$inner_cmd"
  fi
}

progress_bar() {
  local current="$1"
  local total="$2"
  local width=40
  local percent filled empty bar
  if [[ "$total" -le 0 ]]; then
    percent=100
    filled=$width
  else
    percent=$((current * 100 / total))
    filled=$((current * width / total))
  fi
  empty=$((width - filled))
  bar=""
  while [[ ${#bar} -lt "$filled" ]]; do bar="${bar}#"; done
  while [[ "$empty" -gt 0 ]]; do bar="${bar}-"; empty=$((empty - 1)); done
  printf '\r[collect] progress [%s] %s/%s (%s%%)' "$bar" "$current" "$total" "$percent"
}

progress_count() {
  local completed=0 shard_output shard_lines
  for shard_output in "$shard_dir"/raw_actor_responses_shard_*.jsonl; do
    if [[ -f "$shard_output" ]]; then
      shard_lines=$(wc -l < "$shard_output" | tr -d ' ')
      completed=$((completed + shard_lines))
    fi
  done
  if [[ "$completed" -gt "$expected_raw_lines" ]]; then
    completed="$expected_raw_lines"
  fi
  printf '%s' "$completed"
}

progress_monitor() {
  while :; do
    progress_bar "$(progress_count)" "$expected_raw_lines"
    sleep "$progress_interval"
  done
}

cleanup() {
  if [[ -n "${progress_pid:-}" ]]; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
  fi
}

on_exit() {
  local status="$?"
  cleanup || true
  local experiment_end_epoch elapsed
  experiment_end_epoch=$(date +%s)
  elapsed=$((experiment_end_epoch - experiment_start_epoch))
  echo "[collect] total_elapsed_seconds=$elapsed"
  echo "[collect] total_elapsed_time=$(format_duration "$elapsed")"
  exit "$status"
}
trap on_exit EXIT

# -----------------------------
# Debug info
# -----------------------------
echo "[collect] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[collect] Run ID: $RUN_ID"
echo "[collect] repo_root=$repo_root"
echo "[collect] nodes=${nodes_array[*]}"
echo "[collect] local_devices=$local_devices"
echo "[collect] num_shards=$num_shards"
echo "[collect] model_path=$model_path"
echo "[collect] dataset_path=$dataset_path"
echo "[collect] standalone=1 unpruned_qwen3_base=1"
echo "[collect] output_dir=$output_dir"
echo "[collect] raw_jsonl=$raw_jsonl"
echo "[collect] cache_root=$cache_root"
echo "[collect] generation_backend=$generation_backend batch_size=$batch_size generation_max_batch_tokens=$generation_max_batch_tokens use_cache=$use_cache enable_thinking=$enable_thinking num_responses_per_prompt=$num_responses_per_prompt multi_response_temperature=$multi_response_temperature"
echo "[collect] vllm tensor_parallel_size=$tensor_parallel_size gpu_memory_utilization=$gpu_memory_utilization enforce_eager=$enforce_eager"

if [[ ! -d "$repo_root" ]]; then
  echo "Repo root not found: $repo_root" >&2
  exit 1
fi
if [[ ! -e "$model_path" ]]; then
  echo "Model path not found: $model_path" >&2
  exit 1
fi
if [[ ! -f "$dataset_path" ]]; then
  echo "Dataset path not found: $dataset_path" >&2
  exit 1
fi

# Preserve completed/interrupted shard JSONL files from the timed-out run.
# Only remove aggregate/final outputs and stale logs; incomplete shards are resumed below.
rm -f "$raw_jsonl" "$all_trajectories_jsonl" "$all_trajectories_parquet" "$correct_jsonl" \
  "$calib_parquet" "$metrics_json" "$log_dir"/shard_*.log "$shard_dir"/resume_shard_*.jsonl

base_count=$((max_examples / num_shards))
remainder=$((max_examples % num_shards))
pids=()
shard_index=0

# -----------------------------
# Launch generation shards
# -----------------------------
for node in "${nodes_array[@]}"; do
  for gpu_id in "${local_devices_array[@]}"; do
    shard_count=$base_count
    if [[ "$shard_index" -lt "$remainder" ]]; then
      shard_count=$((shard_count + 1))
    fi

    if [[ "$shard_count" -gt 0 ]]; then
      if [[ "$shard_index" -lt "$remainder" ]]; then
        shard_start=$((start_index + shard_index * (base_count + 1)))
      else
        shard_start=$((start_index + remainder * (base_count + 1) + (shard_index - remainder) * base_count))
      fi
      shard_output="$shard_dir/raw_actor_responses_shard_${shard_index}.jsonl"
      shard_log="$log_dir/shard_${shard_index}_${node}.log"
      expected_shard_lines=$((shard_count * num_responses_per_prompt))
      existing_shard_lines=0
      if [[ -f "$shard_output" ]]; then
        existing_shard_lines=$(wc -l < "$shard_output" | tr -d ' ')
      fi

      launch_start="$shard_start"
      launch_count="$shard_count"
      launch_output="$shard_output"
      if [[ "$existing_shard_lines" -eq "$expected_shard_lines" ]]; then
        echo "[collect][resume] shard $shard_index complete: $existing_shard_lines/$expected_shard_lines rows; skipping"
        shard_index=$((shard_index + 1))
        continue
      elif [[ "$existing_shard_lines" -gt 0 && "$existing_shard_lines" -lt "$expected_shard_lines" && $((existing_shard_lines % num_responses_per_prompt)) -eq 0 ]]; then
        completed_prompts=$((existing_shard_lines / num_responses_per_prompt))
        launch_start=$((shard_start + completed_prompts))
        launch_count=$((shard_count - completed_prompts))
        launch_output="$shard_dir/resume_shard_${shard_index}_${SLURM_JOB_ID:-manual}.jsonl"
        rm -f "$launch_output" "$shard_dir/metrics_shard_${shard_index}.json"
        echo "[collect][resume] shard $shard_index partial: $existing_shard_lines/$expected_shard_lines rows; resuming $launch_count prompts at index $launch_start"
      else
        if [[ "$existing_shard_lines" -ne 0 ]]; then
          echo "[collect][resume] shard $shard_index has unusable row count $existing_shard_lines/$expected_shard_lines; regenerating whole shard"
        else
          echo "[collect][resume] shard $shard_index missing; generating whole shard"
        fi
        rm -f "$shard_output" "$shard_dir/metrics_shard_${shard_index}.json"
      fi

      inner_cmd="cd $(printf '%q' "$repo_root") || exit 1; source $(printf '%q' "${VENV}/bin/activate") 2>/dev/null || true; $(declare -f run_generation_shard); python_bin=$(printf '%q' "$python_bin"); model_path=$(printf '%q' "$model_path"); dataset_path=$(printf '%q' "$dataset_path"); shard_dir=$(printf '%q' "$shard_dir"); generation_backend=$(printf '%q' "$generation_backend"); skip_merge=$(printf '%q' "$skip_merge"); seed=$(printf '%q' "$seed"); max_prompt_length=$(printf '%q' "$max_prompt_length"); max_new_tokens=$(printf '%q' "$max_new_tokens"); batch_size=$(printf '%q' "$batch_size"); generation_max_batch_tokens=$(printf '%q' "$generation_max_batch_tokens"); response_log_max=$(printf '%q' "$response_log_max"); num_responses_per_prompt=$(printf '%q' "$num_responses_per_prompt"); multi_response_temperature=$(printf '%q' "$multi_response_temperature"); temperature=$(printf '%q' "$temperature"); top_p=$(printf '%q' "$top_p"); top_k=$(printf '%q' "$top_k"); tensor_parallel_size=$(printf '%q' "$tensor_parallel_size"); gpu_memory_utilization=$(printf '%q' "$gpu_memory_utilization"); dtype=$(printf '%q' "$dtype"); enforce_eager=$(printf '%q' "$enforce_eager"); use_cache=$(printf '%q' "$use_cache"); trust_remote_code=$(printf '%q' "$trust_remote_code"); enable_thinking=$(printf '%q' "$enable_thinking"); response_key=$(printf '%q' "$response_key"); reward_score_dir=$(printf '%q' "$reward_score_dir"); dry_run=$(printf '%q' "$dry_run"); export PYTHONPATH=$(printf '%q' "$PYTHONPATH") UV_CACHE_DIR=$(printf '%q' "$UV_CACHE_DIR") HF_HOME=$(printf '%q' "$HF_HOME") TRANSFORMERS_CACHE=$(printf '%q' "$TRANSFORMERS_CACHE") HF_DATASETS_CACHE=$(printf '%q' "$HF_DATASETS_CACHE") TORCH_HOME=$(printf '%q' "$TORCH_HOME") TRITON_CACHE_DIR=$(printf '%q' "$TRITON_CACHE_DIR") XDG_CACHE_HOME=$(printf '%q' "$XDG_CACHE_HOME") TIKTOKEN_ENCODINGS_BASE=$(printf '%q' "$TIKTOKEN_ENCODINGS_BASE") PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=$(printf '%q' "$TOKENIZERS_PARALLELISM") VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=$(printf '%q' "$VLLM_WORKER_MULTIPROC_METHOD") VLLM_USE_V1=$(printf '%q' "$VLLM_USE_V1"); run_generation_shard $(printf '%q' "$shard_index") $(printf '%q' "$launch_start") $(printf '%q' "$launch_count") $(printf '%q' "$gpu_id") $(printf '%q' "$launch_output"); if [[ $(printf '%q' "$launch_output") != $(printf '%q' "$shard_output") ]]; then cat $(printf '%q' "$launch_output") >> $(printf '%q' "$shard_output"); rm -f $(printf '%q' "$launch_output"); fi"
      if [[ "$dry_run" == "1" ]]; then
        launch_on_node "$node" "$inner_cmd"
      else
        launch_on_node "$node" "$inner_cmd" >"$shard_log" 2>&1 &
        pids+=("$!")
      fi
    fi
    shard_index=$((shard_index + 1))
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "[collect] dry run complete; no generation launched."
  exit 0
fi

progress_monitor &
progress_pid=$!

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
cleanup
progress_pid=""
progress_bar "$(progress_count)" "$expected_raw_lines"
printf '\n'

if [[ "$failed" -ne 0 ]]; then
  echo "One or more generation shards failed. Logs:" >&2
  ls -1 "$log_dir"/shard_*.log >&2 || true
  exit 1
fi

: > "$raw_jsonl"
for ((idx = 0; idx < num_shards; idx++)); do
  shard_output="$shard_dir/raw_actor_responses_shard_${idx}.jsonl"
  if [[ -f "$shard_output" ]]; then
    cat "$shard_output" >> "$raw_jsonl"
  fi
done

raw_lines=$(wc -l < "$raw_jsonl" | tr -d ' ')
echo "[collect] merged raw responses: $raw_lines rows -> $raw_jsonl"
if [[ "$raw_lines" -ne "$expected_raw_lines" ]]; then
  echo "Expected $expected_raw_lines raw responses ($max_examples prompts x $num_responses_per_prompt), got $raw_lines" >&2
  exit 1
fi

"$python_bin" - \
  --model_path "$model_path" \
  --skip_merge "$skip_merge" \
  --trust_remote_code "$trust_remote_code" \
  --raw_jsonl "$raw_jsonl" \
  --all_trajectories_jsonl "$all_trajectories_jsonl" \
  --all_trajectories_parquet "$all_trajectories_parquet" \
  --correct_jsonl "$correct_jsonl" \
  --calib_parquet "$calib_parquet" \
  --metrics_json "$metrics_json" \
  --num_responses_per_prompt "$num_responses_per_prompt" \
  --expected_total "$raw_lines" <<'PY'
import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
parser.add_argument("--expected_total", type=int, default=0)
args = parser.parse_args()


def print_progress(current: int, total: int):
    if total > 0:
        width = 40
        filled = min(width, int(width * current / total))
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * current / total
        sys.stderr.write(f"\r[postprocess] tokenizing [{bar}] {current}/{total} ({percent:5.1f}%)")
    else:
        sys.stderr.write(f"\r[postprocess] tokenizing {current} rows")
    sys.stderr.flush()


raw_path = Path(args.raw_jsonl).expanduser()
all_jsonl_path = Path(args.all_trajectories_jsonl).expanduser()
all_parquet_path = Path(args.all_trajectories_parquet).expanduser()
correct_path = Path(args.correct_jsonl).expanduser()
parquet_path = Path(args.calib_parquet).expanduser()
metrics_path = Path(args.metrics_json).expanduser()
for path in (all_jsonl_path, all_parquet_path, correct_path, parquet_path, metrics_path):
    path.parent.mkdir(parents=True, exist_ok=True)

actor_dir = resolve_actor_hf_dir(args.model_path, skip_merge=args.skip_merge.lower() in {"1", "true", "yes"})
tokenizer = AutoTokenizer.from_pretrained(
    actor_dir,
    trust_remote_code=args.trust_remote_code.lower() in {"1", "true", "yes"},
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

schema = pa.schema(
    [
        ("example_id", pa.int64()),
        ("response_index", pa.int64()),
        ("num_responses_per_prompt", pa.int64()),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("task_score", pa.float64()),
        ("is_correct", pa.bool_()),
        ("prompt_generated_trajectory", pa.string()),
        ("prompt_generated_trajectory_ids", pa.list_(pa.int64())),
    ]
)

batch_size = int(os.environ.get("POSTPROCESS_BATCH_SIZE", "512"))
progress_interval = max(1, args.expected_total // 200) if args.expected_total else 100
all_jsonl_tmp = Path(f"{all_jsonl_path}.tmp")
correct_tmp = Path(f"{correct_path}.tmp")
all_parquet_tmp = all_parquet_path.with_name(f".{all_parquet_path.name}.tmp.parquet")
calib_parquet_tmp = parquet_path.with_name(f".{parquet_path.name}.tmp.parquet")
metrics_tmp = Path(f"{metrics_path}.tmp")

all_writer = pq.ParquetWriter(all_parquet_tmp, schema)
correct_writer = pq.ParquetWriter(calib_parquet_tmp, schema)
all_batch = []
correct_batch = []
seen_prompts = set()
prompt_correct = set()
num_total = 0
num_scored = 0
num_correct = 0


def normalize_row(out_row):
    normalized = dict(out_row)
    if normalized["example_id"] is not None:
        normalized["example_id"] = int(normalized["example_id"])
    normalized["response_index"] = int(normalized["response_index"])
    normalized["num_responses_per_prompt"] = int(normalized["num_responses_per_prompt"])
    if normalized["task_score"] is not None:
        normalized["task_score"] = float(normalized["task_score"])
    normalized["prompt_generated_trajectory_ids"] = [int(token_id) for token_id in normalized["prompt_generated_trajectory_ids"]]
    return normalized


def write_batch(writer, rows):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


try:
    with raw_path.open("r", encoding="utf-8") as input_file, all_jsonl_tmp.open("w", encoding="utf-8") as all_file, correct_tmp.open("w", encoding="utf-8") as correct_file:
        print_progress(0, args.expected_total)
        for line in input_file:
            if not line.strip():
                continue
            num_total += 1
            row = json.loads(line)
            if "is_correct" in row:
                num_scored += 1

            example_id = row.get("example_id")
            seen_prompts.add(example_id)
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            trajectory = f"{prompt}{response}"
            trajectory_ids = tokenizer(
                trajectory,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            out_row = normalize_row(
                {
                    "example_id": example_id,
                    "response_index": row.get("response_index", 0),
                    "num_responses_per_prompt": row.get("num_responses_per_prompt", args.num_responses_per_prompt),
                    "prompt": prompt,
                    "response": response,
                    "task_score": row.get("task_score"),
                    "is_correct": bool(row.get("is_correct", False)),
                    "prompt_generated_trajectory": trajectory,
                    "prompt_generated_trajectory_ids": trajectory_ids,
                }
            )
            all_batch.append(out_row)
            all_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            if out_row["is_correct"]:
                prompt_correct.add(example_id)
                correct_batch.append(out_row)
                correct_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                num_correct += 1

            if len(all_batch) >= batch_size:
                write_batch(all_writer, all_batch)
            if len(correct_batch) >= batch_size:
                write_batch(correct_writer, correct_batch)
            if num_total % progress_interval == 0:
                print_progress(num_total, args.expected_total)

    write_batch(all_writer, all_batch)
    write_batch(correct_writer, correct_batch)
finally:
    all_writer.close()
    correct_writer.close()

print_progress(num_total, args.expected_total)
sys.stderr.write("\n")
sys.stderr.flush()

if not num_correct:
    raise SystemExit("No correct trajectories were collected; try sampling or inspect raw responses.")

prompt_count = len(seen_prompts)
metrics = {
    "num_total": num_total,
    "num_prompts": prompt_count,
    "num_responses_per_prompt": args.num_responses_per_prompt,
    "num_prompts_with_correct_response": len(prompt_correct),
    "prompt_pass_rate": (len(prompt_correct) / prompt_count) if prompt_count else None,
    "num_scored": num_scored,
    "num_correct": num_correct,
    "accuracy": (num_correct / num_scored) if num_scored else None,
    "response_accuracy": (num_correct / num_scored) if num_scored else None,
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

echo "[done] all trajectories parquet: $all_trajectories_parquet"
echo "[done] correct trajectories parquet: $calib_parquet"
echo "[done] use with PUNE as calib_data=$calib_parquet"
