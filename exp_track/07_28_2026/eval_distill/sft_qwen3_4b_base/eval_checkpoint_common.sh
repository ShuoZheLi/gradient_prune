#!/bin/bash
set -euo pipefail

# Common evaluator for SFT Qwen3-4B checkpoints.
# Submit one of the eval_global_step_*.sh wrappers with sbatch.

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

repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
if [[ -z "$repo_root" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work2/09576/shuozhe/gradient_prune"; do
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

scratch_root="${SCRATCH:-/scratch/09576/shuozhe}"
cache_root="${CACHE_ROOT:-${scratch_root}/${USER:-shuozhe}/gradient_prune_cache}"
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
# Paths and evaluation config
# -----------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must be set by the checkpoint wrapper." >&2
  exit 2
fi
checkpoint_name="${CHECKPOINT_NAME:-$(basename -- "$MODEL_PATH")}" 
run_name="${RUN_NAME:-sft_qwen3_4b_base_${checkpoint_name}_math500_eval}"
run_id="${RUN_ID:-${run_name}_${SLURM_JOB_ID:-manual}}"

dataset_path="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
results_base="${RESULTS_BASE:-${RESULTS_ROOT:-${scratch_root}/gradient_prune/results}}"
results_subdir="${RESULTS_SUBDIR:-eval_distill/sft_qwen3_4b_base/${checkpoint_name}}"
run_root="${RUN_OUTPUT_DIR:-${results_base}/${results_subdir}/runs/${run_id}}"
output_dir="${OUTPUT_DIR:-${run_root}}"
shard_dir="${SHARD_DIR:-${output_dir}/shards}"
log_dir="${LOG_DIR:-${output_dir}/logs}"
raw_jsonl="${RAW_JSONL:-${output_dir}/responses.jsonl}"
metrics_json="${METRICS_JSON:-${output_dir}/metrics.json}"
config_file="${CONFIG_FILE:-${log_dir}/config.env}"

max_examples="${MAX_EXAMPLES:-500}"
start_index="${START_INDEX:-0}"
seed="${SEED:-42}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${MAX_NEW_TOKENS:-16384}"
batch_size="${BATCH_SIZE:-64}"
generation_max_batch_tokens="${GENERATION_MAX_BATCH_TOKENS:-0}"
response_log_max="${RESPONSE_LOG_MAX:--1}"
num_responses_per_prompt="${NUM_RESPONSES_PER_PROMPT:-1}"
multi_response_temperature="${MULTI_RESPONSE_TEMPERATURE:-0.7}"
temperature="${TEMPERATURE:-0.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:-0}"
dtype="${DTYPE:-auto}"
case "$dtype" in
  bf16) vllm_dtype="bfloat16" ;;
  fp16) vllm_dtype="float16" ;;
  fp32) vllm_dtype="float32" ;;
  *) vllm_dtype="$dtype" ;;
esac
local_devices="${LOCAL_DEVICES:-${DEVICES:-0}}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
enforce_eager="${ENFORCE_EAGER:-1}"
prompt_key="${PROMPT_KEY:-prompt}"
response_key="${RESPONSE_KEY:-}"
reward_score_dir="${REWARD_SCORE_DIR:-}"
enable_thinking="${ENABLE_THINKING:-auto}"
progress_interval="${PROGRESS_INTERVAL:-10}"
dry_run="${DRY_RUN:-0}"

if [[ "$num_responses_per_prompt" -lt 1 ]]; then
  echo "NUM_RESPONSES_PER_PROMPT must be >= 1; got $num_responses_per_prompt" >&2
  exit 2
fi
if [[ "$max_examples" -lt 0 ]]; then
  echo "MAX_EXAMPLES must be >= 0 for sharded eval; got $max_examples" >&2
  exit 2
fi
if [[ "$batch_size" -lt 1 ]]; then
  echo "BATCH_SIZE must be >= 1; got $batch_size" >&2
  exit 2
fi
if [[ "$dry_run" != "1" && ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ "$dry_run" != "1" && ! -f "$dataset_path" ]]; then
  echo "Dataset path does not exist: $dataset_path" >&2
  exit 2
fi

read -r -a local_devices_array <<< "$local_devices"
if [[ ${#local_devices_array[@]} -lt 1 ]]; then
  echo "LOCAL_DEVICES/DEVICES must contain at least one GPU id per node." >&2
  exit 2
fi

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
expected_raw_lines=$((max_examples * num_responses_per_prompt))

if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "$gpus_per_node" -gt 1 ]]; then
  ntasks_per_node_value="${SLURM_NTASKS_PER_NODE%%(*}"
  if [[ "$ntasks_per_node_value" =~ ^[0-9]+$ && "$ntasks_per_node_value" -lt "$gpus_per_node" ]]; then
    echo "LOCAL_DEVICES requests $gpus_per_node shards per node, but Slurm allocated only $SLURM_NTASKS_PER_NODE task(s) per node." >&2
    echo "Set #SBATCH --ntasks-per-node=$gpus_per_node or reduce LOCAL_DEVICES." >&2
    exit 2
  fi
fi

mkdir -p "$output_dir" "$shard_dir" "$log_dir"
exec > >(tee -a "$log_dir/run.log") 2> >(tee -a "$log_dir/run.err" >&2)

cat > "$config_file" <<EOF_CONFIG
RUN_NAME=$run_name
RUN_ID=$run_id
MODEL_PATH=$MODEL_PATH
CHECKPOINT_NAME=$checkpoint_name
DATASET_PATH=$dataset_path
OUTPUT_DIR=$output_dir
RAW_JSONL=$raw_jsonl
METRICS_JSON=$metrics_json
NODES=${nodes_array[*]}
LOCAL_DEVICES=$local_devices
NUM_SHARDS=$num_shards
MAX_EXAMPLES=$max_examples
START_INDEX=$start_index
SEED=$seed
MAX_PROMPT_LENGTH=$max_prompt_length
MAX_NEW_TOKENS=$max_new_tokens
BATCH_SIZE=$batch_size
GENERATION_MAX_BATCH_TOKENS=$generation_max_batch_tokens
NUM_RESPONSES_PER_PROMPT=$num_responses_per_prompt
TEMPERATURE=$temperature
TOP_P=$top_p
TOP_K=$top_k
DTYPE=$dtype
VLLM_DTYPE=$vllm_dtype
TENSOR_PARALLEL_SIZE=$tensor_parallel_size
GPU_MEMORY_UTILIZATION=$gpu_memory_utilization
ENFORCE_EAGER=$enforce_eager
ENABLE_THINKING=$enable_thinking
EOF_CONFIG

echo "[eval] run_name=$run_name"
echo "[eval] model_path=$MODEL_PATH"
echo "[eval] dataset_path=$dataset_path"
echo "[eval] output_dir=$output_dir"
echo "[eval] nodes=${nodes_array[*]} local_devices=$local_devices num_shards=$num_shards"
echo "[eval] max_examples=$max_examples max_new_tokens=$max_new_tokens batch_size=$batch_size"

run_eval_shard() {
  local shard_id="$1"
  local shard_start="$2"
  local shard_count="$3"
  local shard_device="$4"
  local shard_output="$5"
  local shard_metrics="$6"

  local cmd=(
    create_calibration_dataset/vllm_accuracy_runner.py
    --model_path "$MODEL_PATH"
    --dataset_path "$dataset_path"
    --output_path "$shard_output"
    --metrics_path "$shard_metrics"
    --prompt_key "$prompt_key"
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
    --dtype "$vllm_dtype"
    --enable-thinking "$enable_thinking"
  )
  if [[ "$enforce_eager" == "1" ]]; then
    cmd+=(--enforce_eager)
  else
    cmd+=(--no_enforce_eager)
  fi
  if [[ -n "$response_key" ]]; then
    cmd+=(--response_key "$response_key")
  fi
  if [[ -n "$reward_score_dir" ]]; then
    cmd+=(--reward_score_dir "$reward_score_dir")
  fi

  echo "[eval][shard $shard_id] gpu=$shard_device start=$shard_start count=$shard_count output=$shard_output"
  if [[ "$dry_run" == "1" ]]; then
    printf '[eval][shard %s] command:' "$shard_id"
    printf ' %q' "CUDA_VISIBLE_DEVICES=$shard_device" "$python_bin" "${cmd[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$shard_device" "$python_bin" "${cmd[@]}"
}

launch_shard() {
  local node="$1"
  local shard_id="$2"
  local shard_start="$3"
  local shard_count="$4"
  local shard_device="$5"
  local shard_output="$6"
  local shard_metrics="$7"
  local shard_log="$8"

  if [[ "$dry_run" == "1" ]]; then
    run_eval_shard "$shard_id" "$shard_start" "$shard_count" "$shard_device" "$shard_output" "$shard_metrics"
    return 0
  fi

  if [[ -n "${SLURM_JOB_ID:-}" && "$num_nodes" -gt 1 ]]; then
    local remote_cmd
    remote_cmd="cd $(printf '%q' "$repo_root") && source $(printf '%q' "${VIRTUAL_ENV:-}/bin/activate") 2>/dev/null || true; export PYTHONPATH=$(printf '%q' "$PYTHONPATH") UV_CACHE_DIR=$(printf '%q' "$UV_CACHE_DIR") HF_HOME=$(printf '%q' "$HF_HOME") TRANSFORMERS_CACHE=$(printf '%q' "$TRANSFORMERS_CACHE") HF_DATASETS_CACHE=$(printf '%q' "$HF_DATASETS_CACHE") TORCH_HOME=$(printf '%q' "$TORCH_HOME") TRITON_CACHE_DIR=$(printf '%q' "$TRITON_CACHE_DIR") XDG_CACHE_HOME=$(printf '%q' "$XDG_CACHE_HOME") TIKTOKEN_ENCODINGS_BASE=$(printf '%q' "$TIKTOKEN_ENCODINGS_BASE") PYTHONUNBUFFERED=1 TASK_SCORER_BACKEND=$(printf '%q' "$TASK_SCORER_BACKEND") TOKENIZERS_PARALLELISM=$(printf '%q' "$TOKENIZERS_PARALLELISM") VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=$(printf '%q' "$VLLM_WORKER_MULTIPROC_METHOD") VLLM_USE_V1=$(printf '%q' "$VLLM_USE_V1") MODEL_PATH=$(printf '%q' "$MODEL_PATH") dataset_path=$(printf '%q' "$dataset_path") prompt_key=$(printf '%q' "$prompt_key") response_key=$(printf '%q' "$response_key") reward_score_dir=$(printf '%q' "$reward_score_dir") seed=$(printf '%q' "$seed") max_prompt_length=$(printf '%q' "$max_prompt_length") max_new_tokens=$(printf '%q' "$max_new_tokens") batch_size=$(printf '%q' "$batch_size") generation_max_batch_tokens=$(printf '%q' "$generation_max_batch_tokens") response_log_max=$(printf '%q' "$response_log_max") num_responses_per_prompt=$(printf '%q' "$num_responses_per_prompt") multi_response_temperature=$(printf '%q' "$multi_response_temperature") temperature=$(printf '%q' "$temperature") top_p=$(printf '%q' "$top_p") top_k=$(printf '%q' "$top_k") tensor_parallel_size=$(printf '%q' "$tensor_parallel_size") gpu_memory_utilization=$(printf '%q' "$gpu_memory_utilization") vllm_dtype=$(printf '%q' "$vllm_dtype") enable_thinking=$(printf '%q' "$enable_thinking") enforce_eager=$(printf '%q' "$enforce_eager") dry_run=0 python_bin=$(printf '%q' "$python_bin"); $(declare -f run_eval_shard); run_eval_shard $(printf '%q' "$shard_id") $(printf '%q' "$shard_start") $(printf '%q' "$shard_count") $(printf '%q' "$shard_device") $(printf '%q' "$shard_output") $(printf '%q' "$shard_metrics")"
    srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --nodelist="$node" bash -lc "$remote_cmd" >"$shard_log" 2>&1 &
  else
    run_eval_shard "$shard_id" "$shard_start" "$shard_count" "$shard_device" "$shard_output" "$shard_metrics" >"$shard_log" 2>&1 &
  fi
}

rm -f "$raw_jsonl" "$metrics_json" "$shard_dir"/responses_shard_*.jsonl "$shard_dir"/metrics_shard_*.json "$shard_dir"/shard_*.log
base_count=$((max_examples / num_shards))
remainder=$((max_examples % num_shards))
shard_index=0
pids=()

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
      shard_output="$shard_dir/responses_shard_${shard_index}.jsonl"
      shard_metrics="$shard_dir/metrics_shard_${shard_index}.json"
      shard_log="$shard_dir/shard_${shard_index}.log"
      launch_shard "$node" "$shard_index" "$shard_start" "$shard_count" "$gpu_id" "$shard_output" "$shard_metrics" "$shard_log"
      if [[ "$dry_run" != "1" ]]; then
        pids+=("$!")
      fi
    fi
    shard_index=$((shard_index + 1))
  done
done

if [[ "$dry_run" == "1" ]]; then
  echo "[eval] dry run complete; no evaluation launched."
  exit 0
fi

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
  while [[ "${#bar}" -lt "$filled" ]]; do bar="${bar}#"; done
  while [[ "$empty" -gt 0 ]]; do bar="${bar}-"; empty=$((empty - 1)); done
  printf '\r[eval] progress [%s] %s/%s (%s%%)' "$bar" "$current" "$total" "$percent"
}

progress_count() {
  local completed=0
  local shard_output shard_lines
  for shard_output in "$shard_dir"/responses_shard_*.jsonl; do
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

progress_monitor &
progress_pid=$!

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
kill "$progress_pid" 2>/dev/null || true
wait "$progress_pid" 2>/dev/null || true
progress_bar "$(progress_count)" "$expected_raw_lines"
printf '\n'

if [[ "$failed" -ne 0 ]]; then
  echo "One or more eval shards failed. Logs:" >&2
  ls -1 "$shard_dir"/shard_*.log >&2 || true
  exit 1
fi

: > "$raw_jsonl"
for ((i = 0; i < num_shards; i++)); do
  shard_output="$shard_dir/responses_shard_${i}.jsonl"
  if [[ -f "$shard_output" ]]; then
    cat "$shard_output" >> "$raw_jsonl"
  fi
done

raw_count=$(wc -l < "$raw_jsonl" | tr -d ' ')
if [[ "$raw_count" -ne "$expected_raw_lines" ]]; then
  echo "Merged response count mismatch: expected $expected_raw_lines, got $raw_count" >&2
  echo "Shard logs are in: $shard_dir" >&2
  exit 1
fi

"$python_bin" - "$metrics_json" "$shard_dir" "$max_examples" "$num_responses_per_prompt" <<'PY_AGGREGATE_METRICS'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
max_examples = int(sys.argv[3])
num_responses = int(sys.argv[4])

scores = []
correct = []
num_examples = 0
num_generations = 0
num_scored = 0
num_unscored = 0
for shard_metrics_path in sorted(shard_dir.glob("metrics_shard_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
    shard_metrics = json.loads(shard_metrics_path.read_text(encoding="utf-8"))
    num_examples += int(shard_metrics.get("num_examples", 0))
    num_generations += int(shard_metrics.get("num_generations", 0))
    num_scored += int(shard_metrics.get("num_scored", 0))
    num_unscored += int(shard_metrics.get("num_unscored", 0))

responses_path = metrics_path.parent / "responses.jsonl"
prompt_has_correct = {}
with responses_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_score") is not None:
            score = float(row.get("task_score", 0.0))
            is_correct = bool(row.get("is_correct", score == 1.0))
            scores.append(score)
            correct.append(is_correct)
            prompt_id = row.get("example_id")
            prompt_has_correct[prompt_id] = bool(prompt_has_correct.get(prompt_id, False) or is_correct)

metrics = {
    "num_examples": num_examples,
    "expected_num_examples": max_examples,
    "num_responses_per_prompt": num_responses,
    "num_generations": num_generations,
    "num_scored": num_scored,
    "num_unscored": num_unscored,
}
if scores:
    num_correct = sum(1 for item in correct if item)
    metrics.update({
        "pass@1": num_correct / len(correct),
        "accuracy": num_correct / len(correct),
        "response_accuracy": num_correct / len(correct),
        "prompt_pass_rate": sum(1 for item in prompt_has_correct.values() if item) / num_examples if num_examples else None,
        "num_prompts_with_correct_response": sum(1 for item in prompt_has_correct.values() if item),
        "mean_score": sum(scores) / len(scores),
        "score_sum": sum(scores),
        "num_correct": num_correct,
    })
metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY_AGGREGATE_METRICS

echo "[eval] merged $raw_count responses from $num_shards shard(s): $raw_jsonl"
echo "[eval] metrics: $metrics_json"
cat "$metrics_json"
