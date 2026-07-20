#!/bin/bash
#SBATCH --job-name=qwen2d5_7b_resp_analysis_s0d5
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j_qwen2d5_7b_resp_analysis_s0d5.out
#SBATCH --error=slurm-%j_qwen2d5_7b_resp_analysis_s0d5.err

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="https://api.portkey.ai/v1"
export OPENAI_EVALUATOR_MODEL="@irom-ll37364-op-b37b3e/gpt-5.5"

set -euo pipefail

# Self-contained response-analysis launcher for token entropy and diversity.
# Uses WANDA scores generated from the Qwen2.5-7B-Instruct math7500 correct calibration dataset.
# Configure by editing the Runtime config block or overriding env vars at sbatch time.

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
    if [[ -f "$dir/pyproject.toml" && -d "$dir/response_analysis" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
if [[ -z "$repo_root" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work2/09576/shuozhe/gradient_prune" "/data/shuozhe/gradient_prune"; do
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
export PYTHONPATH="$repo_root:$repo_root/src:${PYTHONPATH:-}"

SCRATCH_ROOT="${SCRATCH:-/tmp/${USER:-shuozhe}}"
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
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TASK_SCORER_BACKEND="${TASK_SCORER_BACKEND:-verl_default}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

python_bin="${PYTHON_BIN:-python3}"

# -----------------------------
# Runtime config
# -----------------------------
RUN_NAME="${RUN_NAME:-qwen2d5_7b_instruct_wanda_cali_data_math7500_response_analysis}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}}"
RESULTS_BASE="${RESULTS_BASE:-${RESULTS_ROOT:-$SCRATCH_ROOT/gradient_prune/results}}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-response_analysis/${RUN_NAME}}"
RUN_ROOT="${RUN_OUTPUT_DIR:-$RESULTS_BASE/$RESULTS_SUBDIR/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
CONFIG_FILE="${CONFIG_FILE:-$LOG_DIR/config.env}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2> >(tee -a "$LOG_DIR/run.err" >&2)

MODEL_PATH="${MODEL_PATH:-/work/09576/shuozhe/saved_model/Qwen2.5-7B-Instruct}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.5}"
BASE_MODEL_ID="${BASE_MODEL_ID:-qwen2d5_7b_instruct_dense}"
PRUNED_MODEL_ID="${PRUNED_MODEL_ID:-qwen2d5_7b_instruct_wanda_s${PRUNING_SPARSITY}}"
DATASET_PATH="${DATASET_PATH:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
CALIBRATION_DATA_LABEL="${CALIBRATION_DATA_LABEL:-Qwen2.5-7B-Instruct math7500 correct calibration dataset}"
SCORE_GENERATION_RUN_ID="${SCORE_GENERATION_RUN_ID:-qwen2d5_7b_instruct_prune_wanda_math7500}"
SCORE_ROOT="${SCORE_ROOT:-/scratch/09576/shuozhe/gradient_prune/results/qwen2d5_7b_instruct_wanda_math7500/scores}"

if [[ -z "${RUN_DENSE+x}" ]]; then
  if [[ "$PRUNING_SPARSITY" == "0" || "$PRUNING_SPARSITY" == "0.0" ]]; then
    RUN_DENSE=1
  else
    RUN_DENSE=0
  fi
fi
if [[ -z "${RUN_PRUNED+x}" ]]; then
  if [[ "$PRUNING_SPARSITY" == "0" || "$PRUNING_SPARSITY" == "0.0" ]]; then
    RUN_PRUNED=0
  else
    RUN_PRUNED=1
  fi
fi
NO_API="${NO_API:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_ON_POLICY_ENTROPY="${RUN_ON_POLICY_ENTROPY:-1}"
RUN_FIXED_PREFIX_ENTROPY="${RUN_FIXED_PREFIX_ENTROPY:-1}"
RUN_SURFACE_DIVERSITY="${RUN_SURFACE_DIVERSITY:-1}"
RUN_SEMANTIC_JUDGE="${RUN_SEMANTIC_JUDGE:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
if [[ "$NO_API" == "1" ]]; then
  RUN_SEMANTIC_JUDGE=0
fi
PARALLEL_GENERATION="${PARALLEL_GENERATION:-auto}"
PARALLEL_ENTROPY="${PARALLEL_ENTROPY:-auto}"

MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
DEBUG_SUBSET="${DEBUG_SUBSET:-}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-42}"
K="${K:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-}"
ENABLE_THINKING="${ENABLE_THINKING:-auto}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
USE_CACHE="${USE_CACHE:-0}"
LOCAL_DEVICES="${LOCAL_DEVICES:-0}"
GENERATION_BACKEND="${GENERATION_BACKEND:-vllm}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
DELETE_VLLM_PRUNED_MODEL="${DELETE_VLLM_PRUNED_MODEL:-1}"

PRUNE_SCORE_KEY="${PRUNE_SCORE_KEY:-}"
PRUNE_GRANULARITY="${PRUNE_GRANULARITY:-rowwise}"
PRUNE_LAMBDA="${PRUNE_LAMBDA:-}"
PRUNE_OPS="${PRUNE_OPS:-}"

FIXED_PREFIX_SOURCE="${FIXED_PREFIX_SOURCE:-dataset_reference}"
FIXED_PREFIX_SHARED_FILE="${FIXED_PREFIX_SHARED_FILE:-}"
FIXED_PREFIX_MAX_RECORDS="${FIXED_PREFIX_MAX_RECORDS:--1}"

OPENAI_EVALUATOR_MODEL="${OPENAI_EVALUATOR_MODEL:-@irom-ll37364-op-b37b3e/gpt-5.5}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.portkey.ai/v1}"
JUDGE_SHUFFLE_REPEATS="${JUDGE_SHUFFLE_REPEATS:-2}"
JUDGE_MAX_PROMPTS="${JUDGE_MAX_PROMPTS:--1}"
JUDGE_JSON_MODE="${JUDGE_JSON_MODE:-schema}"
if [[ "$NO_API" == "1" ]]; then
  JUDGE_DISABLE_API="${JUDGE_DISABLE_API:-1}"
else
  JUDGE_DISABLE_API="${JUDGE_DISABLE_API:-0}"
fi
JUDGE_CACHE_DIR="${JUDGE_CACHE_DIR:-$RUN_ROOT/api_cache}"

BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BASELINE_MODEL="${BASELINE_MODEL:-$BASE_MODEL_ID}"

expand_slurm_nodes() {
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_JOB_NODELIST"
  else
    hostname
  fi
}

nodes_array=()
while IFS= read -r node; do
  [[ -n "$node" ]] && nodes_array+=("$node")
done < <(expand_slurm_nodes)
if [[ "${#nodes_array[@]}" -eq 0 ]]; then
  nodes_array=("$(hostname)")
fi
num_nodes="${#nodes_array[@]}"
if [[ "$PARALLEL_GENERATION" == "auto" ]]; then
  if [[ "$num_nodes" -gt 1 && -n "${SLURM_JOB_ID:-}" && "${MAX_EXAMPLES}" -gt 1 ]]; then
    PARALLEL_GENERATION=1
  else
    PARALLEL_GENERATION=0
  fi
fi
if [[ "$PARALLEL_ENTROPY" == "auto" ]]; then
  if [[ "$num_nodes" -gt 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    PARALLEL_ENTROPY=1
  else
    PARALLEL_ENTROPY=0
  fi
fi

if [[ "$DRY_RUN" != "1" && ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ "$RUN_PRUNED" == "1" && "$DRY_RUN" != "1" && ! -f "$SCORE_ROOT/metadata.json" ]]; then
  echo "Score root does not contain metadata.json: $SCORE_ROOT" >&2
  echo "Set SCORE_ROOT=/path/to/saved/scores or RUN_PRUNED=0." >&2
  exit 3
fi
if [[ "$RUN_SEMANTIC_JUDGE" == "1" && "$JUDGE_DISABLE_API" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "RUN_SEMANTIC_JUDGE=1 requires OPENAI_API_KEY unless JUDGE_DISABLE_API=1." >&2
  exit 4
fi

mkdir -p "$RUN_ROOT"
cat > "$CONFIG_FILE" <<EOF
RUN_NAME=$RUN_NAME
RUN_ID=$RUN_ID
RUN_ROOT=$RUN_ROOT
MODEL_PATH=$MODEL_PATH
BASE_MODEL_ID=$BASE_MODEL_ID
PRUNED_MODEL_ID=$PRUNED_MODEL_ID
DATASET_PATH=$DATASET_PATH
CALIBRATION_DATA_LABEL=$CALIBRATION_DATA_LABEL
SCORE_GENERATION_RUN_ID=$SCORE_GENERATION_RUN_ID
SCORE_ROOT=$SCORE_ROOT
RUN_DENSE=$RUN_DENSE
RUN_PRUNED=$RUN_PRUNED
NO_API=$NO_API
RUN_GENERATION=$RUN_GENERATION
RUN_ON_POLICY_ENTROPY=$RUN_ON_POLICY_ENTROPY
RUN_FIXED_PREFIX_ENTROPY=$RUN_FIXED_PREFIX_ENTROPY
RUN_SURFACE_DIVERSITY=$RUN_SURFACE_DIVERSITY
RUN_SEMANTIC_JUDGE=$RUN_SEMANTIC_JUDGE
JUDGE_DISABLE_API=$JUDGE_DISABLE_API
RUN_AGGREGATE=$RUN_AGGREGATE
PARALLEL_GENERATION=$PARALLEL_GENERATION
PARALLEL_ENTROPY=$PARALLEL_ENTROPY
NODES=${nodes_array[*]}
MAX_EXAMPLES=$MAX_EXAMPLES
DEBUG_SUBSET=$DEBUG_SUBSET
K=$K
TEMPERATURE=$TEMPERATURE
TOP_P=$TOP_P
TOP_K=$TOP_K
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
ENABLE_THINKING=$ENABLE_THINKING
DTYPE=$DTYPE
DEVICE=$DEVICE
LOCAL_DEVICES=$LOCAL_DEVICES
GENERATION_BACKEND=$GENERATION_BACKEND
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
DELETE_VLLM_PRUNED_MODEL=$DELETE_VLLM_PRUNED_MODEL
PRUNING_SPARSITY=$PRUNING_SPARSITY
PRUNE_SCORE_KEY=$PRUNE_SCORE_KEY
PRUNE_GRANULARITY=$PRUNE_GRANULARITY
PRUNE_LAMBDA=$PRUNE_LAMBDA
PRUNE_OPS=$PRUNE_OPS
FIXED_PREFIX_SOURCE=$FIXED_PREFIX_SOURCE
OPENAI_EVALUATOR_MODEL=$OPENAI_EVALUATOR_MODEL
OPENAI_BASE_URL=$OPENAI_BASE_URL
EOF

echo "[$(date)] Score calibration data: $CALIBRATION_DATA_LABEL"
echo "[$(date)] Score generation run: $SCORE_GENERATION_RUN_ID"
echo "[$(date)] Score root: $SCORE_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; config written to $CONFIG_FILE"
  exit 0
fi

common_generation_args=(
  --model_path "$MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --prompt_key "$PROMPT_KEY"
  --seed "$SEED"
  --k "$K"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --enable_thinking "$ENABLE_THINKING"
  --generation_backend "$GENERATION_BACKEND"
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then common_generation_args+=(--vllm_enforce_eager); fi
if [[ -n "$VLLM_MAX_NUM_SEQS" ]]; then common_generation_args+=(--vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"); fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then common_generation_args+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN"); fi
if [[ -n "$RESPONSE_KEY" ]]; then common_generation_args+=(--response_key "$RESPONSE_KEY"); fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then common_generation_args+=(--trust_remote_code); fi
if [[ "$USE_CACHE" == "1" ]]; then common_generation_args+=(--use_cache); fi

common_entropy_args=(
  --model_path "$MODEL_PATH"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --device "$DEVICE"
  --dtype "$DTYPE"
)
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then common_entropy_args+=(--trust_remote_code); fi

prune_args=(--prune_score_dir "$SCORE_ROOT" --pruning_sparsity "$PRUNING_SPARSITY" --prune_granularity "$PRUNE_GRANULARITY")
if [[ -n "$PRUNE_SCORE_KEY" ]]; then prune_args+=(--prune_score_key "$PRUNE_SCORE_KEY"); fi
if [[ -n "$PRUNE_LAMBDA" ]]; then prune_args+=(--prune_lambda "$PRUNE_LAMBDA"); fi
if [[ -n "$PRUNE_OPS" ]]; then
  # shellcheck disable=SC2206
  prune_ops_array=($PRUNE_OPS)
  prune_args+=(--prune_ops "${prune_ops_array[@]}")
fi

launch_on_node() {
  local node="$1"
  local cmd="$2"
  if [[ -n "${SLURM_JOB_ID:-}" && "$node" != "$(hostname)" ]]; then
    srun --nodes=1 --ntasks=1 --exclusive -w "$node" bash -lc "$cmd"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --nodes=1 --ntasks=1 --exclusive -w "$node" bash -lc "$cmd"
  else
    bash -lc "$cmd"
  fi
}

run_generation_shards() {
  local model_id="$1"
  local output_dir="$2"
  local is_pruned="$3"
  local generations="$4"
  shift 4
  local base_args=("$@")
  local shard_dir="$output_dir/generation_shards"
  mkdir -p "$shard_dir" "$LOG_DIR/shards"
  rm -f "$shard_dir"/generations_shard_*.jsonl "$LOG_DIR/shards"/${model_id}_shard_*.log "$generations"

  local total_examples="$MAX_EXAMPLES"
  if [[ -n "$DEBUG_SUBSET" ]]; then total_examples="$DEBUG_SUBSET"; fi
  if [[ "$total_examples" -lt 0 ]]; then
    echo "PARALLEL_GENERATION requires MAX_EXAMPLES or DEBUG_SUBSET to be non-negative." >&2
    exit 5
  fi

  local num_shards="$num_nodes"
  if [[ "$total_examples" -lt "$num_shards" ]]; then num_shards="$total_examples"; fi
  if [[ "$num_shards" -le 0 ]]; then num_shards=1; fi
  local base_count=$((total_examples / num_shards))
  local remainder=$((total_examples % num_shards))
  local pids=()

  echo "[$(date)] Parallel generation for $model_id across $num_shards shard(s): nodes=${nodes_array[*]}"
  for ((shard_index = 0; shard_index < num_shards; shard_index++)); do
    local shard_count="$base_count"
    if [[ "$shard_index" -lt "$remainder" ]]; then shard_count=$((shard_count + 1)); fi
    local shard_start
    if [[ "$shard_index" -lt "$remainder" ]]; then
      shard_start=$((START_INDEX + shard_index * (base_count + 1)))
    else
      shard_start=$((START_INDEX + remainder * (base_count + 1) + (shard_index - remainder) * base_count))
    fi
    local node="${nodes_array[$((shard_index % num_nodes))]}"
    local shard_output="$shard_dir/generations_shard_${shard_index}.jsonl"
    local shard_log="$LOG_DIR/shards/${model_id}_shard_${shard_index}_${node}.log"

    local cmd
    cmd="cd $(printf '%q' "$repo_root") && export PYTHONPATH=$(printf '%q' "$PYTHONPATH") UV_CACHE_DIR=$(printf '%q' "$UV_CACHE_DIR") HF_HOME=$(printf '%q' "$HF_HOME") TRANSFORMERS_CACHE=$(printf '%q' "$TRANSFORMERS_CACHE") HF_DATASETS_CACHE=$(printf '%q' "$HF_DATASETS_CACHE") TORCH_HOME=$(printf '%q' "$TORCH_HOME") TRITON_CACHE_DIR=$(printf '%q' "$TRITON_CACHE_DIR") XDG_CACHE_HOME=$(printf '%q' "$XDG_CACHE_HOME") TIKTOKEN_ENCODINGS_BASE=$(printf '%q' "$TIKTOKEN_ENCODINGS_BASE") PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=$(printf '%q' "$TOKENIZERS_PARALLELISM") TASK_SCORER_BACKEND=$(printf '%q' "$TASK_SCORER_BACKEND") CUDA_VISIBLE_DEVICES=$(printf '%q' "$LOCAL_DEVICES"); $(printf '%q' "$python_bin") -m response_analysis.generate_responses"
    for arg in "${base_args[@]}"; do cmd+=" $(printf '%q' "$arg")"; done
    cmd+=" --start_index $(printf '%q' "$shard_start") --max_examples $(printf '%q' "$shard_count") --output $(printf '%q' "$shard_output")"
    launch_on_node "$node" "$cmd" >"$shard_log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "One or more generation shards failed. Logs are in $LOG_DIR/shards" >&2
    exit 6
  fi

  : > "$generations"
  for ((shard_index = 0; shard_index < num_shards; shard_index++)); do
    local shard_output="$shard_dir/generations_shard_${shard_index}.jsonl"
    if [[ -f "$shard_output" ]]; then cat "$shard_output" >> "$generations"; fi
  done
  echo "[$(date)] Merged $(wc -l < "$generations" | tr -d ' ') rows -> $generations"
}

split_jsonl_for_shards() {
  local input_jsonl="$1"
  local shard_dir="$2"
  local max_records="$3"
  "$python_bin" - <<'PY' "$input_jsonl" "$shard_dir" "$num_nodes" "$max_records"
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
num_nodes = max(1, int(sys.argv[3]))
max_records = int(sys.argv[4])
lines = input_path.read_text(encoding="utf-8").splitlines()
if max_records >= 0:
    lines = lines[:max_records]
num_shards = min(num_nodes, max(1, len(lines)))
shard_dir.mkdir(parents=True, exist_ok=True)
for old in shard_dir.glob("entropy_input_shard_*.jsonl"):
    old.unlink()
base = len(lines) // num_shards
remainder = len(lines) % num_shards
start = 0
for shard_idx in range(num_shards):
    count = base + (1 if shard_idx < remainder else 0)
    shard_lines = lines[start:start + count]
    start += count
    (shard_dir / f"entropy_input_shard_{shard_idx}.jsonl").write_text("\n".join(shard_lines) + ("\n" if shard_lines else ""), encoding="utf-8")
print(num_shards)
PY
}

run_entropy_shards() {
  local model_id="$1"
  local output_dir="$2"
  local mode="$3"
  local source_jsonl="$4"
  local output_parquet="$5"
  local max_records="$6"
  shift 6
  local base_args=("$@")
  local shard_dir="$output_dir/${mode}_entropy_shards"
  mkdir -p "$shard_dir" "$LOG_DIR/shards"
  rm -f "$shard_dir"/token_metrics_shard_*.parquet "$LOG_DIR/shards"/${model_id}_${mode}_entropy_shard_*.log "$output_parquet"

  local num_shards
  num_shards="$(split_jsonl_for_shards "$source_jsonl" "$shard_dir" "$max_records")"
  local pids=()
  echo "[$(date)] Parallel $mode entropy for $model_id across $num_shards shard(s): nodes=${nodes_array[*]}"
  for ((shard_index = 0; shard_index < num_shards; shard_index++)); do
    local node="${nodes_array[$((shard_index % num_nodes))]}"
    local shard_input="$shard_dir/entropy_input_shard_${shard_index}.jsonl"
    local shard_output="$shard_dir/token_metrics_shard_${shard_index}.parquet"
    local shard_log="$LOG_DIR/shards/${model_id}_${mode}_entropy_shard_${shard_index}_${node}.log"
    local cmd
    cmd="cd $(printf '%q' "$repo_root") && export PYTHONPATH=$(printf '%q' "$PYTHONPATH") UV_CACHE_DIR=$(printf '%q' "$UV_CACHE_DIR") HF_HOME=$(printf '%q' "$HF_HOME") TRANSFORMERS_CACHE=$(printf '%q' "$TRANSFORMERS_CACHE") HF_DATASETS_CACHE=$(printf '%q' "$HF_DATASETS_CACHE") TORCH_HOME=$(printf '%q' "$TORCH_HOME") TRITON_CACHE_DIR=$(printf '%q' "$TRITON_CACHE_DIR") XDG_CACHE_HOME=$(printf '%q' "$XDG_CACHE_HOME") TIKTOKEN_ENCODINGS_BASE=$(printf '%q' "$TIKTOKEN_ENCODINGS_BASE") PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=$(printf '%q' "$TOKENIZERS_PARALLELISM") CUDA_VISIBLE_DEVICES=$(printf '%q' "$LOCAL_DEVICES"); $(printf '%q' "$python_bin") -m response_analysis.compute_token_entropy"
    for arg in "${base_args[@]}"; do cmd+=" $(printf '%q' "$arg")"; done
    if [[ "$mode" == "fixed_prefix" ]]; then
      cmd+=" --mode fixed_prefix --prefix_bank $(printf '%q' "$shard_input") --output $(printf '%q' "$shard_output")"
    else
      cmd+=" --input $(printf '%q' "$shard_input") --output $(printf '%q' "$shard_output")"
    fi
    launch_on_node "$node" "$cmd" >"$shard_log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "One or more $mode entropy shards failed. Logs are in $LOG_DIR/shards" >&2
    exit 7
  fi

  "$python_bin" - <<'PY' "$output_parquet" "$shard_dir"
from pathlib import Path
import sys
import pandas as pd
output = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
frames = [pd.read_parquet(path) for path in sorted(shard_dir.glob("token_metrics_shard_*.parquet"))]
output.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
PY
  echo "[$(date)] Merged $mode entropy shards -> $output_parquet"
}

validate_generation_metadata() {
  local generations="$1"
  local model_id="$2"
  local is_pruned="$3"
  "$python_bin" - <<'VALIDATE_GENERATION_PY' "$generations" "$model_id" "$is_pruned" "$PRUNING_SPARSITY"
from pathlib import Path
import json
import sys

generations = Path(sys.argv[1])
model_id = sys.argv[2]
is_pruned = sys.argv[3] == "1"
expected_sparsity = float(sys.argv[4])
rows = 0
bad = []
for line in generations.open(encoding="utf-8"):
    if not line.strip():
        continue
    rows += 1
    record = json.loads(line)
    pruning_info = record.get("pruning_info") or {}
    actual_model_id = record.get("model_id")
    actual_sparsity = float(record.get("pruning_sparsity") or 0.0)
    enabled = bool(pruning_info.get("enabled", False))
    if actual_model_id != model_id:
        bad.append(f"row {rows}: model_id={actual_model_id!r}, expected {model_id!r}")
    if is_pruned:
        if expected_sparsity <= 0.0:
            bad.append(f"row {rows}: pruned pipeline requested with nonpositive expected_sparsity={expected_sparsity}")
        if not enabled:
            bad.append(f"row {rows}: pruning_info.enabled is false for pruned pipeline")
        if abs(actual_sparsity - expected_sparsity) > 1e-12:
            bad.append(f"row {rows}: pruning_sparsity={actual_sparsity}, expected {expected_sparsity}")
    else:
        if enabled:
            bad.append(f"row {rows}: pruning_info.enabled is true for dense pipeline")
if rows == 0:
    bad.append("generation file is empty")
if bad:
    preview = "\n".join(bad[:10])
    raise SystemExit(f"Generation metadata validation failed for {generations}:\n{preview}")
print(f"Validated {rows} generation rows for {model_id} (is_pruned={is_pruned}, expected_sparsity={expected_sparsity})")
VALIDATE_GENERATION_PY
}

run_model_pipeline() {
  local model_id="$1"
  local output_dir="$2"
  local is_pruned="$3"
  mkdir -p "$output_dir"

  local generations="$output_dir/generations.jsonl"
  local token_metrics="$output_dir/token_metrics.parquet"
  local fixed_prefix_bank="$output_dir/fixed_prefix_bank.jsonl"
  local fixed_token_metrics="$output_dir/fixed_token_metrics.parquet"
  local response_metrics="$output_dir/response_metrics.parquet"
  local semantic_judgments="$output_dir/semantic_judgments.jsonl"
  local strategy_metrics="$output_dir/strategy_metrics.parquet"

  local model_generation_args=("${common_generation_args[@]}" --model_id "$model_id")
  local model_entropy_args=("${common_entropy_args[@]}" --model_id "$model_id")
  if [[ "$is_pruned" == "1" ]]; then
    model_generation_args+=("${prune_args[@]}")
    if [[ "$GENERATION_BACKEND" == "vllm" ]]; then model_generation_args+=(--vllm_pruned_model_dir "$output_dir/vllm_pruned_model"); fi
    model_entropy_args+=("${prune_args[@]}")
  else
    model_generation_args+=(--pruning_sparsity 0.0)
    model_entropy_args+=(--pruning_sparsity 0.0)
  fi

  if [[ "$RUN_GENERATION" == "1" ]]; then
    echo "[$(date)] Generating responses for $model_id"
    if [[ "$PARALLEL_GENERATION" == "1" ]]; then
      run_generation_shards "$model_id" "$output_dir" "$is_pruned" "$generations" "${model_generation_args[@]}"
    else
      serial_generation_args=("${model_generation_args[@]}" --start_index "$START_INDEX" --max_examples "$MAX_EXAMPLES" --output "$generations")
      if [[ -n "$DEBUG_SUBSET" ]]; then serial_generation_args+=(--debug_subset "$DEBUG_SUBSET"); fi
      "$python_bin" -m response_analysis.generate_responses "${serial_generation_args[@]}"
    fi
    validate_generation_metadata "$generations" "$model_id" "$is_pruned"
  fi

  if [[ "$is_pruned" == "1" && "$GENERATION_BACKEND" == "vllm" && "$DELETE_VLLM_PRUNED_MODEL" == "1" && "$RUN_GENERATION" == "1" ]]; then
    echo "[$(date)] Deleting temporary pruned vLLM checkpoint for $model_id"
    rm -rf "$output_dir/vllm_pruned_model" "$output_dir/vllm_pruned_model.lock"
  fi

  if [[ "$RUN_ON_POLICY_ENTROPY" == "1" ]]; then
    echo "[$(date)] Computing on-policy entropy for $model_id"
    if [[ "$PARALLEL_ENTROPY" == "1" ]]; then
      run_entropy_shards "$model_id" "$output_dir" on_policy "$generations" "$token_metrics" -1 "${model_entropy_args[@]}"
    else
      "$python_bin" -m response_analysis.compute_token_entropy \
        "${model_entropy_args[@]}" \
        --input "$generations" \
        --output "$token_metrics"
    fi
  fi

  if [[ "$RUN_FIXED_PREFIX_ENTROPY" == "1" ]]; then
    echo "[$(date)] Building fixed prefix bank for $model_id"
    fixed_prefix_args=(
      --tokenizer_path "$MODEL_PATH"
      --output "$fixed_prefix_bank"
      --source "$FIXED_PREFIX_SOURCE"
      --max_examples "$MAX_EXAMPLES"
      --seed "$SEED"
      --prompt_key "$PROMPT_KEY"
      --enable_thinking "$ENABLE_THINKING"
    )
    if [[ "$FIXED_PREFIX_SOURCE" == "dataset_reference" ]]; then
      fixed_prefix_args+=(--dataset_path "$DATASET_PATH")
      if [[ -n "$RESPONSE_KEY" ]]; then fixed_prefix_args+=(--response_key "$RESPONSE_KEY"); fi
    elif [[ "$FIXED_PREFIX_SOURCE" == "generations" ]]; then
      fixed_prefix_args+=(--generations "$generations")
    elif [[ "$FIXED_PREFIX_SOURCE" == "shared_file" ]]; then
      fixed_prefix_args+=(--shared_file "$FIXED_PREFIX_SHARED_FILE")
    fi
    if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then fixed_prefix_args+=(--trust_remote_code); fi
    "$python_bin" -m response_analysis.build_fixed_prefix_bank "${fixed_prefix_args[@]}"

    echo "[$(date)] Computing fixed-prefix entropy for $model_id"
    if [[ "$PARALLEL_ENTROPY" == "1" ]]; then
      run_entropy_shards "$model_id" "$output_dir" fixed_prefix "$fixed_prefix_bank" "$fixed_token_metrics" "$FIXED_PREFIX_MAX_RECORDS" "${model_entropy_args[@]}"
    else
      fixed_entropy_args=("${model_entropy_args[@]}" --mode fixed_prefix --prefix_bank "$fixed_prefix_bank" --output "$fixed_token_metrics")
      if [[ "$FIXED_PREFIX_MAX_RECORDS" -ge 0 ]]; then fixed_entropy_args+=(--max_prefix_records "$FIXED_PREFIX_MAX_RECORDS"); fi
      "$python_bin" -m response_analysis.compute_token_entropy "${fixed_entropy_args[@]}"
    fi
  fi

  if [[ "$RUN_SURFACE_DIVERSITY" == "1" ]]; then
    echo "[$(date)] Computing surface and answer diversity for $model_id"
    "$python_bin" -m response_analysis.compute_surface_diversity \
      --input "$generations" \
      --output "$response_metrics"
  fi

  if [[ "$RUN_SEMANTIC_JUDGE" == "1" ]]; then
    echo "[$(date)] Running semantic strategy judge for $model_id"
    judge_args=(
      --input "$generations"
      --output "$semantic_judgments"
      --metrics_output "$strategy_metrics"
      --cache_dir "$JUDGE_CACHE_DIR/$model_id"
      --model "$OPENAI_EVALUATOR_MODEL"
      --base_url "$OPENAI_BASE_URL"
      --max_prompts "$JUDGE_MAX_PROMPTS"
      --shuffle_repeats "$JUDGE_SHUFFLE_REPEATS"
      --seed "$SEED"
      --json_mode "$JUDGE_JSON_MODE"
    )
    if [[ "$JUDGE_DISABLE_API" == "1" ]]; then judge_args+=(--disable_api); fi
    "$python_bin" -m response_analysis.judge_strategy_diversity "${judge_args[@]}"
  fi
}

model_dirs=()
strategy_args=()
if [[ "$RUN_DENSE" == "1" ]]; then
  dense_dir="$RUN_ROOT/dense"
  run_model_pipeline "$BASE_MODEL_ID" "$dense_dir" 0
  model_dirs+=("$dense_dir")
fi
if [[ "$RUN_PRUNED" == "1" ]]; then
  pruned_dir="$RUN_ROOT/pruned_s${PRUNING_SPARSITY}"
  run_model_pipeline "$PRUNED_MODEL_ID" "$pruned_dir" 1
  model_dirs+=("$pruned_dir")
fi

if [[ "$RUN_AGGREGATE" == "1" ]]; then
  echo "[$(date)] Combining model outputs and aggregating"
  combined_dir="$RUN_ROOT/combined"
  mkdir -p "$combined_dir"
  : > "$combined_dir/generations.jsonl"
  for dir in "${model_dirs[@]}"; do
    if [[ -f "$dir/generations.jsonl" ]]; then cat "$dir/generations.jsonl" >> "$combined_dir/generations.jsonl"; fi
  done

  "$python_bin" - <<'PY' "$combined_dir" "${model_dirs[@]}"
from pathlib import Path
import sys
import pandas as pd
combined = Path(sys.argv[1])
model_dirs = [Path(arg) for arg in sys.argv[2:]]
for name in ["token_metrics", "fixed_token_metrics", "response_metrics", "strategy_metrics"]:
    frames = []
    for directory in model_dirs:
        path = directory / f"{name}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(combined / f"{name}.parquet", index=False)
PY

  aggregate_args=(
    --generations "$combined_dir/generations.jsonl"
    --per_prompt_output "$combined_dir/per_prompt_metrics.csv"
    --aggregate_output "$combined_dir/aggregate_metrics.csv"
    --paired_output "$combined_dir/paired_comparisons.csv"
    --figures_dir "$combined_dir/figures"
    --baseline_model "$BASELINE_MODEL"
    --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    --seed "$SEED"
  )
  if [[ -f "$combined_dir/token_metrics.parquet" ]]; then
    aggregate_args+=(--token_metrics "$combined_dir/token_metrics.parquet")
  fi
  if [[ -f "$combined_dir/fixed_token_metrics.parquet" ]]; then
    aggregate_args+=(--fixed_token_metrics "$combined_dir/fixed_token_metrics.parquet")
  fi
  if [[ -f "$combined_dir/response_metrics.parquet" ]]; then
    aggregate_args+=(--response_metrics "$combined_dir/response_metrics.parquet")
  fi
  if [[ -f "$combined_dir/strategy_metrics.parquet" ]]; then
    aggregate_args+=(--strategy_metrics "$combined_dir/strategy_metrics.parquet")
  fi
  "$python_bin" -m response_analysis.aggregate_results "${aggregate_args[@]}"
fi

echo "[$(date)] Response analysis complete: $RUN_ROOT"
