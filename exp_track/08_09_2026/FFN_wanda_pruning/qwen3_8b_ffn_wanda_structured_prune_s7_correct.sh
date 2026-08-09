#!/bin/bash
#SBATCH --job-name=qwen3_8b_ffn_s0d7
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j_qwen3_8b_ffn_wanda_s0d7_correct.out
#SBATCH --error=slurm-%j_qwen3_8b_ffn_wanda_s0d7_correct.err

set -euo pipefail

# Self-contained multi-node launcher for structured FFN-only Wanda pruning.

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
    if [[ -f "$dir/pyproject.toml" && -d "$dir/src" ]]; then
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
export PYTHONPATH="$repo_root/src:${PYTHONPATH:-}"

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
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

# -----------------------------
# Runtime config
# -----------------------------
DEFAULT_TARGET_INTERMEDIATE_SIZE="${TARGET_INTERMEDIATE_SIZE:-}"
DEFAULT_SPARSITY="${SPARSITY:-0.7}"
SPARSITY_TAG="s${DEFAULT_SPARSITY//./d}"
if [[ -n "$DEFAULT_TARGET_INTERMEDIATE_SIZE" ]]; then
  PRUNE_TAG="target_${DEFAULT_TARGET_INTERMEDIATE_SIZE}"
else
  PRUNE_TAG="${SPARSITY_TAG}_correct"
fi
RUN_NAME="${RUN_NAME:-qwen3_8b_ffn_wanda_structured_${PRUNE_TAG}_math7500}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}}"
RESULTS_BASE="${RESULTS_BASE:-${RESULTS_ROOT:-$SCRATCH_ROOT/gradient_prune/results}}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-qwen3_8b_ffn_wanda_structured_pruning_${PRUNE_TAG}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RESULTS_BASE/$RESULTS_SUBDIR}"
RESULTS_ROOT="${RUN_OUTPUT_DIR:-$EXPERIMENT_ROOT/runs/${RUN_ID}}"
SCORE_ROOT="${SCORE_ROOT:-$EXPERIMENT_ROOT/scores/${PRUNE_TAG}}"
PRUNED_MODEL_DIR="${PRUNED_MODEL_DIR:-$EXPERIMENT_ROOT/models/${PRUNE_TAG}}"
LOG_DIR="${LOG_DIR:-$RESULTS_ROOT/logs}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2> >(tee -a "$LOG_DIR/run.err" >&2)

MODEL_PATH="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
CALIBRATION_PATH="${CALIBRATION_PATH:-/work2/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math7500_correct}"
TARGET_INTERMEDIATE_SIZE="${TARGET_INTERMEDIATE_SIZE:-$DEFAULT_TARGET_INTERMEDIATE_SIZE}"
SPARSITY="${SPARSITY:-$DEFAULT_SPARSITY}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-18432}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"
SAVE_PRUNED_MODEL="${SAVE_PRUNED_MODEL:-1}"
VALIDATE_PRUNED_FORWARD="${VALIDATE_PRUNED_FORWARD:-0}"
VALIDATION_TEXT="${VALIDATION_TEXT:-hello}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$DRY_RUN" != "1" && ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist on this node: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/path/to/HF/model visible on all compute nodes." >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ! -e "$CALIBRATION_PATH" ]]; then
  echo "Calibration path does not exist on this node: $CALIBRATION_PATH" >&2
  echo "Set CALIBRATION_PATH=/path/to/calibration dataset visible on all compute nodes." >&2
  exit 2
fi

# -----------------------------
# Slurm node/device discovery
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

num_nodes="${NUM_NODES:-${#nodes_array[@]}}"
if [[ "$num_nodes" -gt "${#nodes_array[@]}" ]]; then
  echo "NUM_NODES=$num_nodes exceeds allocated/discovered nodes=${#nodes_array[@]}." >&2
  exit 2
fi
master_addr="${MASTER_ADDR:-${nodes_array[0]}}"
master_port="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"

nproc_per_node="${NPROC_PER_NODE:-}"
if [[ -z "$nproc_per_node" ]]; then
  if [[ -n "${LOCAL_DEVICES:-${DEVICES:-}}" ]]; then
    IFS=',' read -r -a local_devices_array <<< "${LOCAL_DEVICES:-${DEVICES:-}}"
    nproc_per_node="${#local_devices_array[@]}"
    export CUDA_VISIBLE_DEVICES="${LOCAL_DEVICES:-${DEVICES:-}}"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    nproc_per_node="$SLURM_GPUS_ON_NODE"
  else
    nproc_per_node="${NUM_GPUS_PER_NODE:-1}"
  fi
fi
world_size=$((num_nodes * nproc_per_node))

if [[ "$num_nodes" -lt 1 || "$nproc_per_node" -lt 1 ]]; then
  echo "num_nodes and nproc_per_node must be positive; got num_nodes=$num_nodes nproc_per_node=$nproc_per_node" >&2
  exit 2
fi

torchrun_args=(
  --nnodes "$num_nodes"
  --nproc-per-node "$nproc_per_node"
  --rdzv-backend c10d
  --rdzv-endpoint "${master_addr}:${master_port}"
  --rdzv-id "ffn_wanda_${SLURM_JOB_ID:-manual}"
)
runner_args=(
  scripts/ffn_wanda_structured_prune.py
  --model "$MODEL_PATH"
  --calibration "$CALIBRATION_PATH"
  --output-dir "$SCORE_ROOT"
  --pruned-model-dir "$PRUNED_MODEL_DIR"
  --calibration-type prompt_response
  --microbatch-size "$MICROBATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --dtype "$DTYPE"
  --seed "$SEED"
  --only-correct
  --trust-remote-code
)
if [[ -n "$TARGET_INTERMEDIATE_SIZE" ]]; then
  runner_args+=(--target-intermediate-size "$TARGET_INTERMEDIATE_SIZE")
elif [[ -n "$SPARSITY" ]]; then
  runner_args+=(--sparsity "$SPARSITY" --round-to-multiple 256)
else
  echo "Set TARGET_INTERMEDIATE_SIZE or SPARSITY." >&2
  exit 2
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  runner_args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$SAVE_PRUNED_MODEL" == "1" || "$SAVE_PRUNED_MODEL" == "true" ]]; then
  runner_args+=(--save-pruned-model)
fi
if [[ "$VALIDATE_PRUNED_FORWARD" == "1" || "$VALIDATE_PRUNED_FORWARD" == "true" ]]; then
  runner_args+=(--validate-pruned-forward --validation-text "$VALIDATION_TEXT")
fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  runner_args+=(--overwrite)
fi

# -----------------------------
# Debug info
# -----------------------------
echo "[ffn-wanda] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[ffn-wanda] Run ID: $RUN_ID"
echo "[ffn-wanda] repo_root=$repo_root"
echo "[ffn-wanda] nodes=${nodes_array[*]}"
echo "[ffn-wanda] num_nodes=$num_nodes nproc_per_node=$nproc_per_node world_size=$world_size"
echo "[ffn-wanda] master=${master_addr}:${master_port}"
echo "[ffn-wanda] results_root=$RESULTS_ROOT"
echo "[ffn-wanda] score_root=$SCORE_ROOT"
echo "[ffn-wanda] pruned_model_dir=$PRUNED_MODEL_DIR"
echo "[ffn-wanda] model_path=$MODEL_PATH"
echo "[ffn-wanda] calibration_path=$CALIBRATION_PATH"
echo "[ffn-wanda] target_intermediate_size=$TARGET_INTERMEDIATE_SIZE sparsity=${SPARSITY:-none}"
echo "[ffn-wanda] save_pruned_model=$SAVE_PRUNED_MODEL validate_pruned_forward=$VALIDATE_PRUNED_FORWARD overwrite=$OVERWRITE"
echo "[ffn-wanda] cache_root=$cache_root"
echo "[ffn-wanda] venv=${VENV:-none}"
echo "[ffn-wanda] python=$(command -v python3 || command -v python || true)"
printf '[ffn-wanda] command:'
printf ' %q' torchrun "${torchrun_args[@]}" "${runner_args[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[ffn-wanda] dry run complete; no pruning launched."
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  srun --nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1 \
    torchrun "${torchrun_args[@]}" "${runner_args[@]}"
else
  torchrun --standalone --nnodes=1 --nproc-per-node="$nproc_per_node" "${runner_args[@]}"
fi
