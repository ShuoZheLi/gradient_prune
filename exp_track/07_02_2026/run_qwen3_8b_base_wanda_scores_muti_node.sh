#!/bin/bash
#SBATCH --job-name=qwen3_8b_wanda_scores
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j_qwen3_8b_wanda_scores.out
#SBATCH --error=slurm-%j_qwen3_8b_wanda_scores.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

# Requested local environment. Override CONDA_SH/CONDA_ENV at sbatch time if needed.
CONDA_SH="${CONDA_SH:-/data/shuozhe/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  set -u
else
  echo "Conda setup not found: $CONDA_SH" >&2
  echo "Set CONDA_SH=/path/to/conda.sh when submitting on the cluster." >&2
  exit 1
fi

find_repo_root() {
  local start_dir="$1"
  local dir
  dir="$(CDPATH= cd -- "$start_dir" 2>/dev/null && pwd)" || return 1
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" && -d "$dir/scripts" && -d "$dir/src" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

# Slurm may execute a spooled copy under /var/spool. Prefer explicit overrides,
# then discover the repo by walking up from submit/current/script directories.
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
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python}"

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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

# -----------------------------
# WANDA scoring config
# -----------------------------
RUN_NAME="${RUN_NAME:-qwen3_8b_base_wanda_scores}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}}"

model_path="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B-Base}"
calibration_path="${CALIBRATION_PATH:-$repo_root/saved_calibration_dataset/qwen2.5-1.5b-instruct_math7500_correct}"
output_dir="${OUTPUT_DIR:-$repo_root/results/07_02_2026/qwen3_8b_base_wanda_scores}"
log_dir="${LOG_DIR:-$output_dir/logs/${RUN_ID}}"

max_samples="${MAX_SAMPLES:-}"
microbatch_size="${MICROBATCH_SIZE:-1}"
max_length="${MAX_LENGTH:-4096}"
dtype="${DTYPE:-bf16}"
seed="${SEED:-42}"
calibration_type="${CALIBRATION_TYPE:-prompt_response}"
prune_ops="${PRUNE_OPS:-}"
shuffle="${SHUFFLE:-0}"
only_correct="${ONLY_CORRECT:-0}"
trust_remote_code="${TRUST_REMOTE_CODE:-0}"
overwrite="${OVERWRITE:-0}"
enable_thinking="${ENABLE_THINKING:-auto}"
dry_run="${DRY_RUN:-0}"

mkdir -p "$output_dir" "$log_dir"

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

if [[ ! -e "$model_path" ]]; then
  echo "Model path not found: $model_path" >&2
  exit 1
fi
if [[ ! -e "$calibration_path" ]]; then
  echo "Calibration path not found: $calibration_path" >&2
  exit 1
fi

score_args=(
  scripts/score_wanda.py
  --model "$model_path"
  --calibration "$calibration_path"
  --output-dir "$output_dir"
  --calibration-type "$calibration_type"
  --microbatch-size "$microbatch_size"
  --max-length "$max_length"
  --dtype "$dtype"
  --seed "$seed"
  --enable-thinking "$enable_thinking"
)
[[ -n "$max_samples" ]] && score_args+=(--max-samples "$max_samples")
[[ "$shuffle" == "1" ]] && score_args+=(--shuffle)
[[ "$only_correct" == "1" ]] && score_args+=(--only-correct)
[[ "$trust_remote_code" == "1" ]] && score_args+=(--trust-remote-code)
[[ "$overwrite" == "1" ]] && score_args+=(--overwrite)
if [[ -n "$prune_ops" ]]; then
  # shellcheck disable=SC2206
  prune_ops_array=($prune_ops)
  score_args+=(--prune-ops "${prune_ops_array[@]}")
fi

torchrun_args=(
  --nnodes "$num_nodes"
  --nproc-per-node "$nproc_per_node"
  --rdzv-backend c10d
  --rdzv-endpoint "${master_addr}:${master_port}"
  --rdzv-id "wanda_${SLURM_JOB_ID:-manual}"
)

# -----------------------------
# Debug info
# -----------------------------
echo "[wanda] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[wanda] Run ID: $RUN_ID"
echo "[wanda] repo_root=$repo_root"
echo "[wanda] nodes=${nodes_array[*]}"
echo "[wanda] num_nodes=$num_nodes nproc_per_node=$nproc_per_node world_size=$world_size"
echo "[wanda] master=${master_addr}:${master_port}"
echo "[wanda] model_path=$model_path"
echo "[wanda] calibration_path=$calibration_path"
echo "[wanda] output_dir=$output_dir"
echo "[wanda] log_dir=$log_dir"
echo "[wanda] cache_root=$cache_root"
echo "[wanda] conda_env=$CONDA_ENV"
echo "[wanda] enable_thinking=$enable_thinking"
printf '[wanda] command:'
printf ' %q' torchrun "${torchrun_args[@]}" "${score_args[@]}"
printf '\n'

if [[ "$dry_run" == "1" ]]; then
  echo "[wanda] dry run complete; no scoring launched."
  exit 0
fi

# One torchrun launcher per node. torchrun coordinates all ranks through the same rendezvous endpoint.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  exec srun --nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1 \
    torchrun "${torchrun_args[@]}" "${score_args[@]}"
fi

exec torchrun --standalone --nnodes=1 --nproc-per-node="$nproc_per_node" "${score_args[@]}"
