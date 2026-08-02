#!/bin/bash
#SBATCH --job-name=nll_qwen3_4b_base_gs0
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=nll_qwen3_4b_base_global_step_0_5x7500-%j.out
#SBATCH --error=nll_qwen3_4b_base_global_step_0_5x7500-%j.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
else
  echo "VENV activate script not found at ${VENV}/bin/activate; using current Python environment."
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/.cache/uv}"
HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${SCRATCH}/data/embeddings}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH}/.cache/torch_extensions}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE" "$TORCH_EXTENSIONS_DIR"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export TORCH_EXTENSIONS_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID || true

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity and paths
# -----------------------------
RUN_NAME="${RUN_NAME:-nll_qwen3_4b_base_global_step_0_5x7500}"
REAL_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_NAME}_${REAL_SLURM_JOB_ID}"

WORK_DIR="${WORK_DIR:-/work/09576/shuozhe/gradient_prune/verl}"
REPO_DIR="${REPO_DIR:-/work/09576/shuozhe/gradient_prune}"
MODEL_INIT_CKPT="${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-4B-Base}"
DATA_FILE="${DATA_FILE:-/work/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math7500_correct_5_response/qwen3-8b-instruct_math7500_correct_5_response.parquet}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${WORK_DIR}:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/verl_runs}"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/teacher_trajectory_nll}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/work/09576/shuozhe/gradient_prune/verl/train_log_archive}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
STDOUT_LOG="${LOG_DIR}/job_${RUN_ID}.txt"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$ARCHIVE_ROOT"

# -----------------------------
# Teacher-trajectory NLL defaults
# -----------------------------
batch_size="${batch_size:-1}"
max_length="${max_length:-14336}"
truncation="${truncation:-right}"
max_samples="${max_samples:--1}"
only_correct="${only_correct:-true}"
prompt_key="${prompt_key:-prompt}"
response_key="${response_key:-response}"
device="${device:-auto}"
dtype="${dtype:-auto}"
trust_remote_code="${trust_remote_code:-false}"
attn_implementation="${attn_implementation:-}"
strict_mask="${strict_mask:-true}"

# This launcher intentionally measures the dense standalone 4B model; no pruning mask or score file is used.

# -----------------------------
# Helpers
# -----------------------------
MODEL_PATH_RESOLVER="${WORK_DIR}/tools/resolve_model_init_path.py"

resolve_model_init_path() {
  local raw_path="$1"
  local role="$2"

  if [[ ! -f "$MODEL_PATH_RESOLVER" ]]; then
    echo "Missing model path resolver: $MODEL_PATH_RESOLVER" >&2
    return 1
  fi

  python3 "$MODEL_PATH_RESOLVER" \
    --path "$raw_path" \
    --role "$role" \
    --log-dir "$LOG_DIR"
}

describe_path() {
  local label="$1"
  local path="$2"

  echo "$label: $path"
  if [[ "$path" == "null" || -z "$path" ]]; then
    echo "  disabled/null"
  elif [[ "$path" == *"://"* ]]; then
    echo "  non-local URI path"
  elif [[ -d "$path" || -f "$path" ]]; then
    ls -ld "$path"
  elif [[ "$path" = /* || "$path" == ./* || "$path" == ../* || "$path" == ~* ]]; then
    echo "  local path not found"
  else
    echo "  passthrough model identifier"
  fi
}

sync_to_work() {
  echo "Syncing NLL outputs/logs back to archive."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run files to: $ARCHIVE_DIR"
  echo "Primary NLL output dir: $OUTPUT_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

MODEL_PATH="$(resolve_model_init_path "$MODEL_INIT_CKPT" actor)"

# -----------------------------
# Debug info and input checks
# -----------------------------
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Run ID: ${RUN_ID}"
echo "SLURM nodes: ${SLURM_JOB_NODELIST:-local}"
echo "SCRATCH: ${SCRATCH}"
echo "RUN_DIR: ${RUN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
describe_path "WORK_DIR" "$WORK_DIR"
describe_path "REPO_DIR" "$REPO_DIR"
describe_path "MODEL_INIT_CKPT" "$MODEL_INIT_CKPT"
describe_path "MODEL_PATH" "$MODEL_PATH"
describe_path "DATA_FILE" "$DATA_FILE"
echo "pruning: disabled"
echo "batch_size: $batch_size"
echo "max_length: $max_length"
echo "truncation: $truncation"
echo "max_samples: $max_samples"
echo "only_correct: $only_correct"

ls -ld "$WORK_DIR" "$REPO_DIR"
ls -lh "$DATA_FILE"

EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_DIR}/src/measure_teacher_trajectory_nll.py}"
NLL_ARGS=(
  --model "$MODEL_PATH"
  --data "$DATA_FILE"
  --output-dir "$OUTPUT_DIR"
  --prompt-key "$prompt_key"
  --response-key "$response_key"
  --batch-size "$batch_size"
  --max-length "$max_length"
  --truncation "$truncation"
  --max-samples "$max_samples"
  --device "$device"
  --dtype "$dtype"
)

if [[ "$strict_mask" == "true" || "$strict_mask" == "True" ]]; then
  NLL_ARGS+=(--strict-mask)
else
  NLL_ARGS+=(--no-strict-mask)
fi
if [[ "$only_correct" == "true" || "$only_correct" == "True" ]]; then
  NLL_ARGS+=(--only-correct)
fi
if [[ "$trust_remote_code" == "true" || "$trust_remote_code" == "True" ]]; then
  NLL_ARGS+=(--trust-remote-code)
fi
if [[ -n "$attn_implementation" ]]; then
  NLL_ARGS+=(--attn-implementation "$attn_implementation")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; resolved NLL args:"
  printf '  %q\n' "${NLL_ARGS[@]}"
  exit 0
fi

# -----------------------------
# Run teacher-trajectory NLL
# -----------------------------
cd "$REPO_DIR"
python3 "$EVAL_SCRIPT" "${NLL_ARGS[@]}" "$@" 2>&1 | tee "$STDOUT_LOG"
