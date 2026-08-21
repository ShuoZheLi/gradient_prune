#!/bin/bash
#SBATCH --job-name=qwen3_8b_dap_nll_sft
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j_qwen3_8b_dap_nll_sft.out
#SBATCH --error=slurm-%j_qwen3_8b_dap_nll_sft.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

find_repo_root() {
  local candidate
  for candidate in \
    "${WORK_DIR:-}" \
    "${REPO_ROOT:-}" \
    "/work2/09576/shuozhe/gradient_prune" \
    "$PWD"; do
    if [[ -n "$candidate" && -f "$candidate/generate_recoverability_scores.py" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

REPO_ROOT="$(find_repo_root)" || {
  echo "Could not locate gradient_prune repository." >&2
  exit 2
}
cd "$REPO_ROOT"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/scratch/09576/shuozhe}}"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/triton/${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

MODEL_PATH="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
REF_DATASET_PATH="${REF_DATASET_PATH:-$REPO_ROOT/tmp/qwen3-8b-instruct_math500_correct.parquet}"
KD_DATASET_PATH="${KD_DATASET_PATH:-$REPO_ROOT/tmp/qwen3-8b-instruct_math7500_correct_5_response.parquet}"

RUN_NAME="${RUN_NAME:-qwen3_8b_dap_nll_sft}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/gradient_prune/results/dap_nll_sft/$RUN_ID}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/recoverability_scores.pt}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2> >(tee -a "$LOG_DIR/run.err" >&2)

for required_path in "$MODEL_PATH" "$REF_DATASET_PATH" "$KD_DATASET_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 3
  fi
done

NUM_PROBES="${NUM_PROBES:-16}"
PROBE_SEED="${PROBE_SEED:-42}"
PROBE_LR_ETA="${PROBE_LR_ETA:-1e-5}"
MAX_REF_SAMPLES="${MAX_REF_SAMPLES:-8}"
MAX_KD_SAMPLES="${MAX_KD_SAMPLES:-8}"
MAX_LENGTH="${MAX_LENGTH:-18432}"
SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-$MAX_LENGTH}"
REF_BATCH_SIZE="${REF_BATCH_SIZE:-1}"
KD_BATCH_SIZE="${KD_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
CONVERGENCE_CHECKPOINTS="${CONVERGENCE_CHECKPOINTS:-2,4,8,16}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"
RUN_CONVERGENCE_ANALYSIS="${RUN_CONVERGENCE_ANALYSIS:-1}"
SAVE_INTERMEDIATE_STATS="${SAVE_INTERMEDIATE_STATS:-0}"
SHUFFLE_DATASETS="${SHUFFLE_DATASETS:-1}"
LOSS_ON="${LOSS_ON:-full_trajectory}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

filter_convergence_checkpoints() {
  local raw="$1"
  local maximum="$2"
  local filtered=""
  local checkpoint
  local -a checkpoints=()
  IFS=',' read -r -a checkpoints <<< "$raw"
  for checkpoint in "${checkpoints[@]}"; do
    checkpoint="${checkpoint//[[:space:]]/}"
    if [[ "$checkpoint" =~ ^[0-9]+$ ]] && (( 10#$checkpoint >= 2 && 10#$checkpoint <= maximum )); then
      filtered+="${filtered:+,}${checkpoint}"
    fi
  done
  printf '%s\n' "$filtered"
}

CONVERGENCE_CHECKPOINTS="$(filter_convergence_checkpoints "$CONVERGENCE_CHECKPOINTS" "$NUM_PROBES")"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --ref_dataset_path "$REF_DATASET_PATH"
  --kd_dataset_path "$KD_DATASET_PATH"
  --probe_lr_eta "$PROBE_LR_ETA"
  --probe_seed "$PROBE_SEED"
  --max_length "$MAX_LENGTH"
  --ref_batch_size "$REF_BATCH_SIZE"
  --kd_batch_size "$KD_BATCH_SIZE"
  --dtype "$DTYPE"
  --device "$DEVICE_MAP"
  --loss_on "$LOSS_ON"
  --truncation_side right
  --hvp_parameter_scope all
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_ARGS+=(--gradient_checkpointing)
fi

echo "[dap_nll_sft] repo_root=$REPO_ROOT"
echo "[dap_nll_sft] model_path=$MODEL_PATH"
echo "[dap_nll_sft] ref_dataset_path=$REF_DATASET_PATH"
echo "[dap_nll_sft] kd_dataset_path=$KD_DATASET_PATH"
echo "[dap_nll_sft] output_path=$OUTPUT_PATH"
echo "[dap_nll_sft] num_probes=$NUM_PROBES eta=$PROBE_LR_ETA"
echo "[dap_nll_sft] max_ref_samples=$MAX_REF_SAMPLES max_kd_samples=$MAX_KD_SAMPLES max_length=$MAX_LENGTH"
echo "[dap_nll_sft] loss_on=$LOSS_ON hvp_parameter_scope=all"
echo "[dap_nll_sft] gradient_checkpointing=$GRADIENT_CHECKPOINTING smoke_max_length=$SMOKE_MAX_LENGTH"
echo "[dap_nll_sft] visible_gpus=${CUDA_VISIBLE_DEVICES:-all}"
echo "[dap_nll_sft] python=$(command -v python)"
echo "[dap_nll_sft] RESOURCE NOTE: Qwen3-8B's seven candidate matrices contain about 6.95B weights."
echo "[dap_nll_sft] RESOURCE NOTE: allow roughly 200+ GiB host RAM and 180+ GiB scratch for M=16 with four score snapshots."
if [[ "$LOSS_ON" == "full_trajectory" ]]; then
  echo "[dap_nll_sft] IMPORTANT: full prompt+response NLL is intentional because the supplied parquet files lack prompt_length/loss_mask metadata."
fi

if [[ "$RUN_SMOKE_FIRST" == "1" ]]; then
  SMOKE_PATH="$OUTPUT_ROOT/smoke/recoverability_smoke.pt"
  mkdir -p "$(dirname "$SMOKE_PATH")"
  python generate_recoverability_scores.py \
    "${COMMON_ARGS[@]}" \
    --output_path "$SMOKE_PATH" \
    --num_probes 1 \
    --max_length "$SMOKE_MAX_LENGTH" \
    --max_ref_samples 1 \
    --max_kd_samples 1 \
    --candidate_modules q_proj \
    --smoke_test
fi

SCORE_ARGS=(
  "${COMMON_ARGS[@]}"
  --output_path "$OUTPUT_PATH"
  --num_probes "$NUM_PROBES"
  --candidate_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
)
if [[ -n "$MAX_REF_SAMPLES" && "$MAX_REF_SAMPLES" != "all" && "$MAX_REF_SAMPLES" != "none" ]]; then
  SCORE_ARGS+=(--max_ref_samples "$MAX_REF_SAMPLES")
fi
if [[ -n "$MAX_KD_SAMPLES" && "$MAX_KD_SAMPLES" != "all" && "$MAX_KD_SAMPLES" != "none" ]]; then
  SCORE_ARGS+=(--max_kd_samples "$MAX_KD_SAMPLES")
fi
if [[ "$SHUFFLE_DATASETS" == "1" ]]; then
  SCORE_ARGS+=(--shuffle_ref --shuffle_kd)
fi
if [[ -n "$CONVERGENCE_CHECKPOINTS" ]]; then
  SCORE_ARGS+=(--convergence_checkpoints "$CONVERGENCE_CHECKPOINTS")
fi
if [[ "$SAVE_INTERMEDIATE_STATS" == "1" ]]; then
  SCORE_ARGS+=(--save_intermediate_stats)
fi

python generate_recoverability_scores.py "${SCORE_ARGS[@]}"

if [[ "$RUN_CONVERGENCE_ANALYSIS" == "1" ]]; then
  mapfile -t SNAPSHOTS < <(find "$OUTPUT_ROOT" -path '*_convergence/scores_probe_*.pt' -type f | sort)
  if (( ${#SNAPSHOTS[@]} >= 2 )); then
    python analyze_recoverability_convergence.py \
      "${SNAPSHOTS[@]}" \
      --output_path "$OUTPUT_ROOT/convergence_spearman.json"
  else
    echo "[dap_nll_sft] fewer than two convergence snapshots; skipping Spearman analysis"
  fi
fi

echo "[dap_nll_sft] completed successfully: $OUTPUT_PATH"
