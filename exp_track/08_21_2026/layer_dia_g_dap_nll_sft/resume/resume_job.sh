#!/bin/bash
#SBATCH --job-name=qwen3_8b_layer_dia_g_resume
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j_qwen3_8b_layer_dia_g_resume.out
#SBATCH --error=slurm-%j_qwen3_8b_layer_dia_g_resume.err

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
    if [[ -n "$candidate" && -f "$candidate/generate_layer_factorized_recoverability_scores.py" ]]; then
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/gradient_prune/results/layer_dia_g_dap_nll_sft/qwen3_8b_layer_dia_g_dap_nll_sft_930296_20260822_231956}"
SCORE_DIR="${SCORE_DIR:-$OUTPUT_ROOT/scores}"
LOG_DIR="$OUTPUT_ROOT/logs"

if [[ ! -d "$SCORE_DIR" ]]; then
  echo "Resume score directory does not exist: $SCORE_DIR" >&2
  exit 3
fi
if ! compgen -G "$SCORE_DIR/manifest.rank*.json" >/dev/null; then
  echo "No rank manifests found in resume score directory: $SCORE_DIR" >&2
  exit 3
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/resume_${SLURM_JOB_ID:-manual}.log") \
  2> >(tee -a "$LOG_DIR/resume_${SLURM_JOB_ID:-manual}.err" >&2)

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/triton/${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID || true

MODEL_PATH="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
REF_DATASET_PATH="${REF_DATASET_PATH:-/work2/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math500_correct/qwen3-8b-instruct_math500_correct.parquet}"
KD_DATASET_PATH="${KD_DATASET_PATH:-/work/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math7500_correct/qwen3-8b-instruct_math7500_correct.parquet}"

for required_path in "$MODEL_PATH" "$REF_DATASET_PATH" "$KD_DATASET_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 4
  fi
done

ETA="${ETA:-1e-5}"
MAX_REF_SAMPLES="${MAX_REF_SAMPLES:-all}"
MAX_KD_SAMPLES="${MAX_KD_SAMPLES:-600}"
MAX_LENGTH="${MAX_LENGTH:-9216}"
REF_BATCH_SIZE="${REF_BATCH_SIZE:-1}"
KD_BATCH_SIZE="${KD_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"
LOSS_ON="${LOSS_ON:-response_only}"
DERIVE_PROMPT_LENGTH_FROM_PROMPT="${DERIVE_PROMPT_LENGTH_FROM_PROMPT:-1}"
PROMPT_TEXT_COLUMN="${PROMPT_TEXT_COLUMN:-prompt}"
CANDIDATE_MODULES="${CANDIDATE_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LAYER_GROUP_SIZE="${LAYER_GROUP_SIZE:-1}"
G_STRUCTURE="${G_STRUCTURE:-diagonal}"
FACTOR_STORAGE_DEVICE="${FACTOR_STORAGE_DEVICE:-cpu}"
FACTOR_CHUNK_SIZE="${FACTOR_CHUNK_SIZE:-2048}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-cpu}"
ACTIVATION_OFFLOAD_PIN_MEMORY="${ACTIVATION_OFFLOAD_PIN_MEMORY:-0}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-1}"
SAVE_FACTOR_DIAGNOSTICS="${SAVE_FACTOR_DIAGNOSTICS:-0}"
SCORE_SHARD_FORMAT="${SCORE_SHARD_FORMAT:-pt}"
SHUFFLE_DATASETS="${SHUFFLE_DATASETS:-1}"
SEED="${SEED:-42}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-1440}"

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
RDZV_PORT="${RDZV_PORT:-29500}"
RUN_ID="qwen3_8b_layer_dia_g_resume_${SLURM_JOB_ID:-manual}"

if (( GPUS_PER_NODE != 1 )); then
  echo "This launcher expects one process/GPU per node; got GPUS_PER_NODE=$GPUS_PER_NODE." >&2
  exit 5
fi
if (( WORLD_SIZE != 8 )); then
  echo "Resume requires the original world size of 8; got WORLD_SIZE=$WORLD_SIZE." >&2
  exit 5
fi

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
MASTER_ADDR=""
for candidate_ip in $head_node_ip; do
  if [[ "$candidate_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    MASTER_ADDR="$candidate_ip"
    break
  fi
done
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="${head_node_ip%% *}"
fi
RDZV_ENDPOINT="$MASTER_ADDR:$RDZV_PORT"

ARGS=(
  --model_path "$MODEL_PATH"
  --ref_dataset_path "$REF_DATASET_PATH"
  --kd_dataset_path "$KD_DATASET_PATH"
  --output_dir "$SCORE_DIR"
  --eta "$ETA"
  --max_length "$MAX_LENGTH"
  --ref_batch_size "$REF_BATCH_SIZE"
  --kd_batch_size "$KD_BATCH_SIZE"
  --candidate_modules "$CANDIDATE_MODULES"
  --layer_group_size "$LAYER_GROUP_SIZE"
  --g_structure "$G_STRUCTURE"
  --factor_storage_device "$FACTOR_STORAGE_DEVICE"
  --factor_chunk_size "$FACTOR_CHUNK_SIZE"
  --activation_offload "$ACTIVATION_OFFLOAD"
  --dtype "$DTYPE"
  --device cuda:0
  --loss_on "$LOSS_ON"
  --prompt_text_column "$PROMPT_TEXT_COLUMN"
  --attention_implementation "$ATTENTION_IMPLEMENTATION"
  --diagnostic_batches "$DIAGNOSTIC_BATCHES"
  --score_shard_format "$SCORE_SHARD_FORMAT"
  --distributed_timeout_minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
  --seed "$SEED"
  --distributed_layer_sharding
  --resume
)
if [[ "$MAX_REF_SAMPLES" != "all" ]]; then
  ARGS+=(--max_ref_samples "$MAX_REF_SAMPLES")
fi
if [[ "$MAX_KD_SAMPLES" != "all" ]]; then
  ARGS+=(--max_kd_samples "$MAX_KD_SAMPLES")
fi
if [[ "$SHUFFLE_DATASETS" == "1" ]]; then
  ARGS+=(--shuffle_ref --shuffle_kd)
fi
if [[ "$SAVE_FACTOR_DIAGNOSTICS" == "1" ]]; then
  ARGS+=(--save_factor_diagnostics)
fi
if [[ "$ACTIVATION_OFFLOAD_PIN_MEMORY" == "1" ]]; then
  ARGS+=(--activation_offload_pin_memory)
fi
if [[ "$DERIVE_PROMPT_LENGTH_FROM_PROMPT" == "1" ]]; then
  ARGS+=(--derive_prompt_length_from_prompt)
fi

echo "[layer_dia_g_dap_resume] output_root=$OUTPUT_ROOT"
echo "[layer_dia_g_dap_resume] score_dir=$SCORE_DIR"
echo "[layer_dia_g_dap_resume] nodes=$NNODES world_size=$WORLD_SIZE timeout_minutes=$DISTRIBUTED_TIMEOUT_MINUTES"

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    source "'"$VENV"'/bin/activate"
    cd "'"$REPO_ROOT"'"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export TRITON_CACHE_DIR="'"$TRITON_CACHE_DIR"'/node_${SLURM_PROCID}"
    mkdir -p "$TRITON_CACHE_DIR"
    torchrun \
      --nnodes="'"$NNODES"'" \
      --nproc_per_node="'"$GPUS_PER_NODE"'" \
      --node_rank="${SLURM_PROCID}" \
      --rdzv_id="'"$RUN_ID"'" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="'"$RDZV_ENDPOINT"'" \
      generate_layer_factorized_recoverability_scores.py \
      "$@"
  ' _ "${ARGS[@]}"

echo "[layer_dia_g_dap_resume] completed successfully: $SCORE_DIR/manifest.json"
