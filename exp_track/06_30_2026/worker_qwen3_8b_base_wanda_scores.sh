#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/shuozhe/gradient_prune"
MODEL_PATH="/data/shuozhe/saved_model/Qwen3-8B-Base"
CALIBRATION_PATH="$REPO_ROOT/saved_calibration_dataset/qwen2.5-1.5b-instruct_math7500_correct"
OUTPUT_DIR="$REPO_ROOT/results/06_30_2026/qwen3_8b_base_wanda_scores"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"

cd "$REPO_ROOT"
set +u
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

score_args=(scripts/score_wanda.py
  --model "$MODEL_PATH"
  --calibration "$CALIBRATION_PATH"
  --output-dir "$OUTPUT_DIR"
  --calibration-type prompt_response
  --microbatch-size "$MICROBATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --dtype "$DTYPE"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  exec torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" "${score_args[@]}"
fi

exec python "${score_args[@]}"
