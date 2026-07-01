#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set +u
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
CACHE_ROOT=${CACHE_ROOT:-/data/shuozhe/gradient_prune/.cache}
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/hf" "$CACHE_ROOT/matplotlib" "$CACHE_ROOT/vllm"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
SEED=222
CONFIG="exp_track/06_27_2026/qwen25_1p5b_math_gblm_style_stage2_finalists_seed222_gpu0123.yaml"
RESULT_ROOT="results/qwen25_1p5b_gblm_style_stage2_finalists_seed222"

echo "=== Continuing Stage 2 finalists seed ${SEED} on GPUs ${CUDA_VISIBLE_DEVICES} ==="
echo "Config: ${CONFIG}"
echo "Result root: ${RESULT_ROOT}"

torchrun --standalone --nproc_per_node "$NUM_GPUS" -m experiment_runner --config "$CONFIG"

python scripts/first_token_all_conditions_diagnostics.py \
  --result-root "$RESULT_ROOT" \
  --device cuda:0 \
  --num-prompts 100 \
  --max-new-tokens 256

python -m plotting \
  --results_csv "$RESULT_ROOT/tables/main_results.csv" \
  --output_dir "$RESULT_ROOT/plots" \
  --score_dir "$RESULT_ROOT/scores"
