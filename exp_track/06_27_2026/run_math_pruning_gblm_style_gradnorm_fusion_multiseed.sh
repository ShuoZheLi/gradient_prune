#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
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

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_VISIBLE_GPUS[@]}
fi
export CUDA_VISIBLE_DEVICES

for SEED in 42 111 222; do
  CONFIG="exp_track/06_27_2026/qwen25_1p5b_math_gblm_style_gradnorm_fusion_seed${SEED}.yaml"
  RESULT_ROOT="results/qwen25_1p5b_gblm_style_gradnorm_fusion_seed${SEED}"
  echo "=== Running GBLM-style gradnorm fusion seed ${SEED} ==="
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
done

python scripts/summarize_gblm_style_gradnorm_fusion_multiseed.py \
  --result-roots \
    results/qwen25_1p5b_gblm_style_gradnorm_fusion_seed42 \
    results/qwen25_1p5b_gblm_style_gradnorm_fusion_seed111 \
    results/qwen25_1p5b_gblm_style_gradnorm_fusion_seed222 \
  --output-dir results/qwen25_1p5b_gblm_style_gradnorm_fusion_multiseed_summary
