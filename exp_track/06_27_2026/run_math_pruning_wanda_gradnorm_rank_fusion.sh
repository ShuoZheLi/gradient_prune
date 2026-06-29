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

CONFIG=exp_track/06_27_2026/qwen25_1p5b_math_wanda_gradnorm_rank_fusion.yaml
RESULT_ROOT=results/qwen25_1p5b_wanda_gradnorm_rank_fusion_math7500

if [[ ! -f "$RESULT_ROOT/masks/method=wanda_gradnorm_rank_fusion/sparsity=0.4/lambda=1.0/metadata.json" ]]; then
  python scripts/build_wanda_gradnorm_rank_fusion_masks.py \
    --config "$CONFIG" \
    --gradient-stats-dir results/qwen25_1p5b_signed_taylor_math7500/stats/gradients \
    --wanda-activation-dir results/qwen25_1p5b_wanda_math7500/stats/activations \
    --device cuda:0
else
  echo "Rank-fusion masks already exist; skipping mask build."
fi

for SPARSITY in 0.1 0.2 0.3 0.4; do
  SRC="results/qwen25_1p5b_wanda_math7500/masks/method=wanda/sparsity=${SPARSITY}/lambda=none"
  DST="${RESULT_ROOT}/masks/method=wanda/sparsity=${SPARSITY}/lambda=none"
  mkdir -p "$(dirname "$DST")"
  rm -rf "$DST"
  cp -a "$SRC" "$DST"
done

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_VISIBLE_GPUS[@]}
fi
export CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node "$NUM_GPUS" -m experiment_runner \
  --config "$CONFIG"

python -m plotting \
  --results_csv "$RESULT_ROOT/tables/main_results.csv" \
  --output_dir "$RESULT_ROOT/plots" \
  --score_dir results/qwen25_1p5b_signed_taylor_math7500/scores

python scripts/first_token_rank_fusion_diagnostics.py \
  --result-root "$RESULT_ROOT" \
  --device cuda:0 \
  --num-prompts 100 \
  --max-new-tokens 256
