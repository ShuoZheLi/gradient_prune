#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

CONFIG=exp_track/06_27_2026/qwen25_1p5b_math_wanda_filter_signed_taylor.yaml
RESULT_ROOT=results/qwen25_1p5b_wanda_filter_signed_taylor_math7500

python scripts/build_wanda_filter_signed_taylor_masks.py \
  --config "$CONFIG" \
  --signed-taylor-score-dir results/qwen25_1p5b_signed_taylor_math7500/scores \
  --wanda-activation-dir results/qwen25_1p5b_wanda_math7500/stats/activations \
  --device cuda:0

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

python scripts/compare_wanda_filter_signed_taylor.py \
  --result-root "$RESULT_ROOT" \
  --wanda-root results/qwen25_1p5b_wanda_math7500 \
  --signed-taylor-score-dir results/qwen25_1p5b_signed_taylor_math7500/scores \
  --wanda-activation-dir results/qwen25_1p5b_wanda_math7500/stats/activations \
  --output-dir "$RESULT_ROOT/rerank_diagnostics" \
  --device cuda:0
