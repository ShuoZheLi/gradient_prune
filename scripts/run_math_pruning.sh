#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"


CUDA_VISIBLE_DEVICES=1,2,3 python -m experiment_runner \
  --config configs/qwen25_1p5b_math.yaml

python -m plotting \
  --results_csv results/qwen25_1p5b_signed_taylor_math/tables/main_results.csv \
  --output_dir results/qwen25_1p5b_signed_taylor_math/plots \
  --score_dir results/qwen25_1p5b_signed_taylor_math/scores
