#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m experiment_runner \
  --config configs/qwen25_0p5b_debug.yaml

python -m plotting \
  --results_csv results/debug/tables/main_results.csv \
  --output_dir results/debug/plots \
  --score_dir results/debug/scores
