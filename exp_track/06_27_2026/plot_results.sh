#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RESULTS_CSV=${1:-results/qwen25_1p5b_signed_taylor_math/tables/main_results.csv}
OUTPUT_DIR=${2:-results/qwen25_1p5b_signed_taylor_math/plots}
SCORE_DIR=${3:-results/qwen25_1p5b_signed_taylor_math/scores}

python -m plotting \
  --results_csv "$RESULTS_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --score_dir "$SCORE_DIR"
