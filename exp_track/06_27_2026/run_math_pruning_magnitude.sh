#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_VISIBLE_GPUS[@]}
fi
export CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node "$NUM_GPUS" -m experiment_runner \
  --config exp_track/06_27_2026/qwen25_1p5b_math_magnitude.yaml

python -m plotting \
  --results_csv results/qwen25_1p5b_magnitude_math/tables/main_results.csv \
  --output_dir results/qwen25_1p5b_magnitude_math/plots \
  --score_dir results/qwen25_1p5b_magnitude_math/scores
