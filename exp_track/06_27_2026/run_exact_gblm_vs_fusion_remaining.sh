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
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export HF_HOME="$REPO_ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$REPO_ROOT/.cache/hf/hub"
export TRANSFORMERS_CACHE="$REPO_ROOT/.cache/hf/transformers"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm"
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export TMPDIR="$REPO_ROOT/.cache/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_VISIBLE_GPUS[@]}
fi

configs=(
  exp_track/06_27_2026/qwen25_1p5b_exact_gblm_vs_fusion_additive_seed42.yaml
  exp_track/06_27_2026/qwen25_1p5b_exact_gblm_vs_fusion_additive_seed111.yaml
  exp_track/06_27_2026/qwen25_1p5b_exact_gblm_vs_fusion_additive_seed222.yaml
  exp_track/06_27_2026/qwen25_1p5b_exact_gblm_vs_fusion_fusion_seed42_lambda1_seed42.yaml
)

for config in "${configs[@]}"; do
  echo "[$(date)] Running ${config} on ${NUM_GPUS} GPUs"
  torchrun --standalone --nproc_per_node "$NUM_GPUS" -m experiment_runner --config "$config"
done
