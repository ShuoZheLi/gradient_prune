#!/bin/bash
#SBATCH --job-name=eval_qwen3_4b_global_step_800
#SBATCH --account=ASC26008
#SBATCH --partition=gh-dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j_eval_qwen3_4b_global_step_800.out
#SBATCH --error=slurm-%j_eval_qwen3_4b_global_step_800.err

set -euo pipefail

export CHECKPOINT_NAME="global_step_800"
export MODEL_PATH="/scratch/09576/shuozhe/verl_runs/sft_qwen3_4b_base_5x7500_874991/train_log/global_step_800"
export DATASET_PATH="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
export RUN_NAME="${RUN_NAME:-sft_qwen3_4b_base_global_step_800_math500_eval}"

common_script="${EVAL_COMMON_SCRIPT:-}"
if [[ -z "$common_script" ]]; then
  common_candidates=(
    "$(dirname -- "${BASH_SOURCE[0]}")/eval_checkpoint_common.sh"
    "${SLURM_SUBMIT_DIR:-}/eval_checkpoint_common.sh"
    "/data/shuozhe/gradient_prune/exp_track/07_28_2026/eval_distill/sft_qwen3_4b_base/eval_checkpoint_common.sh"
    "/work2/09576/shuozhe/gradient_prune/exp_track/07_28_2026/eval_distill/sft_qwen3_4b_base/eval_checkpoint_common.sh"
  )
  for candidate in "${common_candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate" ]]; then
      common_script="$candidate"
      break
    fi
  done
fi
if [[ -z "$common_script" || ! -f "$common_script" ]]; then
  echo "Could not locate eval_checkpoint_common.sh. Set EVAL_COMMON_SCRIPT=/path/to/eval_checkpoint_common.sh" >&2
  exit 1
fi
exec "$common_script"
