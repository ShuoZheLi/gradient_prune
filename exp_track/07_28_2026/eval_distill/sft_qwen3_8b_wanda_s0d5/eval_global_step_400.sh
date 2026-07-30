#!/bin/bash
#SBATCH --job-name=eval_qwen3_8b_wanda_s0d5_global_step_400
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j_eval_qwen3_8b_wanda_s0d5_global_step_400.out
#SBATCH --error=slurm-%j_eval_qwen3_8b_wanda_s0d5_global_step_400.err

set -euo pipefail

export CHECKPOINT_NAME="global_step_400"
export MODEL_PATH="/scratch/09576/shuozhe/verl_runs/sft_qwen3_8b_wanda_s0d5_base_5x7500_875886/train_log/global_step_400"
export DATASET_PATH="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
export RUN_NAME="${RUN_NAME:-sft_qwen3_8b_wanda_s0d5_base_global_step_400_math500_eval}"

common_script="${EVAL_COMMON_SCRIPT:-}"
if [[ -z "$common_script" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_checkpoint_common.sh" ]]; then
    common_script="${SLURM_SUBMIT_DIR}/eval_checkpoint_common.sh"
  else
    common_script="$(dirname -- "${BASH_SOURCE[0]}")/eval_checkpoint_common.sh"
  fi
fi
if [[ ! -f "$common_script" ]]; then
  echo "Could not locate eval_checkpoint_common.sh beside eval_global_step script. Submit from that directory or set EVAL_COMMON_SCRIPT=/path/to/eval_checkpoint_common.sh" >&2
  exit 1
fi
exec "$common_script"
