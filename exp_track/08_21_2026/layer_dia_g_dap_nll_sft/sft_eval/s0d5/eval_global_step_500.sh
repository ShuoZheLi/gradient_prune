#!/bin/bash
#SBATCH --job-name=eval_layer_dia_s0d5_gs500
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j_eval_layer_dia_s0d5_gs500.out
#SBATCH --error=slurm-%j_eval_layer_dia_s0d5_gs500.err

set -euo pipefail

export CHECKPOINT_NAME="global_step_500"
export MODEL_PATH="/scratch/09576/shuozhe/verl_runs/sft_qwen3_8b_layer_dia_g_dap_nll_sft_s0d5_base_5x7500_934351/train_log/global_step_500"
export DATASET_PATH="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
export RUN_NAME="${RUN_NAME:-sft_qwen3_8b_layer_dia_g_dap_nll_sft_s0d5_global_step_500_math500_eval}"
export RESULTS_SUBDIR="${RESULTS_SUBDIR:-eval_distill/layer_dia_g_dap_nll_sft/s0d5/global_step_500}"

common_script="${EVAL_COMMON_SCRIPT:-}"
if [[ -z "$common_script" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_checkpoint_common.sh" ]]; then
    common_script="${SLURM_SUBMIT_DIR}/eval_checkpoint_common.sh"
  else
    common_script="$(dirname -- "${BASH_SOURCE[0]}")/eval_checkpoint_common.sh"
  fi
fi
if [[ ! -f "$common_script" ]]; then
  echo "Could not locate eval_checkpoint_common.sh. Submit with submit_all_eval.sh or set EVAL_COMMON_SCRIPT=/absolute/path/to/eval_checkpoint_common.sh." >&2
  exit 1
fi
exec "$common_script"
