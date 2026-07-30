#!/bin/bash
#SBATCH --job-name=eval_qwen3_1d7b_global_step_500
#SBATCH --account=ASC26008
#SBATCH --partition=gh-dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j_eval_qwen3_1d7b_global_step_500.out
#SBATCH --error=slurm-%j_eval_qwen3_1d7b_global_step_500.err

set -euo pipefail

export CHECKPOINT_NAME="global_step_500"
export MODEL_PATH="/scratch/09576/shuozhe/verl_runs/sft_qwen3_1d7b_base_5x7500_874989/train_log/global_step_500"
export DATASET_PATH="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
export RUN_NAME="${RUN_NAME:-sft_qwen3_1d7b_base_global_step_500_math500_eval}"

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/eval_checkpoint_common.sh"
