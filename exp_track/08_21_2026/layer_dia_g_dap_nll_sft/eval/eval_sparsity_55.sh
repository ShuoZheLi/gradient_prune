#!/bin/bash
#SBATCH --job-name=eval_layer_dia_s55
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j_eval_layer_dia_s55.out
#SBATCH --error=slurm-%j_eval_layer_dia_s55.err

set -euo pipefail

export SPARSITY_PERCENT="55"
export PRUNING_SPARSITY="0.55"

common_script="${EVAL_COMMON_SCRIPT:-}"
if [[ -z "$common_script" ]]; then
  candidate_dirs=(
    "${SLURM_SUBMIT_DIR:-}"
    "${WORK_DIR:-}/exp_track/08_21_2026/layer_dia_g_dap_nll_sft/eval"
    "/work2/09576/shuozhe/gradient_prune/exp_track/08_21_2026/layer_dia_g_dap_nll_sft/eval"
    "$(dirname -- "${BASH_SOURCE[0]}")"
  )
  for candidate_dir in "${candidate_dirs[@]}"; do
    if [[ -n "$candidate_dir" && -f "$candidate_dir/eval_sparsity_common.sh" ]]; then
      common_script="$candidate_dir/eval_sparsity_common.sh"
      break
    fi
  done
fi
if [[ ! -f "$common_script" ]]; then
  echo "Could not locate eval_sparsity_common.sh. Set EVAL_COMMON_SCRIPT=/absolute/path/to/eval_sparsity_common.sh." >&2
  exit 1
fi
exec "$common_script"
