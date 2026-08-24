#!/bin/bash
#SBATCH --job-name=eval_layer_dia_s35
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j_eval_layer_dia_s35.out
#SBATCH --error=slurm-%j_eval_layer_dia_s35.err

set -euo pipefail

export SPARSITY_PERCENT="35"
export PRUNING_SPARSITY="0.35"

common_script="${EVAL_COMMON_SCRIPT:-$(dirname -- "${BASH_SOURCE[0]}")/eval_sparsity_common.sh}"
if [[ ! -f "$common_script" ]]; then
  echo "Could not locate eval_sparsity_common.sh: $common_script" >&2
  exit 1
fi
exec "$common_script"
