#!/bin/bash
#SBATCH --job-name=collect_qwen3_sparse_s0d4_n5
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j_collect_qwen3_sparse_s0d4_n5.out
#SBATCH --error=slurm-%j_collect_qwen3_sparse_s0d4_n5.err

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_script="${script_dir}/collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh"

export PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.4}"
export SPARSITY_LABEL="${SPARSITY_LABEL:-s0d4}"
export RUN_NAME="${RUN_NAME:-collect_qwen3_8b_wanda_s0d4_${PRUNE_GRANULARITY:-layerwise}_math7500_5_response}"

exec "$base_script" "$@"
