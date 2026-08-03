#!/bin/bash
#SBATCH --job-name=qwen3_8b_math500_layer_s0d5_k1
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=1:30:00
#SBATCH --output=slurm-%j_qwen3_8b_math500_layer_s0d5_k1.out
#SBATCH --error=slurm-%j_qwen3_8b_math500_layer_s0d5_k1.err

set -euo pipefail

# Purpose: reproduce the 07/21 Qwen3-8B WANDA Math500 response-analysis run,
# but with layerwise/global pruning and greedy pass@1 settings to mimic the
# sparse-update mask used by the SFT step-0 checkpoint.

export PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.5}"
export PRUNE_GRANULARITY="${PRUNE_GRANULARITY:-layerwise}"
export K="${K:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-0}"
export RUN_DENSE="${RUN_DENSE:-0}"
export RUN_PRUNED="${RUN_PRUNED:-1}"

# Keep only the pieces needed for pass@1 unless overridden at submission time.
export RUN_GENERATION="${RUN_GENERATION:-1}"
export RUN_ON_POLICY_ENTROPY="${RUN_ON_POLICY_ENTROPY:-0}"
export RUN_FIXED_PREFIX_ENTROPY="${RUN_FIXED_PREFIX_ENTROPY:-0}"
export RUN_SURFACE_DIVERSITY="${RUN_SURFACE_DIVERSITY:-1}"
export RUN_SEMANTIC_JUDGE="${RUN_SEMANTIC_JUDGE:-0}"
export RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
export NO_API="${NO_API:-1}"

export RUN_NAME="${RUN_NAME:-qwen3_8b_wanda_math500_layerwise_s0d5_k1_greedy}"
export RESULTS_SUBDIR="${RESULTS_SUBDIR:-response_analysis/${RUN_NAME}}"
export PRUNED_MODEL_ID="${PRUNED_MODEL_ID:-qwen3_8b_wanda_layerwise_s${PRUNING_SPARSITY}}"

base_script="/data/shuozhe/gradient_prune/exp_track/07_21_2026/qwen3_8b_math500/qwen3_8b_wanda_math500_response_analysis_multi_node_sparsity_0d5.sh"
if [[ ! -f "$base_script" ]]; then
  echo "Missing base response-analysis script: $base_script" >&2
  exit 1
fi

printf '[layerwise-check] base_script=%s\n' "$base_script"
printf '[layerwise-check] PRUNING_SPARSITY=%s PRUNE_GRANULARITY=%s K=%s TEMPERATURE=%s TOP_P=%s TOP_K=%s\n' \
  "$PRUNING_SPARSITY" "$PRUNE_GRANULARITY" "$K" "$TEMPERATURE" "$TOP_P" "$TOP_K"
printf '[layerwise-check] RUN_NAME=%s RESULTS_SUBDIR=%s\n' "$RUN_NAME" "$RESULTS_SUBDIR"

exec "$base_script"
