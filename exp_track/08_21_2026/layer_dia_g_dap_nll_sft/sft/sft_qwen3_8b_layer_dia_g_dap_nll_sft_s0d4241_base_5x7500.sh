#!/bin/bash
#SBATCH --job-name=sft_qwen3_8b_layer_dia_s0d4241
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=sft_qwen3_8b_layer_dia_s0d4241_5x7500-%j.out
#SBATCH --error=sft_qwen3_8b_layer_dia_s0d4241_5x7500-%j.err

set -euo pipefail

export RUN_NAME="${RUN_NAME:-sft_qwen3_8b_layer_dia_g_dap_nll_sft_s0d4241_base_5x7500}"
export PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.4241}"
export SCORE_ROOT="${SCORE_ROOT:-/scratch/09576/shuozhe/gradient_prune/results/layer_dia_g_dap_nll_sft/qwen3_8b_layer_dia_g_dap_nll_sft_930296_20260822_231956/scores}"
export PRUNE_SCORE_KEY="${PRUNE_SCORE_KEY:-score}"

find_repo_root() {
  local start_dir="$1"
  local dir
  dir="$(CDPATH= cd -- "$start_dir" 2>/dev/null && pwd)" || return 1
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/exp_track/08_10_2026/wanda_sft/sft_qwen3_8b_wanda_s0d5_base_5x7500.sh" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

base_script="${SFT_BASE_SCRIPT:-}"
if [[ -z "$base_script" ]]; then
  repo_root="${REPO_ROOT:-}"
  if [[ -z "$repo_root" ]]; then
    for candidate in \
      "${SLURM_SUBMIT_DIR:-}" \
      "$PWD" \
      "$(dirname -- "${BASH_SOURCE[0]}")" \
      "/work2/09576/shuozhe/gradient_prune"; do
      [[ -z "$candidate" ]] && continue
      if repo_root="$(find_repo_root "$candidate")"; then
        break
      fi
    done
  fi
  if [[ -n "$repo_root" ]]; then
    base_script="$repo_root/exp_track/08_10_2026/wanda_sft/sft_qwen3_8b_wanda_s0d5_base_5x7500.sh"
  fi
fi

if [[ ! -f "$base_script" ]]; then
  echo "Could not locate the base Qwen3-8B sparse-SFT launcher. Set SFT_BASE_SCRIPT=/absolute/path/to/sft_qwen3_8b_wanda_s0d5_base_5x7500.sh." >&2
  exit 1
fi

exec "$base_script" "$@"
