#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluator_script="${EVAL_CHECKPOINT_COMMON_SCRIPT:-}"
if [[ -z "$evaluator_script" ]]; then
  repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
  if [[ -z "$repo_root" ]]; then
    repo_root="$(CDPATH= cd -- "$script_dir/../../../../.." && pwd)"
  fi
  evaluator_script="${repo_root}/exp_track/08_10_2026/eval_distill/sft_qwen3_8b_ffn_wanda_s0d7_lr_1e5_902622/eval_checkpoint_common.sh"
fi

if [[ ! -f "$evaluator_script" ]]; then
  echo "Could not locate the maintained eval_checkpoint_common.sh: $evaluator_script" >&2
  echo "Set EVAL_CHECKPOINT_COMMON_SCRIPT=/absolute/path/to/eval_checkpoint_common.sh." >&2
  exit 1
fi

exec "$evaluator_script"
