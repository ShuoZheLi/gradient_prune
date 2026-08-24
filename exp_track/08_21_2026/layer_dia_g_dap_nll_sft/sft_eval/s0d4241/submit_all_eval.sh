#!/bin/bash
set -euo pipefail
sleep 4h
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
common_script="${script_dir}/eval_checkpoint_common.sh"

for step in 0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800; do
  sbatch --export="ALL,EVAL_COMMON_SCRIPT=${common_script}" "${script_dir}/eval_global_step_${step}.sh"
done
