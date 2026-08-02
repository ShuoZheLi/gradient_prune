#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for step in 0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850; do
  script="$script_dir/eval_global_step_${step}.sh"
  echo "Submitting $script"
  sbatch "$script"
done
