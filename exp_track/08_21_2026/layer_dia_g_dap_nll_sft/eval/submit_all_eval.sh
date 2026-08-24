#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for percent in 90 80 70 65 60 55 50 45 40 35 30 25 20 15 10 5 0; do
  sbatch "$script_dir/eval_sparsity_${percent}.sh"
done
