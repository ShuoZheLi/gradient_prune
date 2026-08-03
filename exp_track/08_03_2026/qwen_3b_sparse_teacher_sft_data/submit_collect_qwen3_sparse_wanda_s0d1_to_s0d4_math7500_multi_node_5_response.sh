#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for label in s0d1 s0d2 s0d3 s0d4; do
  sbatch "$script_dir/collect_qwen3_sparse_wanda_${label}_math7500_multi_node_5_response.sh" "$@"
done
