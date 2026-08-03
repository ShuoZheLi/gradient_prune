#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$script_dir/collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh" "$@"
