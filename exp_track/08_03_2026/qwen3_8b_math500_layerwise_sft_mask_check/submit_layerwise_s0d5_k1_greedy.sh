#!/bin/bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$script_dir/qwen3_8b_wanda_math500_layerwise_s0d5_k1_greedy.sh" "$@"
