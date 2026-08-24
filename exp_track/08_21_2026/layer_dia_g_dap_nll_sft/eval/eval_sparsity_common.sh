#!/bin/bash
set -euo pipefail

# Materialize a temporary score-pruned HF checkpoint, then delegate Math500
# generation/scoring to the established eval_checkpoint_common.sh evaluator.

if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" && -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

find_repo_root() {
  local start_dir="$1"
  local dir
  dir="$(CDPATH= cd -- "$start_dir" 2>/dev/null && pwd)" || return 1
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" && -f "$dir/response_analysis/pruning.py" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
if [[ -z "$repo_root" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work2/09576/shuozhe/gradient_prune"; do
    [[ -z "$candidate" ]] && continue
    if repo_root="$(find_repo_root "$candidate")"; then
      break
    fi
  done
fi
if [[ -z "$repo_root" || ! -d "$repo_root" ]]; then
  echo "Could not locate gradient_prune repo. Set WORK_DIR=/path/to/gradient_prune." >&2
  exit 1
fi
cd "$repo_root"
export WORK_DIR="$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python3}"
scratch_root="${SCRATCH:-/scratch/09576/shuozhe}"
cache_root="${CACHE_ROOT:-${scratch_root}/${USER:-shuozhe}/gradient_prune_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

original_model_path="${BASE_MODEL_PATH:-${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B}}"
score_dir="${SCORE_DIR:-/scratch/09576/shuozhe/gradient_prune/results/layer_dia_g_dap_nll_sft/qwen3_8b_layer_dia_g_dap_nll_sft_930296_20260822_231956/scores}"
pruning_sparsity="${PRUNING_SPARSITY:?PRUNING_SPARSITY must be set by the wrapper}"
sparsity_percent="${SPARSITY_PERCENT:?SPARSITY_PERCENT must be set by the wrapper}"
prune_score_key="${PRUNE_SCORE_KEY:-}"
prune_granularity="${PRUNE_GRANULARITY:-rowwise}"
prune_ops="${PRUNE_OPS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
prune_dtype="${PRUNE_DTYPE:-bf16}"
prune_device="${PRUNE_DEVICE:-cuda:0}"
dry_run="${DRY_RUN:-0}"

"$python_bin" - "$pruning_sparsity" "$sparsity_percent" "$prune_granularity" <<'PY_VALIDATE_ARGS'
import sys

sparsity = float(sys.argv[1])
percent = int(sys.argv[2])
granularity = sys.argv[3]
if not 0.0 <= sparsity <= 1.0:
    raise SystemExit(f"PRUNING_SPARSITY must be in [0, 1], got {sparsity}")
if not 0 <= percent <= 100:
    raise SystemExit(f"SPARSITY_PERCENT must be in [0, 100], got {percent}")
if abs(sparsity * 100.0 - percent) > 1e-8:
    raise SystemExit(f"PRUNING_SPARSITY={sparsity} disagrees with SPARSITY_PERCENT={percent}")
if granularity not in {"rowwise", "layerwise"}:
    raise SystemExit(f"Unsupported PRUNE_GRANULARITY={granularity!r}")
PY_VALIDATE_ARGS
is_dense="$($python_bin - "$pruning_sparsity" <<'PY_IS_DENSE'
import sys

print("1" if float(sys.argv[1]) == 0.0 else "0")
PY_IS_DENSE
)"

checkpoint_name="${CHECKPOINT_NAME:-sparsity_${sparsity_percent}}"
export CHECKPOINT_NAME="$checkpoint_name"
export DATASET_PATH="${DATASET_PATH:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
export RUN_NAME="${RUN_NAME:-qwen3_8b_layer_dia_g_dap_nll_sft_sparsity_${sparsity_percent}_math500_eval}"
export RESULTS_SUBDIR="${RESULTS_SUBDIR:-eval_distill/layer_dia_g_dap_nll_sft/${checkpoint_name}}"
results_base="${RESULTS_BASE:-${RESULTS_ROOT:-${scratch_root}/gradient_prune/results}}"
export RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${results_base}/${RESULTS_SUBDIR}/runs/${RUN_ID}}"
score_config_file="${RUN_OUTPUT_DIR}/logs/score_pruning_config.env"

job_token="${SLURM_JOB_ID:-manual_$$}"
materialized_root="${MATERIALIZED_MODEL_ROOT:-${scratch_root}/gradient_prune/tmp/layer_dia_g_dap_nll_sft_eval/${job_token}}"
materialized_model_dir="${MATERIALIZED_MODEL_DIR:-${materialized_root}/${checkpoint_name}}"
keep_pruned_model="${KEEP_PRUNED_MODEL:-0}"

cleanup() {
  local status="$?"
  if [[ "$is_dense" != "1" && "$keep_pruned_model" != "1" ]]; then
    rm -rf "$materialized_model_dir" "${materialized_model_dir}.tmp.$$"
    rmdir "$materialized_root" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$is_dense" == "1" ]]; then
  export MODEL_PATH="$original_model_path"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-$original_model_path}"
  echo "[score-eval] sparsity=0; evaluating the dense base model"
else
  export MODEL_PATH="$materialized_model_dir"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-$materialized_model_dir}"

  if [[ "$dry_run" != "1" ]]; then
    for required_path in "$original_model_path" "$score_dir/metadata.json" "$score_dir/manifest.json"; do
      if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 2
      fi
    done

    IFS=',' read -r -a prune_ops_array <<< "$prune_ops"
    "$python_bin" - "$original_model_path" "$score_dir" "${prune_ops_array[@]}" <<'PY_VALIDATE_SCORES'
import json
import sys
from pathlib import Path

from transformers import AutoConfig

model_path = sys.argv[1]
score_dir = Path(sys.argv[2])
ops = [item.strip() for item in sys.argv[3:] if item.strip()]
metadata = json.loads((score_dir / "metadata.json").read_text(encoding="utf-8"))
manifest = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8"))
modules = metadata.get("modules")
if not isinstance(modules, dict):
    raise SystemExit("Score metadata must contain a modules mapping")
if metadata.get("score_key") != "score":
    raise SystemExit(f"Expected metadata score_key='score', got {metadata.get('score_key')!r}")
if metadata.get("source_manifest") != "manifest.json":
    raise SystemExit(f"Expected metadata source_manifest='manifest.json', got {metadata.get('source_manifest')!r}")
if manifest.get("method") != "layerwise_factorized_recoverability":
    raise SystemExit(f"Unexpected score method: {manifest.get('method')!r}")
if manifest.get("g_structure") != "diagonal":
    raise SystemExit(f"Expected diagonal G scores, got g_structure={manifest.get('g_structure')!r}")
if manifest.get("score_shard_format") != "pt":
    raise SystemExit(f"Expected pt score shards, got {manifest.get('score_shard_format')!r}")
manifest_modules = manifest.get("modules")
if not isinstance(manifest_modules, dict):
    raise SystemExit("Score manifest must contain a modules mapping")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
num_layers = int(getattr(config, "num_hidden_layers"))
op_paths = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
unknown = sorted(set(ops) - op_paths.keys())
if unknown:
    raise SystemExit(f"Unsupported PRUNE_OPS entries: {unknown}")
expected = {
    f"model.layers.{layer_index}.{op_paths[op]}"
    for layer_index in range(num_layers)
    for op in ops
}
missing = sorted(expected - modules.keys())
if missing:
    preview = ", ".join(missing[:5])
    raise SystemExit(f"Score metadata is missing {len(missing)} expected modules; first entries: {preview}")
extra = sorted(modules.keys() - expected)
if extra:
    preview = ", ".join(extra[:5])
    raise SystemExit(f"Score metadata contains {len(extra)} unexpected modules; first entries: {preview}")
manifest_expected = {f"{name}.weight" for name in expected}
manifest_missing = sorted(manifest_expected - manifest_modules.keys())
if manifest_missing:
    preview = ", ".join(manifest_missing[:5])
    raise SystemExit(f"Score manifest is missing {len(manifest_missing)} expected parameters; first entries: {preview}")
for name in sorted(expected):
    manifest_entry = manifest_modules[f"{name}.weight"]
    if not isinstance(manifest_entry, dict) or manifest_entry.get("shard") != modules[name]:
        raise SystemExit(f"Manifest/metadata shard mismatch for {name}")
missing_files = [name for name in sorted(expected) if not (score_dir / modules[name]).is_file()]
if missing_files:
    preview = ", ".join(missing_files[:5])
    raise SystemExit(f"Score files are missing for {len(missing_files)} modules; first entries: {preview}")
print(f"[score-eval] validated {len(expected)} score shards for {num_layers} layers")
PY_VALIDATE_SCORES

    rm -rf "${materialized_model_dir}.tmp.$$"
    mkdir -p "${materialized_model_dir}.tmp.$$"
    "$python_bin" - \
      "$original_model_path" \
      "$score_dir" \
      "$pruning_sparsity" \
      "$prune_score_key" \
      "$prune_granularity" \
      "$prune_ops" \
      "$prune_dtype" \
      "$prune_device" \
      "${materialized_model_dir}.tmp.$$" <<'PY_MATERIALIZE'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_analysis.pruning import apply_score_pruning

(
    model_path,
    score_dir,
    sparsity_text,
    score_key,
    granularity,
    prune_ops_text,
    dtype_name,
    device,
    output_dir,
) = sys.argv[1:]
sparsity = float(sparsity_text)
dtype = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}[dtype_name]
prune_ops = [item.strip() for item in prune_ops_text.split(",") if item.strip()]

print(f"[score-eval] loading base model from {model_path} on {device}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
pruning_info = apply_score_pruning(
    model,
    score_dir=score_dir,
    sparsity=sparsity,
    score_key=score_key or None,
    prune_ops=prune_ops,
    granularity=granularity,
)
output_path = Path(output_dir)
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
(output_path / "pruning_info.json").write_text(
    json.dumps(pruning_info, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(pruning_info, indent=2, sort_keys=True))
PY_MATERIALIZE
    rm -rf "$materialized_model_dir"
    mv "${materialized_model_dir}.tmp.$$" "$materialized_model_dir"
  fi
fi

if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$(dirname -- "$score_config_file")"
  cat > "$score_config_file" <<EOF_SCORE_CONFIG
BASE_MODEL_PATH=$original_model_path
EVAL_MODEL_PATH=$MODEL_PATH
SCORE_DIR=$score_dir
PRUNING_SPARSITY=$pruning_sparsity
SPARSITY_PERCENT=$sparsity_percent
PRUNE_SCORE_KEY=$prune_score_key
PRUNE_GRANULARITY=$prune_granularity
PRUNE_OPS=$prune_ops
PRUNE_DTYPE=$prune_dtype
PRUNE_DEVICE=$prune_device
KEEP_PRUNED_MODEL=$keep_pruned_model
EOF_SCORE_CONFIG
  if [[ -f "$materialized_model_dir/pruning_info.json" ]]; then
    cp "$materialized_model_dir/pruning_info.json" "${RUN_OUTPUT_DIR}/pruning_info.json"
  else
    cat > "${RUN_OUTPUT_DIR}/pruning_info.json" <<EOF_DENSE_INFO
{
  "enabled": false,
  "requested_sparsity": 0.0,
  "actual_sparsity": 0.0,
  "num_pruned_weights": 0,
  "num_total_prunable_weights": 0
}
EOF_DENSE_INFO
  fi
fi

export SCORE_DIR="$score_dir"
export PRUNING_SPARSITY="$pruning_sparsity"
export PRUNE_SCORE_KEY="$prune_score_key"
export PRUNE_GRANULARITY="$prune_granularity"
export PRUNE_OPS="$prune_ops"

delegate_script="${EVAL_DELEGATE_SCRIPT:-${repo_root}/exp_track/08_10_2026/eval_distill/sft_qwen3_8b_ffn_wanda_s0d7_lr_1e5_902622/eval_checkpoint_common.sh}"
if [[ ! -f "$delegate_script" ]]; then
  echo "Could not locate the shared Math500 evaluator: $delegate_script" >&2
  exit 3
fi

echo "[score-eval] base_model=$original_model_path"
echo "[score-eval] score_dir=$score_dir"
echo "[score-eval] sparsity=$pruning_sparsity (${sparsity_percent}%) granularity=$prune_granularity"
echo "[score-eval] eval_model=$MODEL_PATH"
"$delegate_script"
