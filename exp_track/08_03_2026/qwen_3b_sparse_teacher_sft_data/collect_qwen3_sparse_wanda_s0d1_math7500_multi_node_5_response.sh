#!/bin/bash
#SBATCH --job-name=collect_qwen3_sparse_s0d1_n5
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j_collect_qwen3_sparse_s0d1_n5.out
#SBATCH --error=slurm-%j_collect_qwen3_sparse_s0d1_n5.err

set -euo pipefail

# Build a Qwen3 WANDA-pruned teacher checkpoint, then collect Math7500
# correct 5-response SFT data using the existing 08/01 collector.
#
# NOTE: The default score root below is the Qwen3-8B score directory used by
# the 08/01 SFT sparse-update scripts, so the default model is Qwen3-8B.
# If you want Qwen3-4B/3B, you must also provide matching same-shape scores.

if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

find_repo_root() {
  local start_dir="$1"
  local dir
  dir="$(CDPATH= cd -- "$start_dir" 2>/dev/null && pwd)" || return 1
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" && -d "$dir/create_calibration_dataset" ]]; then
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
  echo "Could not locate gradient_prune repo. Set WORK_DIR=/path/to/gradient_prune when submitting." >&2
  exit 1
fi
cd "$repo_root"
export PYTHONPATH="${repo_root}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python3}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/tmp/${USER:-shuozhe}}}"
cache_root="${CACHE_ROOT:-${SCRATCH_ROOT}/gradient_prune_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${cache_root}/tiktoken}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-8B}}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.1}"
PRUNE_GRANULARITY="${PRUNE_GRANULARITY:-layerwise}"
PRUNE_SCORE_KEY="${PRUNE_SCORE_KEY:-}"
SCORE_ROOT="${SCORE_ROOT:-${SCRATCH:-/scratch/09576/shuozhe}/gradient_prune/results/qwen3_8b_wanda_math7500/scores}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SPARSITY_LABEL="${SPARSITY_LABEL:-s0d1}"
GRANULARITY_LABEL="${GRANULARITY_LABEL:-${PRUNE_GRANULARITY}}"
TEACHER_ID="${TEACHER_ID:-qwen3_8b_wanda_${SPARSITY_LABEL}_${GRANULARITY_LABEL}}"
PRUNED_TEACHER_DIR="${PRUNED_TEACHER_DIR:-${SCRATCH_ROOT}/gradient_prune/pruned_teachers/${TEACHER_ID}}"

COLLECT_SCRIPT="${COLLECT_SCRIPT:-${repo_root}/exp_track/08_01_2026/collect_sft_train_data/collect_deepseek_r1_distill_llama_8b_correct_math7500_multi_node_5_response.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/saved_calibration_dataset/${TEACHER_ID}_math7500_correct_5_response}"
RUN_NAME="${RUN_NAME:-collect_${TEACHER_ID}_math7500_5_response}"

if [[ ! -f "$COLLECT_SCRIPT" ]]; then
  echo "Missing base collector: $COLLECT_SCRIPT" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -d "$BASE_MODEL_PATH" ]]; then
  echo "Base model path does not exist: $BASE_MODEL_PATH" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ! -f "$SCORE_ROOT/metadata.json" ]]; then
  echo "Score root does not contain metadata.json: $SCORE_ROOT" >&2
  exit 3
fi

has_hf_checkpoint() {
  local path="$1"
  [[ -f "$path/config.json" ]] && { compgen -G "$path/*.safetensors" >/dev/null || compgen -G "$path/pytorch_model*.bin" >/dev/null; }
}

build_pruned_teacher() {
  if has_hf_checkpoint "$PRUNED_TEACHER_DIR"; then
    echo "[sparse-teacher] Reusing pruned teacher: $PRUNED_TEACHER_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; would build pruned teacher at: $PRUNED_TEACHER_DIR"
    return 0
  fi

  mkdir -p "$PRUNED_TEACHER_DIR"
  echo "[sparse-teacher] Building pruned teacher"
  echo "  base_model: $BASE_MODEL_PATH"
  echo "  score_root: $SCORE_ROOT"
  echo "  sparsity: $PRUNING_SPARSITY"
  echo "  granularity: $PRUNE_GRANULARITY"
  echo "  output: $PRUNED_TEACHER_DIR"

  "$python_bin" - \
    --base_model_path "$BASE_MODEL_PATH" \
    --score_root "$SCORE_ROOT" \
    --output_dir "$PRUNED_TEACHER_DIR" \
    --sparsity "$PRUNING_SPARSITY" \
    --granularity "$PRUNE_GRANULARITY" \
    --score_key "$PRUNE_SCORE_KEY" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --trust_remote_code "$TRUST_REMOTE_CODE" <<'PY'
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_analysis.pruning import apply_score_pruning


def resolve_dtype(name: str):
    name = str(name).lower()
    if name in {"auto", "none"}:
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", required=True)
parser.add_argument("--score_root", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--sparsity", type=float, required=True)
parser.add_argument("--granularity", choices=("rowwise", "layerwise"), required=True)
parser.add_argument("--score_key", default="")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--trust_remote_code", default="0")
args = parser.parse_args()

trust_remote_code = args.trust_remote_code.lower() in {"1", "true", "yes", "on"}
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=trust_remote_code)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=resolve_dtype(args.dtype),
    trust_remote_code=trust_remote_code,
    device_map=None,
).to(args.device)
pruning_info = apply_score_pruning(
    model,
    score_dir=args.score_root,
    sparsity=args.sparsity,
    score_key=args.score_key or None,
    granularity=args.granularity,
)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
(output_dir / "pruning_info.json").write_text(json.dumps(pruning_info, indent=2, default=str) + "\n", encoding="utf-8")
print(json.dumps(pruning_info, indent=2, default=str))
PY
}

build_pruned_teacher

export MODEL_PATH="$PRUNED_TEACHER_DIR"
export RUN_NAME
export OUTPUT_DIR
export SKIP_MERGE="${SKIP_MERGE:-1}"
export TRUST_REMOTE_CODE
export NUM_RESPONSES_PER_PROMPT="${NUM_RESPONSES_PER_PROMPT:-5}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-0}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"

export RAW_JSONL="${RAW_JSONL:-${OUTPUT_DIR}/raw_actor_responses.jsonl}"
export ALL_TRAJECTORIES_JSONL="${ALL_TRAJECTORIES_JSONL:-${OUTPUT_DIR}/all_actor_trajectories.jsonl}"
export ALL_TRAJECTORIES_PARQUET="${ALL_TRAJECTORIES_PARQUET:-${OUTPUT_DIR}/all_actor_trajectories.parquet}"
export CORRECT_JSONL="${CORRECT_JSONL:-${OUTPUT_DIR}/correct_actor_responses.jsonl}"
export CALIB_PARQUET="${CALIB_PARQUET:-${OUTPUT_DIR}/${TEACHER_ID}_math7500_correct_5_response.parquet}"
export METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/metrics.json}"

echo "[sparse-teacher] Collecting sparse-teacher SFT data"
echo "  collector: $COLLECT_SCRIPT"
echo "  teacher_model: $MODEL_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  calib_parquet: $CALIB_PARQUET"

exec "$COLLECT_SCRIPT"
