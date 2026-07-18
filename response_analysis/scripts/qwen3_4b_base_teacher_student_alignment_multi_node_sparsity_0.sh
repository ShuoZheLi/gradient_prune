#!/bin/bash
#SBATCH --job-name=qwen3_4b_align_sparsity_0
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j_qwen3_4b_align_sparsity_0.out
#SBATCH --error=slurm-%j_qwen3_4b_align_sparsity_0.err

set -euo pipefail

# Multi-node teacher-student alignment launcher.
# Scores a teacher generations.jsonl under a student model, sharding JSONL rows
# across allocated Slurm nodes and merging per-example/aggregate outputs.

# -----------------------------
# Environment setup
# -----------------------------
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
    if [[ -f "$dir/pyproject.toml" && -d "$dir/response_analysis" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

repo_root="${WORK_DIR:-${REPO_ROOT:-}}"
if [[ -z "$repo_root" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname -- "${BASH_SOURCE[0]}")" "/work2/09576/shuozhe/gradient_prune" "/data/shuozhe/gradient_prune"; do
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
export PYTHONPATH="$repo_root:$repo_root/src:${PYTHONPATH:-}"

SCRATCH_ROOT="${SCRATCH:-/scratch/09576/shuozhe}"
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
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TIKTOKEN_ENCODINGS_BASE"

python_bin="${PYTHON_BIN:-python3}"

# -----------------------------
# Runtime config
# -----------------------------
RUN_NAME="${RUN_NAME:-qwen3_4b_base_teacher_student_alignment_sparsity_0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}}"
RESULTS_BASE="${RESULTS_BASE:-${RESULTS_ROOT:-$SCRATCH_ROOT/gradient_prune/results}}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-response_analysis/${RUN_NAME}}"
RUN_ROOT="${RUN_OUTPUT_DIR:-$RESULTS_BASE/$RESULTS_SUBDIR/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
CONFIG_FILE="${CONFIG_FILE:-$LOG_DIR/config.env}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2> >(tee -a "$LOG_DIR/run.err" >&2)

MODEL_PATH="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-4B-Base}"
MODEL_ID="${MODEL_ID:-qwen3_4b_base_sparsity_0}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0}"
INPUT_JSONL="${INPUT_JSONL:-/scratch/09576/shuozhe/gradient_prune/results/response_analysis/qwen3_8b_wanda_response_analysis/runs/qwen3_8b_wanda_response_analysis_839833_20260717_160149/combined/generations.jsonl}"
BACKEND="${BACKEND:-vllm}"
DTYPE="${DTYPE:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-${MAX_EXAMPLES:--1}}"
SHARD_COUNT="${SHARD_COUNT:-auto}"

SHARD_ROOT="$RUN_ROOT/shards"
COMBINED_DIR="$RUN_ROOT/combined"
mkdir -p "$SHARD_ROOT" "$COMBINED_DIR"

if [[ "$BACKEND" != "hf" && "$BACKEND" != "vllm" ]]; then
  echo "BACKEND must be hf or vllm, got: $BACKEND" >&2
  exit 1
fi
if [[ "$PRUNING_SPARSITY" != "0" && "$PRUNING_SPARSITY" != "0.0" ]]; then
  echo "This launcher is labeled for sparsity 0; got PRUNING_SPARSITY=$PRUNING_SPARSITY" >&2
  exit 1
fi
if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Input JSONL does not exist: $INPUT_JSONL" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

mapfile -t NODE_NAMES < <(if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then scontrol show hostnames "$SLURM_JOB_NODELIST"; else hostname; fi)
NODE_COUNT="${#NODE_NAMES[@]}"
if [[ "$NODE_COUNT" -lt 1 ]]; then
  NODE_NAMES=("$(hostname)")
  NODE_COUNT=1
fi

if [[ "$SHARD_COUNT" == "auto" ]]; then
  SHARD_COUNT="$NODE_COUNT"
fi
if [[ "$SHARD_COUNT" -lt 1 ]]; then
  echo "SHARD_COUNT must be >= 1, got: $SHARD_COUNT" >&2
  exit 1
fi
if [[ "$SHARD_COUNT" -gt "$NODE_COUNT" ]]; then
  echo "SHARD_COUNT=$SHARD_COUNT exceeds allocated node count=$NODE_COUNT" >&2
  exit 1
fi

cat > "$CONFIG_FILE" <<CONFIG
RUN_NAME=$RUN_NAME
RUN_ID=$RUN_ID
RUN_ROOT=$RUN_ROOT
MODEL_PATH=$MODEL_PATH
MODEL_ID=$MODEL_ID
PRUNING_SPARSITY=$PRUNING_SPARSITY
INPUT_JSONL=$INPUT_JSONL
BACKEND=$BACKEND
DTYPE=$DTYPE
BATCH_SIZE=$BATCH_SIZE
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
ENFORCE_EAGER=$ENFORCE_EAGER
MAX_SEQUENCE_LENGTH=$MAX_SEQUENCE_LENGTH
TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE
DEBUG_SUBSET=$DEBUG_SUBSET
SHARD_COUNT=$SHARD_COUNT
NODE_NAMES=${NODE_NAMES[*]}
CONFIG

printf '=== Teacher-student alignment config ===\n'
cat "$CONFIG_FILE"
printf '========================================\n'

split_input() {
  "$python_bin" - "$INPUT_JSONL" "$SHARD_ROOT" "$SHARD_COUNT" "$DEBUG_SUBSET" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
shard_count = int(sys.argv[3])
debug_subset = int(sys.argv[4])

lines = input_path.read_text(encoding="utf-8").splitlines()
if debug_subset >= 0:
    lines = lines[:debug_subset]
if not lines:
    raise SystemExit(f"No input rows to score after DEBUG_SUBSET={debug_subset}")
shard_count = min(shard_count, len(lines))
shard_root.mkdir(parents=True, exist_ok=True)
for old in shard_root.glob("input_shard_*.jsonl"):
    old.unlink()
for shard_idx in range(shard_count):
    shard_lines = lines[shard_idx::shard_count]
    shard_path = shard_root / f"input_shard_{shard_idx:03d}.jsonl"
    shard_path.write_text("\n".join(shard_lines) + ("\n" if shard_lines else ""), encoding="utf-8")
    print(f"{shard_path}\t{len(shard_lines)}")
PY
}

merge_outputs() {
  local per_example_output="$COMBINED_DIR/per_example_alignment.jsonl"
  local aggregate_output="$COMBINED_DIR/aggregate_alignment.json"
  : > "$per_example_output"
  for shard_dir in "$SHARD_ROOT"/output_shard_*; do
    [[ -f "$shard_dir/per_example_alignment.jsonl" ]] || continue
    cat "$shard_dir/per_example_alignment.jsonl" >> "$per_example_output"
  done
  "$python_bin" - "$per_example_output" "$aggregate_output" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from response_analysis.compute_teacher_student_alignment import aggregate_alignment
from response_analysis.io_utils import read_jsonl

per_example = Path(sys.argv[1])
aggregate_path = Path(sys.argv[2])
rows = read_jsonl(per_example)
if not rows:
    raise SystemExit(f"No per-example rows found in {per_example}")
aggregate_path.write_text(json.dumps(aggregate_alignment(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Merged {len(rows)} rows into {per_example}")
print(f"Wrote aggregate metrics to {aggregate_path}")
PY
}

build_alignment_args() {
  local shard_input="$1"
  local shard_output="$2"
  local args=(
    -m response_analysis.compute_teacher_student_alignment
    --backend "$BACKEND"
    --input "$shard_input"
    --model_path "$MODEL_PATH"
    --model_id "$MODEL_ID"
    --output_dir "$shard_output"
    --dtype "$DTYPE"
    --batch_size "$BATCH_SIZE"
  )
  if [[ "$BACKEND" == "hf" ]]; then
    args+=(--device "${DEVICE:-cuda:0}")
  else
    args+=(--tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION")
    if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then args+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN"); fi
    if [[ "$ENFORCE_EAGER" == "1" ]]; then args+=(--enforce_eager); else args+=(--no-enforce_eager); fi
  fi
  if [[ -n "$MAX_SEQUENCE_LENGTH" ]]; then args+=(--max_sequence_length "$MAX_SEQUENCE_LENGTH" --skip_overlength); fi
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then args+=(--trust_remote_code); fi
  printf '%q ' "$python_bin" "${args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  split_input | tee "$LOG_DIR/shards.tsv"
  EFFECTIVE_SHARD_COUNT="$(wc -l < "$LOG_DIR/shards.tsv" | tr -d ' ')"
  first_shard="$SHARD_ROOT/input_shard_000.jsonl"
  first_output="$SHARD_ROOT/output_shard_000"
  echo "Requested SHARD_COUNT=$SHARD_COUNT; effective shard count after splitting=$EFFECTIVE_SHARD_COUNT"
  echo "DRY_RUN=1; first shard command:"
  build_alignment_args "$first_shard" "$first_output"
  echo
  exit 0
fi

split_input | tee "$LOG_DIR/shards.tsv"
EFFECTIVE_SHARD_COUNT="$(wc -l < "$LOG_DIR/shards.tsv" | tr -d ' ')"
echo "Requested SHARD_COUNT=$SHARD_COUNT; effective shard count after splitting=$EFFECTIVE_SHARD_COUNT"

pids=()
for shard_idx in $(seq 0 $((EFFECTIVE_SHARD_COUNT - 1))); do
  node="${NODE_NAMES[$shard_idx]}"
  shard_input="$SHARD_ROOT/input_shard_$(printf '%03d' "$shard_idx").jsonl"
  shard_output="$SHARD_ROOT/output_shard_$(printf '%03d' "$shard_idx")"
  shard_log="$LOG_DIR/alignment_shard_$(printf '%03d' "$shard_idx").log"
  shard_err="$LOG_DIR/alignment_shard_$(printf '%03d' "$shard_idx").err"
  mkdir -p "$shard_output"
  cmd="cd $(printf '%q' "$repo_root") && export PYTHONPATH=$(printf '%q' "$PYTHONPATH") && $(build_alignment_args "$shard_input" "$shard_output")"
  echo "Launching shard $shard_idx on $node: $cmd"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --exclusive -N1 -n1 -w "$node" bash -lc "$cmd" >"$shard_log" 2>"$shard_err" &
  else
    bash -lc "$cmd" >"$shard_log" 2>"$shard_err" &
  fi
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one alignment shard failed. Check $LOG_DIR/alignment_shard_*.err" >&2
  exit 1
fi

merge_outputs

echo "Alignment run complete."
echo "Per-example: $COMBINED_DIR/per_example_alignment.jsonl"
echo "Aggregate:   $COMBINED_DIR/aggregate_alignment.json"
