#!/bin/bash
#SBATCH --job-name=sft_qwen3_8b_wanda_s0d5
#SBATCH --account=ASC26008
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=05:30:00
#SBATCH --output=sft_qwen3_8b_wanda_s0d5_5x7500-%j.out
#SBATCH --error=sft_qwen3_8b_wanda_s0d5_5x7500-%j.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
else
  echo "VENV activate script not found at ${VENV}/bin/activate; using current Python environment."
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/.cache/uv}"
HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${SCRATCH}/data/embeddings}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH}/.cache/torch_extensions}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE" "$TORCH_EXTENSIONS_DIR"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export TORCH_EXTENSIONS_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Avoid inheriting incompatible launch variables from other runs.
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID || true

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity and paths
# -----------------------------
export WANDB_PROJECT="prune_for_post_train"
WANDB_PROJECT="prune_for_post_train"
RUN_NAME="${RUN_NAME:-sft_qwen3_8b_wanda_s0d5_base_5x7500}"
REAL_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_NAME}_${REAL_SLURM_JOB_ID}"

HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE:-}"
HF_MODULES_CACHE_ROOT="${HF_MODULES_CACHE:-}"
export HF_DATASETS_CACHE_ROOT
export HF_MODULES_CACHE_ROOT

WORK_DIR="${WORK_DIR:-/work/09576/shuozhe/gradient_prune/verl}"
MODEL_INIT_CKPT="${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-/work/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math7500_correct_5_response/qwen3-8b-instruct_math7500_correct_5_response.parquet}"
VAL_FILE="${VAL_FILE:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
MATH_500_EVAL_FILE="${MATH_500_EVAL_FILE:-${VAL_FILE}}"
AIME_24_25_26_EVAL_FILE="${AIME_24_25_26_EVAL_FILE:-/work/09576/shuozhe/saved_dataset/MetaMathQA-math-500/aime_24_25_26.parquet}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.5}"
SCORE_ROOT="${SCORE_ROOT:-${SCRATCH}/gradient_prune/results/qwen3_8b_wanda_math7500/scores}"
PRUNE_SCORE_KEY="${PRUNE_SCORE_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/verl_runs}"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_LOG_DIR="${RUN_DIR}/train_log"
SPARSE_MASK_PATH="${SPARSE_MASK_PATH:-${RUN_DIR}/sparse_update_masks/wanda_s${PRUNING_SPARSITY}.pt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/work/09576/shuozhe/gradient_prune/verl/train_log_archive}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
TRAIN_STDOUT_LOG="${TRAIN_LOG_DIR}/job_${RUN_ID}.txt"

mkdir -p "$LOG_DIR" "$TRAIN_LOG_DIR" "$ARCHIVE_ROOT"

# -----------------------------
# SFT training defaults
# -----------------------------
# Defaults target Qwen3-8B WANDA-pruned sparse SFT on the full 5-response x 7500 math dataset.
train_batch_size=${train_batch_size:-128}
micro_batch_size_per_gpu=${micro_batch_size_per_gpu:-2}
max_length=${max_length:-18432}
max_token_len_per_gpu=${max_token_len_per_gpu:-36864}
truncation=${truncation:-error}
lr=${lr:-5e-6}
total_epochs=${total_epochs:-5}
save_freq=${save_freq:-50}
save_initial_checkpoint=${save_initial_checkpoint:-True}
num_workers=${num_workers:-8}

# Sparse SFT: WANDA top-score entries are kept/trainable; all pruned entries are zeroed and stay zero.
sparse_update_enabled=${sparse_update_enabled:-true}
sparse_zero_frozen_params=${sparse_zero_frozen_params:-true}
sparse_verify_frozen_weights=${sparse_verify_frozen_weights:-true}
sparse_verification_interval=${sparse_verification_interval:-10}

# eval_method: loss, generation_reward, or both.
eval_method=${eval_method:-generation_reward}
eval_before_train=${eval_before_train:-False}
eval_freq=${eval_freq:--1}
loss_eval_freq=${loss_eval_freq:--1}
generation_eval_freq=${generation_eval_freq:--1}

# loss_eval_files must be SFT messages format; generation_eval_files must be PPO prompt+reward_model format.
loss_eval_files=${loss_eval_files:-__TRAIN_FILE__}
if [[ -z "${generation_eval_files+x}" ]]; then
  generation_eval_files="${MATH_500_EVAL_FILE} ${AIME_24_25_26_EVAL_FILE}"
  generation_eval_names=${generation_eval_names:-"math_500 aime_24_25_26"}
else
  generation_eval_names=${generation_eval_names:-}
fi
val_max_samples=${val_max_samples:--1}
train_max_samples=${train_max_samples:--1}
trainer_logger=${trainer_logger:-'["console","wandb"]'}
resume_mode=${resume_mode:-auto}
resume_from_path=${resume_from_path:-null}

# Qwen3 thinking mode for generation eval. Set generation_eval_enable_thinking=False to disable.
generation_eval_enable_thinking=${generation_eval_enable_thinking:-True}
generation_eval_batch_size=${generation_eval_batch_size:-16}
generation_max_new_tokens=${generation_max_new_tokens:-18432}
generation_do_sample=${generation_do_sample:-False}
generation_temperature=${generation_temperature:-0.0}
generation_top_p=${generation_top_p:-1.0}
generation_top_k=${generation_top_k:-null}
generation_num_samples=${generation_num_samples:-1}
generation_dtype=${generation_dtype:-null}

generation_backend=${generation_backend:-vllm}
generation_vllm_gpu_memory_utilization=${generation_vllm_gpu_memory_utilization:-0.4}
generation_vllm_host_ip=${generation_vllm_host_ip:-127.0.0.1}
generation_vllm_enforce_eager=${generation_vllm_enforce_eager:-True}
generation_vllm_sync_weights=${generation_vllm_sync_weights:-True}
generation_vllm_enable_multiprocessing=${generation_vllm_enable_multiprocessing:-False}

# -----------------------------
# Multi-node torchrun config
# -----------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
RDZV_PORT="${RDZV_PORT:-29500}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  nodes_array=($nodes)
  head_node="${nodes_array[0]}"
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
else
  head_node="$(hostname)"
  head_node_ip="$(hostname --ip-address 2>/dev/null || hostname -I 2>/dev/null || echo 127.0.0.1)"
fi

resolved_head_node_ip=""
for candidate_ip in $head_node_ip; do
  if [[ "$candidate_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    resolved_head_node_ip="$candidate_ip"
    break
  fi
done
if [[ -z "$resolved_head_node_ip" ]]; then
  for candidate_ip in $head_node_ip; do
    resolved_head_node_ip="$candidate_ip"
    break
  done
fi
if [[ -z "$resolved_head_node_ip" ]]; then
  echo "Failed to resolve a usable IP address for torchrun head node $head_node." >&2
  exit 1
fi
MASTER_ADDR="$resolved_head_node_ip"
MASTER_PORT="$RDZV_PORT"
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# -----------------------------
# Helpers
# -----------------------------
MODEL_PATH_RESOLVER="${WORK_DIR}/tools/resolve_model_init_path.py"

resolve_model_init_path() {
  local raw_path="$1"
  local role="$2"

  if [[ ! -f "$MODEL_PATH_RESOLVER" ]]; then
    echo "Missing model path resolver: $MODEL_PATH_RESOLVER" >&2
    return 1
  fi

  python3 "$MODEL_PATH_RESOLVER" \
    --path "$raw_path" \
    --role "$role" \
    --log-dir "$LOG_DIR"
}

describe_path() {
  local label="$1"
  local path="$2"

  echo "$label: $path"
  if [[ "$path" == "null" || -z "$path" ]]; then
    echo "  disabled/null"
  elif [[ "$path" == *"://"* ]]; then
    echo "  non-local URI path"
  elif [[ -d "$path" || -f "$path" ]]; then
    ls -ld "$path"
  elif [[ "$path" = /* || "$path" == ./* || "$path" == ../* || "$path" == ~* ]]; then
    echo "  local path not found"
  else
    echo "  passthrough model identifier"
  fi
}

normalize_hydra_list_if_many() {
  python3 - "$1" <<'PY'
import json
import shlex
import sys

raw = sys.argv[1].strip()
if raw.startswith(("[", "{")) or raw in {"", "null"}:
    print(raw)
else:
    items = shlex.split(raw)
    print(json.dumps(items, separators=(",", ":")) if len(items) > 1 else raw)
PY
}

list_eval_paths() {
  python3 - "$1" <<'PY'
import json
import shlex
import sys

raw = sys.argv[1].strip()
if not raw or raw == "null":
    raise SystemExit
if raw.startswith("["):
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        items = [item.strip().strip('"\'') for item in raw[1:-1].split(",") if item.strip()]
elif raw.startswith("{"):
    body = raw[1:-1]
    items = []
    for pair in body.split(","):
        if ":" in pair:
            items.append(pair.split(":", 1)[1].strip().strip('"\''))
else:
    items = shlex.split(raw)
for item in items:
    print(item)
PY
}

print_planned_checkpoint_dirs() {
  python3 - \
    "$TRAIN_FILE" \
    "$train_max_samples" \
    "$train_batch_size" \
    "$total_epochs" \
    "$save_freq" \
    "$save_initial_checkpoint" \
    "$TRAIN_LOG_DIR" \
    "$NNODES" \
    "$GPUS_PER_NODE" <<'PY'
import os
import sys

(
    train_file,
    train_max_samples,
    train_batch_size,
    total_epochs,
    save_freq,
    save_initial_checkpoint,
    train_log_dir,
    nnodes,
    gpus_per_node,
) = sys.argv[1:]


def parse_int(name, value):
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"Unable to parse {name}={value!r} while planning checkpoint paths.") from exc


def truthy(value):
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


try:
    import pyarrow.parquet as pq

    num_samples = pq.ParquetFile(train_file).metadata.num_rows
except Exception as exc:
    print(f"Unable to plan checkpoint paths because train parquet row count failed: {exc}")
    raise SystemExit(0)

max_samples = parse_int("train_max_samples", train_max_samples)
if max_samples > 0:
    num_samples = min(num_samples, max_samples)

global_batch_size = parse_int("train_batch_size", train_batch_size)
epochs = parse_int("total_epochs", total_epochs)
dp_size = parse_int("NNODES", nnodes) * parse_int("GPUS_PER_NODE", gpus_per_node)

if dp_size <= 0 or global_batch_size <= 0 or epochs <= 0:
    print("Unable to plan checkpoint paths because dp size, batch size, or epochs is non-positive.")
    raise SystemExit(0)

batch_size_per_dp = global_batch_size // dp_size
if batch_size_per_dp <= 0:
    print("Unable to plan checkpoint paths because train_batch_size is smaller than data-parallel size.")
    raise SystemExit(0)

# Match SFTTrainer: DistributedSampler(drop_last=True) then StatefulDataLoader(drop_last=True).
samples_per_dp = num_samples // dp_size
steps_per_epoch = samples_per_dp // batch_size_per_dp
total_steps = steps_per_epoch * epochs

planned_steps = []
if total_steps > 0 and truthy(save_initial_checkpoint):
    planned_steps.append(0)

if save_freq == "after_each_epoch":
    freq = steps_per_epoch
else:
    freq = parse_int("save_freq", save_freq)

if total_steps > 0 and freq > 0:
    planned_steps.extend(range(freq, total_steps + 1, freq))

if total_steps > 0:
    planned_steps.append(total_steps)

seen = set()
unique_steps = []
for step in planned_steps:
    if step not in seen:
        seen.add(step)
        unique_steps.append(step)

print("Planned checkpoint directories, one per line:")
for step in unique_steps:
    print(os.path.join(train_log_dir, f"global_step_{step}"))
PY
}

build_sparse_update_mask() {
  if [[ "$sparse_update_enabled" != "true" && "$sparse_update_enabled" != "True" ]]; then
    echo "Sparse update disabled; skipping WANDA mask build."
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; skipping WANDA mask build. Planned mask path: $SPARSE_MASK_PATH"
    return 0
  fi

  if [[ -f "$SPARSE_MASK_PATH" ]]; then
    echo "Reusing sparse-update mask: $SPARSE_MASK_PATH"
    return 0
  fi

  if [[ ! -f "$SCORE_ROOT/metadata.json" ]]; then
    echo "WANDA score root does not contain metadata.json: $SCORE_ROOT" >&2
    echo "Set SCORE_ROOT=/path/to/scores from the 07_21_2026 qwen3_8b_math500 WANDA script." >&2
    exit 3
  fi

  mkdir -p "$(dirname "$SPARSE_MASK_PATH")"
  echo "Building sparse-update WANDA mask"
  echo "  model: $MODEL_PATH"
  echo "  score root: $SCORE_ROOT"
  echo "  sparsity: $PRUNING_SPARSITY"
  echo "  output: $SPARSE_MASK_PATH"
  python3 "$WORK_DIR/tools/build_sparse_update_mask.py" \
    --model_name_or_path "$MODEL_PATH" \
    --output_path "$SPARSE_MASK_PATH" \
    --wanda_score_dir "$SCORE_ROOT" \
    --sparsity "$PRUNING_SPARSITY" \
    --mode wanda_top
}

sync_to_work() {
  echo "Syncing lightweight logs/metadata back to archive; checkpoints stay in RUN_DIR."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a \
    --exclude='**/global_step_*' \
    --exclude='**/checkpoints/**' \
    --exclude='**/model_world_size_*_rank_*.pt' \
    --exclude='**/optim_world_size_*_rank_*.pt' \
    --exclude='**/extra_state_world_size_*_rank_*.pt' \
    --exclude='**/huggingface/**' \
    --exclude='**/*.safetensors' \
    --exclude='**/pytorch_model*.bin' \
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived lightweight run files to: $ARCHIVE_DIR"
  echo "Checkpoint/model artifacts remain at: $TRAIN_LOG_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

MODEL_PATH="$(resolve_model_init_path "$MODEL_INIT_CKPT" actor)"
build_sparse_update_mask

# -----------------------------
# Prepare SFT training data
# -----------------------------
RAW_TRAIN_FILE="$TRAIN_FILE"
SFT_PREPARED_DATA_DIR="${SFT_PREPARED_DATA_DIR:-${RUN_DIR}/prepared_data}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${SFT_PREPARED_DATA_DIR}/train_sft_messages.parquet}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:--1}"
SFT_DEDUP_BY_PROMPT="${SFT_DEDUP_BY_PROMPT:-false}"
SFT_RESPONSE_FILTER_CORRECT="${SFT_RESPONSE_FILTER_CORRECT:-true}"
SFT_ENABLE_THINKING_COLUMN="${SFT_ENABLE_THINKING_COLUMN:-}"
SFT_REUSE_PREPARED_DATA="${SFT_REUSE_PREPARED_DATA:-false}"

prepare_sft_data() {
  local input_path="$1"
  local output_path="$2"
  local converter="$WORK_DIR/tools/prepare_sft_messages_data.py"
  local converter_args=(
    --input "$input_path"
    --output "$output_path"
    --max-samples "$SFT_MAX_SAMPLES"
  )

  if [[ "$SFT_DEDUP_BY_PROMPT" == "true" ]]; then
    converter_args+=(--dedup-by-prompt)
  fi
  if [[ "$SFT_RESPONSE_FILTER_CORRECT" != "true" ]]; then
    converter_args+=(--no-filter-correct)
  fi
  if [[ -n "$SFT_ENABLE_THINKING_COLUMN" ]]; then
    converter_args+=(--enable-thinking "$SFT_ENABLE_THINKING_COLUMN")
  fi

  python3 "$converter" "${converter_args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; skipping SFT data preparation."
else
  mkdir -p "$SFT_PREPARED_DATA_DIR"
  case "${TRAIN_FILE##*.}" in
    jsonl|parquet|pq)
      if [[ "$SFT_REUSE_PREPARED_DATA" == "true" && -f "$SFT_TRAIN_FILE" ]]; then
        echo "Reusing prepared SFT messages dataset: $SFT_TRAIN_FILE" | tee "$LOG_DIR/prepare_sft_data.log"
      else
        echo "Preparing SFT messages dataset from: $TRAIN_FILE"
        prepare_sft_data "$TRAIN_FILE" "$SFT_TRAIN_FILE" | tee "$LOG_DIR/prepare_sft_data.log"
      fi
      TRAIN_FILE="$SFT_TRAIN_FILE"
      ;;
    *)
      echo "Unsupported TRAIN_FILE extension: $TRAIN_FILE" >&2
      exit 1
      ;;
  esac
fi
if [[ "$loss_eval_files" == "__TRAIN_FILE__" || "$loss_eval_files" == "$RAW_TRAIN_FILE" ]]; then
  loss_eval_files="$TRAIN_FILE"
fi
generation_eval_files="$(normalize_hydra_list_if_many "$generation_eval_files")"
if [[ -n "$generation_eval_names" ]]; then
  generation_eval_names="$(normalize_hydra_list_if_many "$generation_eval_names")"
fi
PRIMARY_VAL_FILE="$(list_eval_paths "$generation_eval_files" | head -n 1)"

# -----------------------------
# Debug info and input checks
# -----------------------------
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Run ID: ${RUN_ID}"
echo "SLURM nodes: ${SLURM_JOB_NODELIST:-local}"
echo "Head node: ${head_node}"
echo "Rendezvous endpoint: ${RDZV_ENDPOINT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NNODES: ${NNODES}"
echo "SCRATCH: ${SCRATCH}"
echo "RUN_DIR: ${RUN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "TRAIN_LOG_DIR: ${TRAIN_LOG_DIR}"
describe_path "WORK_DIR" "$WORK_DIR"
describe_path "MODEL_INIT_CKPT" "$MODEL_INIT_CKPT"
describe_path "MODEL_PATH" "$MODEL_PATH"
describe_path "SCORE_ROOT" "$SCORE_ROOT"
echo "PRUNING_SPARSITY: $PRUNING_SPARSITY"
echo "SPARSE_MASK_PATH: $SPARSE_MASK_PATH"
echo "sparse_update_enabled: $sparse_update_enabled"
echo "sparse_zero_frozen_params: $sparse_zero_frozen_params"
echo "sparse_verify_frozen_weights: $sparse_verify_frozen_weights"
describe_path "TRAIN_FILE" "$TRAIN_FILE"
describe_path "VAL_FILE" "$VAL_FILE"
describe_path "loss_eval_files" "$loss_eval_files"
describe_path "generation_eval_files" "$generation_eval_files"
if [[ -n "$generation_eval_names" ]]; then
  echo "generation_eval_names: $generation_eval_names"
fi
echo "eval_method: $eval_method"
echo "trainer_logger: $trainer_logger"
echo "truncation: $truncation"
echo "generation_eval_enable_thinking: $generation_eval_enable_thinking"

ls -ld "$WORK_DIR"
ls -lh "$TRAIN_FILE"
while IFS= read -r eval_path; do
  if [[ "$eval_path" != "null" && -n "$eval_path" ]]; then
    ls -lh "$eval_path"
  fi
done < <(list_eval_paths "$generation_eval_files")

cd "$WORK_DIR"

TRAINER_ARGS=(
  data.train_files="$TRAIN_FILE"
  data.val_files="$PRIMARY_VAL_FILE"
  data.loss_val_files="$loss_eval_files"
  data.generation_eval_files="$generation_eval_files"
  data.generation_eval_batch_size="$generation_eval_batch_size"
  data.generation_eval_apply_chat_template_kwargs.enable_thinking="$generation_eval_enable_thinking"
  data.messages_key=messages
  data.train_batch_size="$train_batch_size"
  data.micro_batch_size_per_gpu="$micro_batch_size_per_gpu"
  data.max_length="$max_length"
  data.truncation="$truncation"
  data.max_token_len_per_gpu="$max_token_len_per_gpu"
  data.train_max_samples="$train_max_samples"
  data.val_max_samples="$val_max_samples"
  data.ignore_input_ids_mismatch=True
  data.num_workers="$num_workers"
  optim.lr="$lr"
  engine=fsdp
  model.path="$MODEL_PATH"
  trainer.default_local_dir="$TRAIN_LOG_DIR"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$RUN_ID"
  trainer.total_epochs="$total_epochs"
  trainer.save_freq="$save_freq"
  trainer.save_initial_checkpoint="$save_initial_checkpoint"
  trainer.test_freq="$eval_freq"
  trainer.loss_test_freq="$loss_eval_freq"
  trainer.eval_method="$eval_method"
  trainer.val_before_train="$eval_before_train"
  trainer.generation_eval.test_freq="$generation_eval_freq"
  trainer.generation_eval.max_new_tokens="$generation_max_new_tokens"
  trainer.generation_eval.do_sample="$generation_do_sample"
  trainer.generation_eval.temperature="$generation_temperature"
  trainer.generation_eval.top_p="$generation_top_p"
  trainer.generation_eval.top_k="$generation_top_k"
  trainer.generation_eval.n="$generation_num_samples"
  trainer.generation_eval.dtype="$generation_dtype"
  trainer.generation_eval.backend="$generation_backend"
  trainer.generation_eval.vllm_gpu_memory_utilization="$generation_vllm_gpu_memory_utilization"
  trainer.generation_eval.vllm_host_ip="$generation_vllm_host_ip"
  trainer.generation_eval.vllm_enforce_eager="$generation_vllm_enforce_eager"
  trainer.generation_eval.vllm_sync_weights="$generation_vllm_sync_weights"
  trainer.generation_eval.vllm_enable_multiprocessing="$generation_vllm_enable_multiprocessing"
  trainer.logger="$trainer_logger"
  trainer.resume_mode="$resume_mode"
  trainer.resume_from_path="$resume_from_path"
  trainer.nnodes="$NNODES"
  trainer.n_gpus_per_node="$GPUS_PER_NODE"
  sparse_update.enabled="$sparse_update_enabled"
  sparse_update.mask_path="$SPARSE_MASK_PATH"
  sparse_update.mode=wanda_top
  sparse_update.zero_frozen_params="$sparse_zero_frozen_params"
  sparse_update.restore_frozen_after_step=true
  sparse_update.mask_optimizer_state=true
  sparse_update.verify_frozen_weights="$sparse_verify_frozen_weights"
  sparse_update.verification_interval="$sparse_verification_interval"
  sparse_update.strict_load=true
)

if [[ -n "$generation_eval_names" ]]; then
  TRAINER_ARGS+=(data.generation_eval_names="$generation_eval_names")
fi

print_planned_checkpoint_dirs

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; resolved trainer args:"
  printf '  %q\n' "${TRAINER_ARGS[@]}"
  exit 0
fi

# -----------------------------
# Run SFT training
# -----------------------------
# One Slurm task launches one torchrun agent per node. torchrun then launches GPUS_PER_NODE
# local trainer processes on that node. SLURM_PROCID is the node_rank because ntasks-per-node=1.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    if [[ -f "'"${VENV}"'/bin/activate" ]]; then
      source "'"${VENV}"'/bin/activate"
    fi
    cd "'"${WORK_DIR}"'"
    export PYTHONPATH="'"${WORK_DIR}"'${PYTHONPATH:+:${PYTHONPATH}}"
    export UV_CACHE_DIR="'"${UV_CACHE_DIR}"'"
    export HF_HOME="'"${HF_HOME}"'"
    node_cache_root="${TMPDIR:-/tmp}/verl_'"${RUN_ID}"'_node_${SLURM_PROCID}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_ROOT:-${node_cache_root}/huggingface_datasets}"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE_ROOT:-${node_cache_root}/huggingface_modules}"
    export TIKTOKEN_ENCODINGS_BASE="'"${TIKTOKEN_ENCODINGS_BASE}"'"
    export TORCH_EXTENSIONS_DIR="'"${TORCH_EXTENSIONS_DIR}"'/node_${SLURM_PROCID}"
    mkdir -p "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TORCH_EXTENSIONS_DIR"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=true
    export HYDRA_FULL_ERROR="'"${HYDRA_FULL_ERROR}"'"
    export NCCL_DEBUG="'"${NCCL_DEBUG}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"

    torchrun \
      --nnodes="'"${NNODES}"'" \
      --nproc_per_node="'"${GPUS_PER_NODE}"'" \
      --node_rank="${SLURM_PROCID}" \
      --rdzv_id="'"${RUN_ID}"'" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="'"${RDZV_ENDPOINT}"'" \
      -m verl.trainer.sft_trainer \
      "$@"
  ' _ "${TRAINER_ARGS[@]}" "$@" 2>&1 | tee "$TRAIN_STDOUT_LOG"
