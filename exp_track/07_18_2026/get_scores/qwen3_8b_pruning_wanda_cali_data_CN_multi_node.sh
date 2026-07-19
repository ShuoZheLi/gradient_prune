#!/bin/bash
#SBATCH --job-name=qwen3_8b_prune_wanda
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j_qwen3_8b_prune_wanda.out
#SBATCH --error=slurm-%j_qwen3_8b_prune_wanda.err

set -euo pipefail

# Self-contained multi-node pruning launcher.
# Edit the "Experiment config" block below instead of maintaining a separate YAML file.

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
    if [[ -f "$dir/pyproject.toml" && -d "$dir/src" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

# Slurm may execute a spooled copy under /var/spool. Prefer explicit overrides,
# then discover the repo by walking up from submit/current/script directories.
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
export PYTHONPATH="$repo_root/src:${PYTHONPATH:-}"

SCRATCH_ROOT="${SCRATCH:-/tmp/${USER:-shuozhe}}"
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

# -----------------------------
# Runtime config
# -----------------------------
RUN_NAME="${RUN_NAME:-qwen3_8b_prune_wanda_math7500}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME/_prune_/_}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}}"
RESULTS_BASE="${RESULTS_BASE:-${RESULTS_ROOT:-$SCRATCH_ROOT/gradient_prune/results}}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-$EXPERIMENT_NAME}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RESULTS_BASE/$RESULTS_SUBDIR}"
RESULTS_ROOT="${RUN_OUTPUT_DIR:-$EXPERIMENT_ROOT/runs/${RUN_ID}}"
LOAD_SCORES="${LOAD_SCORES:-true}"
SHARED_SCORE_ROOT="${SHARED_SCORE_ROOT:-$EXPERIMENT_ROOT/scores}"
if [[ "$LOAD_SCORES" == "true" ]]; then
  SCORE_ROOT="${SCORE_ROOT:-$SHARED_SCORE_ROOT}"
else
  SCORE_ROOT="${SCORE_ROOT:-$RESULTS_ROOT/scores}"
fi
LOG_DIR="${LOG_DIR:-$RESULTS_ROOT/logs}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2> >(tee -a "$LOG_DIR/run.err" >&2)

CONFIG_FILE="${CONFIG_FILE:-$LOG_DIR/config.yaml}"
CHINESE_CALIBRATION_PATH="${CHINESE_CALIBRATION_PATH:-$LOG_DIR/chinese_calibration_text.jsonl}"
mkdir -p "$(dirname -- "$CHINESE_CALIBRATION_PATH")"

cat > "$CHINESE_CALIBRATION_PATH" <<'JSONL'
{"text":"北京市位于华北平原北部，是中华人民共和国的首都。北京有三千多年建城史和八百多年建都史，故宫、天坛、颐和园等古迹见证了城市的历史变迁。今天的北京也是全国政治、文化、国际交往和科技创新中心，地铁、高铁与航空网络把城市同全国各地紧密连接。"}
{"text":"长江发源于青藏高原，流经中国西部、中部和东部多个省市，最终注入东海。长江流域水系发达，土地肥沃，孕育了丰富的农业、航运和城市文明。三峡、洞庭湖、鄱阳湖以及长江三角洲共同构成了多样的自然与经济景观。"}
{"text":"春节是中国最重要的传统节日之一。节日前，人们常常打扫房屋、置办年货、贴春联和窗花；除夕夜，全家团聚吃年夜饭，守岁迎新。春节期间，拜年、发红包、舞龙舞狮和逛庙会等习俗表达了人们辞旧迎新、祝愿平安的心情。"}
{"text":"中国古典诗词讲究意境、音韵和凝练的表达。诗人常借山水、明月、风雨、花木等意象抒发情感，也通过边塞、田园、送别和怀古等题材记录社会生活。唐诗和宋词在汉语文学史上影响深远，至今仍被广泛诵读和研究。"}
{"text":"现代农业越来越重视科学管理。农民可以利用土壤检测、气象预报、节水灌溉和病虫害监测来安排播种、施肥与收获。电子商务和冷链物流的发展，也帮助新鲜农产品更快进入城市市场，提高了乡村产业的组织效率。"}
{"text":"人工智能技术正在改变学习、科研和生产方式。语言模型可以辅助阅读、翻译、写作和程序开发，但可靠使用仍需要明确问题、核对事实、保护隐私，并由人类对关键决策负责。技术进步应当服务于人的创造力、教育机会和社会福祉。"}
JSONL
export CHINESE_CALIBRATION_PATH

MODEL_PATH="${MODEL_PATH:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
if [[ "$DRY_RUN" != "1" && ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist on this node: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/path/to/HF/model visible on all compute nodes." >&2
  exit 2
fi
if [[ "$LOAD_SCORES" == "true" && "$DRY_RUN" != "1" && ! -f "$SCORE_ROOT/metadata.json" ]]; then
  echo "Score root does not contain saved scores: $SCORE_ROOT" >&2
  echo "Set SCORE_ROOT=/path/to/saved/scores or set RESULTS_SUBDIR to the experiment directory containing scores/." >&2
  exit 3
fi

cat > "$CONFIG_FILE" <<'YAML'
# ============================================================================
# Experiment config
# Edit this YAML block to change the pruning/evaluation settings.
# ============================================================================
experiment_name: __EXPERIMENT_NAME__
seed: 42

model:
  model_name_or_path: __MODEL_PATH__
  dtype: bf16
  device: cuda:0
  trust_remote_code: true

pruning:
  prune_ops: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  sparsities: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  granularity: rowwise
  save_pruned_models: false
  load_masks: false
  load_scores: __LOAD_SCORES__
  score_root: __SCORE_ROOT__

# methods: [dense, magnitude, wanda, gradient_norm, signed_first_order, signed_taylor, hybrid_wanda_signed_taylor]
methods: [wanda]

hybrid:
  lambda_values: [0.1]
  # lambda_values: [0.001, 0.01, 0.1, 1.0, 10.0]

calibration:
  type: text
  path: __CHINESE_CALIBRATION_PATH__
  only_correct: false
  loss_on: full_text
  max_samples: null
  microbatch_size: 16
  fisher_estimator: per_example
  max_length: 18432

calibration_ce:
  enabled: false
  backend: vllm
  path: /work2/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math500_correct
  type: prompt_response
  only_correct: true
  loss_on: response_only
  max_samples: null
  batch_size: 64
  data_parallel_size: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true
  shared_vllm: false
  max_length: 18432

heldout_ce:
  enabled: false
  backend: vllm
  path: /work2/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math500_correct
  loss_on: response_only
  max_samples: null
  batch_size: 64
  data_parallel_size: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true
  max_length: 18432

text_ppl:
  enabled: true
  backend: vllm
  dataset_name: wikitext
  dataset_config: wikitext-2-raw-v1
  split: validation
  text_key: text
  max_samples: 256
  batch_size: 64
  data_parallel_size: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true
  max_length: 18432

task_accuracy:
  enabled: true
  dataset_path: /work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet
  backend: vllm
  max_examples: null
  prompt_key: prompt
  response_key: null
  reward_score_dir: null
  scorer_backend: verl_math_reward
  max_prompt_length: 2048
  max_new_tokens: 16384
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  batch_size: 64
  data_parallel_size: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true
  enable_thinking: true

output:
  root_dir: __RESULTS_ROOT__
  save_stats: true
  save_masks: false
  save_plots: true
YAML

python3 - "$CONFIG_FILE" "$RESULTS_ROOT" "$MODEL_PATH" "$SCORE_ROOT" "$EXPERIMENT_NAME" "$LOAD_SCORES" <<'CONFIG_PATH_PY'
import os
import sys
from pathlib import Path
config_path = Path(sys.argv[1])
results_root = sys.argv[2]
model_path = sys.argv[3]
score_root = sys.argv[4]
experiment_name = sys.argv[5]
load_scores = sys.argv[6]
chinese_calibration_path = Path(os.environ["CHINESE_CALIBRATION_PATH"]).resolve().as_posix()
text = config_path.read_text()
text = text.replace("__RESULTS_ROOT__", results_root)
text = text.replace("__MODEL_PATH__", model_path)
text = text.replace("__SCORE_ROOT__", score_root)
text = text.replace("__EXPERIMENT_NAME__", experiment_name)
text = text.replace("__LOAD_SCORES__", load_scores)
text = text.replace("__CHINESE_CALIBRATION_PATH__", chinese_calibration_path)
config_path.write_text(text)
CONFIG_PATH_PY

# -----------------------------
# Slurm node/device discovery
# -----------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
  nodes_array=("$(hostname)")
fi
if [[ ${#nodes_array[@]} -lt 1 ]]; then
  echo "Could not discover any Slurm nodes." >&2
  exit 1
fi

num_nodes="${NUM_NODES:-${#nodes_array[@]}}"
if [[ "$num_nodes" -gt "${#nodes_array[@]}" ]]; then
  echo "NUM_NODES=$num_nodes exceeds allocated/discovered nodes=${#nodes_array[@]}." >&2
  exit 2
fi
master_addr="${MASTER_ADDR:-${nodes_array[0]}}"
master_port="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"

nproc_per_node="${NPROC_PER_NODE:-}"
if [[ -z "$nproc_per_node" ]]; then
  if [[ -n "${LOCAL_DEVICES:-${DEVICES:-}}" ]]; then
    IFS=',' read -r -a local_devices_array <<< "${LOCAL_DEVICES:-${DEVICES:-}}"
    nproc_per_node="${#local_devices_array[@]}"
    export CUDA_VISIBLE_DEVICES="${LOCAL_DEVICES:-${DEVICES:-}}"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    nproc_per_node="$SLURM_GPUS_ON_NODE"
  else
    nproc_per_node="${NUM_GPUS_PER_NODE:-1}"
  fi
fi
world_size=$((num_nodes * nproc_per_node))

if [[ "$num_nodes" -lt 1 || "$nproc_per_node" -lt 1 ]]; then
  echo "num_nodes and nproc_per_node must be positive; got num_nodes=$num_nodes nproc_per_node=$nproc_per_node" >&2
  exit 2
fi

torchrun_args=(
  --nnodes "$num_nodes"
  --nproc-per-node "$nproc_per_node"
  --rdzv-backend c10d
  --rdzv-endpoint "${master_addr}:${master_port}"
  --rdzv-id "prune_wanda_${SLURM_JOB_ID:-manual}"
)
runner_args=(
  -m experiment_runner
  --config "$CONFIG_FILE"
)
prep_runner_args=(
  -m experiment_runner
  --config "$CONFIG_FILE"
  --prep-only
)
sharded_runner_args=(
  -m experiment_runner
  --config "$CONFIG_FILE"
  --condition-shard-id __CONDITION_SHARD_ID__
  --num-condition-shards __NUM_CONDITION_SHARDS__
)
merge_runner_args=(
  -m experiment_runner
  --config "$CONFIG_FILE"
  --merge-only
)

# This WANDA sweep loads precomputed scores, so there is no distributed
# calibration work left to do. Use all allocated nodes as independent condition
# evaluators by default. Set PREP_WITH_TORCHRUN=1 only for configs that need
# multi-node gradient/activation collection before the sweep.
PREP_WITH_TORCHRUN="${PREP_WITH_TORCHRUN:-0}"
SHARDED_EVAL="${SHARDED_EVAL:-1}"

# -----------------------------
# Debug info
# -----------------------------
echo "[prune] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[prune] Run ID: $RUN_ID"
echo "[prune] Run timestamp: $RUN_TIMESTAMP"
echo "[prune] load_scores=$LOAD_SCORES"
echo "[prune] repo_root=$repo_root"
echo "[prune] nodes=${nodes_array[*]}"
echo "[prune] num_nodes=$num_nodes nproc_per_node=$nproc_per_node world_size=$world_size"
echo "[prune] master=${master_addr}:${master_port}"
echo "[prune] experiment_name=$EXPERIMENT_NAME"
echo "[prune] results_base=$RESULTS_BASE"
echo "[prune] results_subdir=$RESULTS_SUBDIR"
echo "[prune] experiment_root=$EXPERIMENT_ROOT"
echo "[prune] results_root=$RESULTS_ROOT"
echo "[prune] score_root=$SCORE_ROOT"
echo "[prune] shared_score_root=$SHARED_SCORE_ROOT"
echo "[prune] model_path=$MODEL_PATH"
echo "[prune] log_dir=$LOG_DIR"
echo "[prune] cache_root=$cache_root"
echo "[prune] venv=${VENV:-none}"
echo "[prune] python=$(command -v python3 || command -v python || true)"
printf '[prune] command:'
printf ' %q' torchrun "${torchrun_args[@]}" "${prep_runner_args[@]}"
printf '\n'
echo "[prune] prep_with_torchrun=$PREP_WITH_TORCHRUN sharded_eval=$SHARDED_EVAL"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[prune] dry run complete; no pruning launched."
  exit 0
fi

if [[ "$PREP_WITH_TORCHRUN" == "1" ]]; then
  # One torchrun launcher per node. torchrun coordinates all ranks through the same rendezvous endpoint.
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1 \
      torchrun "${torchrun_args[@]}" "${prep_runner_args[@]}"
  else
    torchrun --standalone --nnodes=1 --nproc-per-node="$nproc_per_node" "${prep_runner_args[@]}"
  fi
fi

if [[ "$SHARDED_EVAL" == "1" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1 \
      bash -lc 'set -euo pipefail; shard_id=${SLURM_PROCID:?}; args=(); for arg in "$@"; do arg=${arg/__CONDITION_SHARD_ID__/$shard_id}; arg=${arg/__NUM_CONDITION_SHARDS__/'"$num_nodes"'}; args+=("$arg"); done; python "${args[@]}"' \
      _ "${sharded_runner_args[@]}"
  else
    python "${runner_args[@]}"
  fi
  python "${merge_runner_args[@]}"
else
  python "${runner_args[@]}"
fi
