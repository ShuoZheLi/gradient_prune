#!/bin/bash
set -euo pipefail

# Self-contained pruning launcher.
# Edit the "Experiment config" block below instead of maintaining a separate YAML file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_VISIBLE_GPUS[@]}
fi
export CUDA_VISIBLE_DEVICES

CONFIG_FILE="$(mktemp --suffix=.yaml "${TMPDIR:-/tmp}/qwen3_8b_wanda_7500.XXXXXX")"
cleanup() {
  rm -f "$CONFIG_FILE"
}
trap cleanup EXIT

cat > "$CONFIG_FILE" <<'YAML'
# ============================================================================
# Experiment config
# Edit this YAML block to change the pruning/evaluation settings.
# ============================================================================
experiment_name: qwen3_8b_wanda_math7500
seed: 42

model:
  model_name_or_path: /data/shuozhe/saved_model/Qwen3.1-8B-Instruct
  dtype: bf16
  device: cuda:0
  trust_remote_code: true

pruning:
  prune_ops: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  sparsities: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  granularity: rowwise
  save_pruned_models: false
  load_masks: false
  load_scores: false
  score_root: /data/shuozhe/gradient_prune/results/qwen3_8b_wanda_math/scores

# methods: [dense, magnitude, wanda, gradient_norm, signed_first_order, signed_taylor, hybrid_wanda_signed_taylor]
methods: [wanda]

hybrid:
  lambda_values: [0.1]
  # lambda_values: [0.001, 0.01, 0.1, 1.0, 10.0]

calibration:
  type: prompt_response
  path: /data/shuozhe/gradient_prune/saved_calibration_dataset/qwen2.5-1.5b-instruct_math7500_correct
  only_correct: true
  loss_on: full_trajectory
  max_samples: 500
  microbatch_size: 32
  fisher_estimator: per_example
  max_length: 2048

calibration_ce:
  enabled: false
  backend: vllm
  path: /data/shuozhe/gradient_prune/saved_calibration_dataset/qwen2.5-1.5b-instruct_math500_correct
  type: prompt_response
  only_correct: true
  loss_on: response_only
  max_samples: 500
  batch_size: 64
  data_parallel_size: 3
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
  path: /data/shuozhe/gradient_prune/saved_calibration_dataset/qwen2.5-1.5b-instruct_math500_correct
  loss_on: response_only
  max_samples: 256
  batch_size: 64
  data_parallel_size: 3
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
  data_parallel_size: 3
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true
  max_length: 18432

task_accuracy:
  enabled: true
  dataset_path: /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet
  backend: vllm
  max_examples: 500
  prompt_key: prompt
  response_key: null
  reward_score_dir: null
  max_prompt_length: 2048
  max_new_tokens: 16384
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  batch_size: 64
  data_parallel_size: 3
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: auto
  enforce_eager: true
  trust_remote_code: true

output:
  root_dir: results/qwen3_8b_wanda_math7500
  save_stats: true
  save_masks: true
  save_plots: true
YAML

RESULTS_ROOT="results/qwen3_8b_wanda_math7500"

torchrun --standalone --nproc_per_node "$NUM_GPUS" -m experiment_runner \
  --config "$CONFIG_FILE"

python -m plotting \
  --results_csv "$RESULTS_ROOT/tables/main_results.csv" \
  --output_dir "$RESULTS_ROOT/plots" \
  --score_dir "$RESULTS_ROOT/scores"
