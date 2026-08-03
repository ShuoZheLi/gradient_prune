# Qwen3 Sparse Teacher SFT Data Collection

This folder contains a wrapper to collect Math7500 correct 5-response SFT data from a WANDA-pruned Qwen3 teacher.

## Main Script

```bash
sbatch collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh
```

or:

```bash
submit_collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh
```

## What It Does

1. Builds a temporary/prereusable HF checkpoint for a WANDA-pruned Qwen3 teacher.
2. Uses the score directory from the 08/01 sparse SFT script:

```text
${SCRATCH}/gradient_prune/results/qwen3_8b_wanda_math7500/scores
```

3. Reuses the existing 08/01 DeepSeek-R1-Distill-Llama-8B collection script to generate 5 responses per Math7500 prompt and keep only correct trajectories.

## Defaults

```bash
BASE_MODEL_PATH=/work2/09576/shuozhe/saved_model/Qwen3-8B
PRUNING_SPARSITY=0.5
PRUNE_GRANULARITY=layerwise
SCORE_ROOT=${SCRATCH}/gradient_prune/results/qwen3_8b_wanda_math7500/scores
NUM_RESPONSES_PER_PROMPT=5
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=0
```

The default is Qwen3-8B because the requested score root is a Qwen3-8B WANDA score directory. If you meant a true Qwen3-3B/4B model, provide a matching score directory with the same tensor shapes; the Qwen3-8B scores will not match a smaller model.

## Output

Default output goes to:

```text
saved_calibration_dataset/qwen3_8b_wanda_s0d5_layerwise_math7500_correct_5_response/
```

The final training parquet defaults to:

```text
saved_calibration_dataset/qwen3_8b_wanda_s0d5_layerwise_math7500_correct_5_response/qwen3_8b_wanda_s0d5_layerwise_math7500_correct_5_response.parquet
```

## Useful Overrides

Use rowwise pruning instead of layerwise:

```bash
PRUNE_GRANULARITY=rowwise sbatch collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh
```

Use a different compatible Qwen3 model and score root:

```bash
BASE_MODEL_PATH=/path/to/qwen3-model \
SCORE_ROOT=/path/to/matching/wanda/scores \
TEACHER_ID=qwen3_custom_wanda_s0d5 \
sbatch collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh
```

Reuse or choose a specific pruned-teacher checkpoint directory:

```bash
PRUNED_TEACHER_DIR=/scratch/09576/shuozhe/gradient_prune/pruned_teachers/qwen3_8b_wanda_s0d5_layerwise \
sbatch collect_qwen3_sparse_wanda_s0d5_math7500_multi_node_5_response.sh
```
