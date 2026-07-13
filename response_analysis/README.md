# Response Analysis: Token Entropy and Diversity

This package evaluates pruned and unpruned models along separate axes:

- **Token entropy**: local next-token uncertainty, reported on-policy and on fixed teacher-forced prefixes.
- **Response diversity**: surface wording, parsed final-answer diversity, and semantic reasoning-strategy diversity.

The code reuses repository prompt normalization, Qwen `enable_thinking`, answer extraction, and task reward scoring from `src/task_scoring.py` and `create_calibration_dataset/model_accuracy_test.py`. It does not modify training code.

## Quick debug run

```bash
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh && conda activate verl
export PYTHONPATH=/data/shuozhe/gradient_prune:$PYTHONPATH

python -m response_analysis.generate_responses \
  --model_path /data/shuozhe/saved_model/Qwen3-0.6B \
  --model_id qwen3_0_6b \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output outputs/generations.jsonl \
  --debug_subset 4 --k 2 --temperature 1.0 --top_p 1.0 \
  --max_new_tokens 256 --enable_thinking true

python -m response_analysis.compute_token_entropy \
  --model_path /data/shuozhe/saved_model/Qwen3-0.6B \
  --input outputs/generations.jsonl \
  --output outputs/token_metrics.parquet

python -m response_analysis.build_fixed_prefix_bank \
  --tokenizer_path /data/shuozhe/saved_model/Qwen3-0.6B \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output outputs/fixed_prefix_bank.jsonl \
  --max_examples 4 --enable_thinking true

python -m response_analysis.compute_token_entropy \
  --mode fixed_prefix \
  --model_path /data/shuozhe/saved_model/Qwen3-0.6B \
  --prefix_bank outputs/fixed_prefix_bank.jsonl \
  --output outputs/fixed_token_metrics.parquet

python -m response_analysis.compute_surface_diversity \
  --input outputs/generations.jsonl \
  --output outputs/response_metrics.parquet
```

## Pruning from saved score files

Both generation and entropy CLIs can load score directories produced by the WANDA pruning jobs, for example a directory containing `metadata.json` and files named like `model__layers__0__self_attn__q_proj.pt`. Pass `--prune_score_dir` and `--pruning_sparsity`; the score key is inferred from metadata when possible, typically `wanda`.

```bash
python -m response_analysis.generate_responses \
  --model_path /data/shuozhe/saved_model/Qwen3-8B \
  --model_id qwen3_8b_wanda_s0.5 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output outputs/qwen3_8b_wanda_s0.5/generations.jsonl \
  --k 16 --temperature 1.0 --top_p 1.0 --enable_thinking true \
  --prune_score_dir /scratch/09576/shuozhe/gradient_prune/results/qwen3_8b_wanda_math7500/scores \
  --pruning_sparsity 0.5 --prune_granularity rowwise

python -m response_analysis.compute_token_entropy \
  --model_path /data/shuozhe/saved_model/Qwen3-8B \
  --model_id qwen3_8b_wanda_s0.5 \
  --input outputs/qwen3_8b_wanda_s0.5/generations.jsonl \
  --output outputs/qwen3_8b_wanda_s0.5/token_metrics.parquet \
  --prune_score_dir /scratch/09576/shuozhe/gradient_prune/results/qwen3_8b_wanda_math7500/scores \
  --pruning_sparsity 0.5 --prune_granularity rowwise
```

The mask rule matches the repository pruning code: lower scores are pruned, with `rowwise` pruning `floor(input_dim * sparsity)` weights per output row and `layerwise` pruning the lowest-scoring weights within each module. The model is pruned in memory after loading; training code is not touched.

## Semantic strategy judge

The judge uses an OpenAI-compatible Chat Completions endpoint only as a semantic evaluator. Keep the evaluator configurable:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.portkey.ai/v1'
export OPENAI_EVALUATOR_MODEL='@irom-ll37364-op-b37b3e/gpt-5.5'

python -m response_analysis.judge_strategy_diversity \
  --input outputs/generations.jsonl \
  --output outputs/semantic_judgments.jsonl \
  --metrics_output outputs/strategy_metrics.parquet
```

Every request is hashed and cached under `outputs/api_cache`, so reruns do not make extra API calls. Use `--disable_api` to require cache hits only.

## Aggregate

```bash
python -m response_analysis.aggregate_results \
  --token_metrics outputs/token_metrics.parquet \
  --fixed_token_metrics outputs/fixed_token_metrics.parquet \
  --response_metrics outputs/response_metrics.parquet \
  --strategy_metrics outputs/strategy_metrics.parquet \
  --per_prompt_output outputs/per_prompt_metrics.csv \
  --aggregate_output outputs/aggregate_metrics.csv
```

Outputs include per-prompt metrics, aggregate summary CSV, paired bootstrap comparisons, and scatter plots in `outputs/figures/`.

## Important interpretation

Do not compare only on-policy token entropy and conclude that a model is intrinsically smoother. On-policy entropy mixes uncertainty with differences in visited prefixes. Fixed-prefix entropy evaluates all models on identical teacher-forced states and is the safer distributional comparison.
