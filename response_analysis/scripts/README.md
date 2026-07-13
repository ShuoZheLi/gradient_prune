# Cluster Launchers

## Qwen3-8B WANDA Response Analysis

Launcher:

```bash
sbatch response_analysis/scripts/qwen3_8b_wanda_response_analysis_multi_node.sh
```

Common overrides:

```bash
sbatch \
  --export=ALL,\
MODEL_PATH=/work2/09576/shuozhe/saved_model/Qwen3-8B,\
DATASET_PATH=/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet,\
SCORE_ROOT=/scratch/09576/shuozhe/gradient_prune/results/qwen3_8b_wanda_math7500/scores,\
PRUNING_SPARSITY=0.5,\
MAX_EXAMPLES=500,K=16,MAX_NEW_TOKENS=2048,ENABLE_THINKING=true \
  response_analysis/scripts/qwen3_8b_wanda_response_analysis_multi_node.sh
```

Useful toggles:

- `PARALLEL_GENERATION=auto`: shard generation across Slurm nodes when `--nodes > 1`; set `0` for serial or `1` to force sharding.
- `PARALLEL_ENTROPY=auto`: shard on-policy and fixed-prefix entropy across Slurm nodes after generation/prefix-bank creation.
- `RUN_DENSE=1` / `RUN_PRUNED=1`: run dense and/or score-pruned model.
- `RUN_GENERATION=1`: generate `K` responses per prompt.
- `RUN_ON_POLICY_ENTROPY=1`: compute token entropy on generated trajectories.
- `RUN_FIXED_PREFIX_ENTROPY=1`: build a fixed prefix bank and compute teacher-forced entropy.
- `RUN_SURFACE_DIVERSITY=1`: compute surface and final-answer diversity.
- `RUN_SEMANTIC_JUDGE=1`: run the OpenAI/Portkey semantic strategy judge.
- `RUN_AGGREGATE=1`: combine model outputs and write aggregate CSVs/figures.
- `DEBUG_SUBSET=8`: quick small run.
- `DRY_RUN=1`: only resolve paths and write `config.env`.

Multi-node behavior:

- Generation is sharded by prompt range across allocated Slurm nodes with `srun -N1 -n1 -w <node>`.
- Shards write `dense/generation_shards/generations_shard_*.jsonl` or `pruned_s*/generation_shards/...`, then merge into `generations.jsonl`.
- Entropy can also be sharded across nodes; entropy shards are merged back into `token_metrics.parquet` and `fixed_token_metrics.parquet`.
- Diversity, semantic judging, and aggregation run after merge on the launcher process. This avoids duplicated API calls and keeps paired aggregation deterministic.
- `MAX_EXAMPLES` or `DEBUG_SUBSET` must be non-negative for sharded generation.

Semantic judge example:

```bash
export OPENAI_API_KEY='...'
sbatch --export=ALL,RUN_SEMANTIC_JUDGE=1,OPENAI_BASE_URL=https://api.portkey.ai/v1,OPENAI_EVALUATOR_MODEL=@irom-ll37364-op-b37b3e/gpt-5.5 \
  response_analysis/scripts/qwen3_8b_wanda_response_analysis_multi_node.sh
```

Default output root:

```text
$SCRATCH/gradient_prune/results/response_analysis/<RUN_NAME>/runs/<RUN_ID>/
├── dense/
├── pruned_s<PRUNING_SPARSITY>/
├── combined/
└── logs/
```
