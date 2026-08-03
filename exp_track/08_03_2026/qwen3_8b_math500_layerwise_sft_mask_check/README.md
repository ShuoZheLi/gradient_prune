# Qwen3-8B WANDA Layerwise SFT-Mask Check

This folder tests whether the large `global_step_0` accuracy drop is explained by a mask-construction mismatch.

## Hypothesis

The 07/21 response-analysis scripts prune WANDA scores rowwise by default, while the sparse SFT step-0 checkpoint uses the VERL sparse-update WANDA mask, which keeps the top WANDA scores globally within each tensor.

If the layerwise/global response-analysis run drops toward the SFT `global_step_0` pass@1, the mismatch is likely confirmed.

## Script

Run:

```bash
sbatch /data/shuozhe/gradient_prune/exp_track/08_03_2026/qwen3_8b_math500_layerwise_sft_mask_check/qwen3_8b_wanda_math500_layerwise_s0d5_k1_greedy.sh
```

or:

```bash
/data/shuozhe/gradient_prune/exp_track/08_03_2026/qwen3_8b_math500_layerwise_sft_mask_check/submit_layerwise_s0d5_k1_greedy.sh
```

## Overrides Applied

The main script is self-contained and does not call any previous `exp_track` shell script.

It bakes in the same response-analysis workflow with these diagnostic defaults:

```bash
PRUNING_SPARSITY=0.5
PRUNE_GRANULARITY=layerwise
K=1
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=0
RUN_DENSE=0
RUN_PRUNED=1
RUN_ON_POLICY_ENTROPY=0
RUN_FIXED_PREFIX_ENTROPY=0
RUN_SURFACE_DIVERSITY=1
RUN_SEMANTIC_JUDGE=0
RUN_AGGREGATE=1
```

`PRUNE_GRANULARITY=layerwise` matches the sparse-update mask more closely because the sparse-update builder keeps the top WANDA entries after flattening each tensor.

## What To Compare

Compare the aggregate `pass_at_1` from this run against:

1. The original 07/21 rowwise response-analysis run for `s0d5`.
2. The SFT step-0 eval from:

```text
/data/shuozhe/gradient_prune/exp_track/08_01_2026/eval_distill_on_dsk_r1_distill_llama_8b_dataset/sft_qwen3_8b_wanda_s0d5/eval_global_step_0.sh
```

Use `pass_at_1`, not `pass_at_k` or `avg_at_k`, because this wrapper sets `K=1` to make the comparison greedy pass@1.

## Notes

The wrapper disables entropy and semantic-judge stages by default to keep the diagnostic focused on generation correctness. You can re-enable them with environment overrides if needed.
