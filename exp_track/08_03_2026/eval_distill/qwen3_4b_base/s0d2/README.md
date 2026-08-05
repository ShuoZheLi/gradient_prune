# Eval Distill: Qwen3-4B Base from Qwen3-8B WANDA s0d2 Teacher

These scripts evaluate Math500 accuracy for checkpoints from:

`/scratch/09576/shuozhe/verl_runs/sft_qwen3_4b_base_qwen3_8b_wanda_s0d2_layerwise_math7500_correct_5_response_888757/train_log`

Submit all checkpoints with:

```bash
./submit_all_eval.sh
```

Submit one checkpoint with:

```bash
sbatch eval_global_step_0.sh
```

Results default to `$SCRATCH/gradient_prune/results/eval_distill/qwen3_4b_base/s0d2/<checkpoint>/runs/<run_id>`.
