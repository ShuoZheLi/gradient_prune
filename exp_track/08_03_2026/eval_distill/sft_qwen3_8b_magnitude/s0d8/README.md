# Eval Distill: Qwen3-8B Magnitude s0d8

These scripts evaluate Math500 accuracy for checkpoints from:

`/scratch/09576/shuozhe/verl_runs/sft_qwen3_8b_magnitude_s0d8_base_5x7500_891260/train_log`

Submit all checkpoints with:

```bash
sbatch eval_global_step_0.sh
# or
./submit_all_eval.sh
```

Results default to `$SCRATCH/gradient_prune/results/eval_distill/sft_qwen3_8b_magnitude/s0d8/<checkpoint>/runs/<run_id>`.
