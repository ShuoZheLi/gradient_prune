# Eval Distill: Student Qwen3-8B WANDA s0d5 from s0d4 Teacher

These scripts evaluate Math500 accuracy for checkpoints from:

`/scratch/09576/shuozhe/verl_runs/sft_qwen3_8b_wanda_s0d5_base_qwen3_8b_wanda_s0d2_layerwise_math7500_correct_5_response_888642/train_log`

Submit all checkpoints with:

```bash
sbatch eval_global_step_0.sh
# or
./submit_all_eval.sh
```

Results default to `$SCRATCH/gradient_prune/results/eval_distill/student_qwen3_8b_s0d5/s0d2/<checkpoint>/runs/<run_id>`.
