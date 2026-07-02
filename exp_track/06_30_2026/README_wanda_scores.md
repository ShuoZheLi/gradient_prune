# Qwen3-8B-Base WANDA Score Collection

This experiment computes only WANDA pruning scores for:

- Model: `/data/shuozhe/saved_model/Qwen3-8B-Base`
- Calibration dataset: `/data/shuozhe/gradient_prune/saved_calibration_dataset/qwen2.5-1.5b-instruct_math7500_correct`
- Output: `/data/shuozhe/gradient_prune/results/06_30_2026/qwen3_8b_base_wanda_scores`

Run or resume launch:

```bash
bash /data/shuozhe/gradient_prune/exp_track/06_30_2026/run_qwen3_8b_base_wanda_scores.sh
```

Useful overrides:

```bash
NUM_GPUS=4 MAX_LENGTH=4096 MICROBATCH_SIZE=1 DTYPE=bf16 bash /data/shuozhe/gradient_prune/exp_track/06_30_2026/run_qwen3_8b_base_wanda_scores.sh
```

Monitor:

```bash
tail -f /data/shuozhe/gradient_prune/exp_track/06_30_2026/logs/qwen3_8b_base_wanda_scores.log
```

Each module score file is a `.pt` dictionary with key `wanda`. `metadata.json` maps module names to score files.
