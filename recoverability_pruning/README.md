# Recoverability-aware pruning scores

`generate_recoverability_scores.py` estimates the dense reference curvature and
reference/KD recovery covariance with shared Rademacher probes. It never modifies
or prunes the checkpoint.

The supplied trajectory parquet files do not contain prompt boundaries or loss
masks. Use `--loss_on full_trajectory` for those files. The CLI rejects
`response_only` unless an explicit `prompt_length` column exists, and rejects
`loss_mask` unless an explicit mask column exists.

## Required smoke test

Run one shared reference/KD HVP before a full job:

```bash
python generate_recoverability_scores.py \
  --model_path /path/to/model \
  --ref_dataset_path /path/to/ref.parquet \
  --kd_dataset_path /path/to/kd.parquet \
  --output_path /path/to/smoke.pt \
  --probe_lr_eta 1e-5 \
  --num_probes 1 \
  --max_ref_samples 1 \
  --max_kd_samples 1 \
  --max_length 128 \
  --candidate_modules q_proj \
  --hvp_parameter_scope all \
  --loss_on full_trajectory \
  --smoke_test
```

The model is loaded with eager attention for reverse-over-reverse autodiff.
For long-context large-model runs, `--gradient_checkpointing` enables PyTorch's
non-reentrant checkpointing path. Hugging Face requires train mode for this path,
so the loader forces every `torch.nn.Dropout` probability to zero before enabling
train mode; this keeps the scoring objective deterministic while reducing retained
activation memory.

## Full score generation

```bash
python generate_recoverability_scores.py \
  --model_path /path/to/model \
  --ref_dataset_path /path/to/ref.parquet \
  --kd_dataset_path /path/to/kd.parquet \
  --output_path /path/to/recoverability_scores.pt \
  --probe_lr_eta 1e-5 \
  --num_probes 16 \
  --candidate_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --hvp_parameter_scope all \
  --loss_on full_trajectory \
  --convergence_checkpoints 2,4,8,16
```

The output preserves exact Hugging Face parameter names such as
`model.layers.0.self_attn.q_proj.weight`. It contains `scores`, `damage`, and
`recovery`; `--save_intermediate_stats` additionally stores `h_ref` and `rho`.

The HVP scope must match the parameters updated by the future SFT step. The
default `all` is correct for ordinary full-model SFT. Use `transformer` or
`candidates` only when the corresponding excluded parameters will actually be
frozen during SFT.

## Multi-node probe parallelism

Use `torchrun` with `--distributed_probe_parallel` to distribute probes across
one-GPU nodes. Every rank loads one complete dense model replica on its local
GPU, receives a disjoint contiguous range of probe indices, and evaluates both
the reference and KD HVP for those same probes. This is probe parallelism, not
DDP/FSDP model sharding; second-order autodiff never crosses ranks.

Each nonzero rank writes its local Welford state to shared storage. Rank zero
merges the states with the exact parallel sample-covariance formula and is the
only rank that writes convergence snapshots, diagnostics, and the final score
file. The result is estimator-equivalent to sequential probe order up to normal
floating-point reduction-order differences.

```bash
torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node=1 \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  generate_recoverability_scores.py \
  ... \
  --device cuda:0 \
  --distributed_probe_parallel \
  --distributed_state_dir /shared/path/distributed_state
```

`num_probes` must be at least the world size. Distributed convergence snapshots
can be written while rank zero is processing its own initial probe block and
after each complete rank block is merged. For example, with 16 probes and four
ranks, `2,4,8,12,16` are supported, while `6` is not.

For Qwen3-8B's default seven candidate matrices, the three float32 Welford
accumulators contain about 6.95B values each in aggregate. Plan for roughly
80 GiB host RAM on every worker, over 200 GiB on rank zero during final score
construction, and around 500 GiB of shared storage when intermediate tensors,
four convergence snapshots, and temporary rank states coexist. Probe
parallelism reduces elapsed probe time approximately in proportion to the
number of nodes; it does not divide dataset batches among nodes or reduce these
per-rank model/statistics requirements.

## Probe convergence

```bash
python analyze_recoverability_convergence.py \
  /path/to/convergence/scores_probe_*.pt \
  --output_path /path/to/convergence/spearman.json
```

The report includes global, per-layer, and per-matrix Spearman correlations.
Large tensors are deterministically sampled according to the analyzer limits,
and every entry records whether its correlation is exact or sampled.
