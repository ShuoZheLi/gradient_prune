# Signed Taylor Pruning

Clean experimental framework for pruning HuggingFace causal LMs with magnitude, WANDA, gradient-norm, signed first-order, Signed Taylor, and hybrid WANDA + Signed Taylor scores.

Signed Taylor uses the pruning perturbation `Δw = -w` and scores each scalar weight with:

```text
score = -g * w + 0.5 * h * w^2
h = mean(grad^2)
```

Lower scores are pruned. Negative scores are preserved as meaningful predictions that pruning may reduce calibration CE.

## Setup

```bash
cd /data/shuozhe/gradient_prune
pip install -e .[test]
```

In the provided environment:

```bash
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
cd /data/shuozhe/gradient_prune
```

## Run

```bash
bash scripts/run_debug.sh
bash scripts/run_math_pruning.sh
bash scripts/plot_results.sh
```

Outputs are written under `results/<experiment>/`, including `tables/`, `plots/`, `masks/`, `stats/`, `scores/`, and optional `models/`.

For vLLM task accuracy on small models, set `task_accuracy.data_parallel_size: 4` and `task_accuracy.tensor_parallel_size: 1` to launch four one-GPU vLLM workers, split examples across them, and merge the results. `task_accuracy.batch_size` controls each worker's generation chunk size. The runner saves the current dense/pruned model checkpoint under that method's accuracy directory before launching vLLM.

## Scientific checks

Do not judge Signed Taylor by calibration CE alone. Compare same-sparsity held-out accuracy, held-out CE, WikiText perplexity, calibration-vs-heldout gaps, and accuracy drops against dense, magnitude, WANDA, gradient-norm, and hybrid baselines.
