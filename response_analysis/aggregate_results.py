from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-prompt metrics and paired pruned-vs-unpruned comparisons.")
    parser.add_argument("--generations", default="outputs/generations.jsonl")
    parser.add_argument("--token_metrics", default="outputs/token_metrics.parquet")
    parser.add_argument("--fixed_token_metrics", default=None)
    parser.add_argument("--response_metrics", default="outputs/response_metrics.parquet")
    parser.add_argument("--strategy_metrics", default=None)
    parser.add_argument("--per_prompt_output", default="outputs/per_prompt_metrics.csv")
    parser.add_argument("--aggregate_output", default="outputs/aggregate_metrics.csv")
    parser.add_argument("--paired_output", default="outputs/paired_comparisons.csv")
    parser.add_argument("--figures_dir", default="outputs/figures")
    parser.add_argument("--baseline_model", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_parquet_optional(path: str | None) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def flatten_token_metrics(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    metrics = ["token_entropy_mean", "token_entropy_sum", "top1_probability_mean", "top1_logit_margin_mean", "token_logprob_mean", "response_length"]
    keep = ["model_id", "prompt_id"] + (["pruning_sparsity"] if "pruning_sparsity" in df.columns else [])
    grouped = df.groupby(["model_id", "prompt_id"], as_index=False).agg({col: "mean" for col in metrics if col in df.columns})
    if "pruning_sparsity" in df.columns:
        sparsity = df.groupby(["model_id", "prompt_id"], as_index=False)["pruning_sparsity"].first()
        grouped = grouped.merge(sparsity, on=["model_id", "prompt_id"], how="left")
    rename = {col: f"{prefix}{col}" for col in metrics if col in grouped.columns}
    return grouped.rename(columns=rename)


def paired_bootstrap(diff: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if diff.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(samples, diff.size))
    means = diff[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def paired_comparisons(per_prompt: pd.DataFrame, baseline_model: str | None, samples: int, seed: int) -> pd.DataFrame:
    if per_prompt.empty or "model_id" not in per_prompt.columns:
        return pd.DataFrame()
    models = sorted(per_prompt["model_id"].dropna().unique())
    if not models:
        return pd.DataFrame()
    baseline = baseline_model or models[0]
    metrics = [
        col
        for col in per_prompt.columns
        if col not in {"model_id", "prompt_id", "pruning_sparsity"} and pd.api.types.is_numeric_dtype(per_prompt[col])
    ]
    rows = []
    base = per_prompt[per_prompt["model_id"] == baseline].set_index("prompt_id")
    for model in models:
        if model == baseline:
            continue
        current = per_prompt[per_prompt["model_id"] == model].set_index("prompt_id")
        common = base.index.intersection(current.index)
        for metric in metrics:
            diff = (current.loc[common, metric] - base.loc[common, metric]).dropna().to_numpy(dtype=float)
            if diff.size == 0:
                continue
            lo, hi = paired_bootstrap(diff, samples, seed)
            rows.append(
                {
                    "baseline_model": baseline,
                    "model_id": model,
                    "metric": metric,
                    "num_prompts": int(diff.size),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(np.median(diff)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "fraction_pruned_higher": float(np.mean(diff > 0)),
                }
            )
    return pd.DataFrame(rows)


def write_figures(per_prompt: pd.DataFrame, figures_dir: str | Path) -> None:
    if per_prompt.empty:
        return
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def scatter(x: str, y: str, filename: str) -> None:
        if x not in per_prompt.columns or y not in per_prompt.columns:
            return
        plt.figure(figsize=(6, 4))
        for model_id, group in per_prompt.groupby("model_id"):
            plt.scatter(group[x], group[y], s=12, alpha=0.65, label=str(model_id))
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=200)
        plt.close()

    scatter("token_entropy_mean", "strategy_entropy", "token_entropy_vs_strategy_entropy.png")
    scatter("token_entropy_mean", "accuracy", "token_entropy_vs_correctness.png")
    scatter("pruning_sparsity", "token_entropy_mean", "pruning_sparsity_vs_token_entropy.png")
    scatter("pruning_sparsity", "effective_num_strategies", "pruning_sparsity_vs_effective_strategy_count.png")


def main() -> None:
    args = parse_args()
    response = read_parquet_optional(args.response_metrics)
    token = flatten_token_metrics(read_parquet_optional(args.token_metrics))
    fixed = flatten_token_metrics(read_parquet_optional(args.fixed_token_metrics), prefix="fixed_prefix_")
    strategy = read_parquet_optional(args.strategy_metrics)

    frames = [df for df in (response, token, fixed, strategy) if not df.empty]
    if not frames:
        per_prompt = pd.DataFrame()
    else:
        per_prompt = frames[0]
        for frame in frames[1:]:
            per_prompt = per_prompt.merge(frame, on=["model_id", "prompt_id"], how="outer", suffixes=("", "_dup"))
            for col in list(per_prompt.columns):
                if col.endswith("_dup"):
                    base = col[: -4]
                    if base in per_prompt.columns:
                        per_prompt[base] = per_prompt[base].combine_first(per_prompt[col])
                    per_prompt = per_prompt.drop(columns=[col])

    per_prompt_path = Path(args.per_prompt_output)
    per_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    per_prompt.to_csv(per_prompt_path, index=False)

    numeric_cols = [col for col in per_prompt.columns if col not in {"prompt_id"} and pd.api.types.is_numeric_dtype(per_prompt[col])]
    aggregate = per_prompt.groupby("model_id", as_index=False)[numeric_cols].mean() if not per_prompt.empty else pd.DataFrame()
    summary_order = [
        "model_id",
        "accuracy",
        "token_entropy_mean",
        "fixed_prefix_token_entropy_mean",
        "token_entropy_sum",
        "answer_entropy",
        "strategy_entropy",
        "effective_num_strategies",
        "response_length",
        "mean_response_length",
        "pruning_sparsity",
    ]
    aggregate = aggregate[[col for col in summary_order if col in aggregate.columns] + [col for col in aggregate.columns if col not in summary_order]]
    Path(args.aggregate_output).parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.aggregate_output, index=False)

    paired = paired_comparisons(per_prompt, args.baseline_model, args.bootstrap_samples, args.seed)
    Path(args.paired_output).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.paired_output, index=False)
    write_figures(per_prompt, args.figures_dir)


if __name__ == "__main__":
    main()
