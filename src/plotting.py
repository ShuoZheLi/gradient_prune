from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def _method_label(row):
    method = row["method"]
    if method == "hybrid_wanda_signed_taylor" and not pd.isna(row.get("lambda_value")):
        return f"hybrid λ={row['lambda_value']}"
    return method


def plot_metric(df, output_dir: Path, metric: str, ylabel: str, filename: str):
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    data = df.copy()
    data["label"] = data.apply(_method_label, axis=1)
    plotted = False
    for label, group in data.groupby("label"):
        group = group.sort_values("sparsity")
        if group[metric].notna().any():
            plt.plot(group["sparsity"], group[metric], marker="o", label=label)
            plotted = True
    plt.xlabel("sparsity")
    plt.ylabel(ylabel)
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"{filename}.{ext}")
    plt.close()


def plot_calibration_ce_change_vs_accuracy_change(df, output_dir: Path):
    dense = df[df["method"] == "dense"].sort_values("sparsity").head(1)
    if dense.empty:
        return
    dense_ce = float(dense.iloc[0].get("calibration_ce", float("nan")))
    dense_acc = float(dense.iloc[0].get("task_accuracy", float("nan")))
    plt.figure(figsize=(7, 5))
    for _, row in df.iterrows():
        if pd.isna(row.get("calibration_ce")) or pd.isna(row.get("task_accuracy")):
            continue
        plt.scatter(row["calibration_ce"] - dense_ce, row["task_accuracy"] - dense_acc)
        plt.annotate(_method_label(row), (row["calibration_ce"] - dense_ce, row["task_accuracy"] - dense_acc), fontsize=6)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("calibration CE pruned - dense")
    plt.ylabel("accuracy pruned - dense")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"calibration_ce_change_vs_accuracy_change.{ext}")
    plt.close()


def plot_negative_score_fraction(score_dir: str | None, output_dir: Path):
    if not score_dir:
        return
    score_path = Path(score_dir)
    rows = []
    for file in score_path.glob("*.pt"):
        obj = torch.load(file, map_location="cpu")
        if not isinstance(obj, dict):
            continue
        for method in ("signed_first_order", "signed_taylor"):
            score = obj.get(method)
            if score is None:
                continue
            name = file.stem.replace("__", ".")
            match = re.search(r"layers\.(\d+)", name)
            rows.append({"method": method, "layer": int(match.group(1)) if match else 0, "fraction": float((score < 0).float().mean().item())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    for method, group in df.groupby("method"):
        grouped = group.groupby("layer")["fraction"].mean().reset_index()
        plt.plot(grouped["layer"], grouped["fraction"], marker="o", label=method)
    plt.xlabel("layer index")
    plt.ylabel("fraction score < 0")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"negative_score_fraction_by_layer.{ext}")
    plt.close()


def plot_score_histograms(score_dir: str | None, output_dir: Path, max_layers: int = 3):
    if not score_dir:
        return
    files = sorted(Path(score_dir).glob("*.pt"))[:max_layers]
    for file in files:
        obj = torch.load(file, map_location="cpu")
        score = obj.get("signed_taylor") if isinstance(obj, dict) else None
        if score is None:
            continue
        name = file.stem.replace("__", ".")
        layer_match = re.search(r"layers\.(\d+)", name)
        layer = layer_match.group(1) if layer_match else "unknown"
        module_name = re.sub(r"[^A-Za-z0-9_]+", "_", name.split("layers.")[-1])
        plt.figure(figsize=(7, 5))
        plt.hist(score.flatten().numpy(), bins=100)
        plt.axvline(0, color="red", linewidth=1)
        plt.xlabel("signed Taylor score")
        plt.ylabel("count")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(output_dir / f"score_hist_layer_{layer}_{module_name}.{ext}")
        plt.close()


def make_plots(results_csv: str | Path, output_dir: str | Path, score_dir: str | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    plot_metric(df, output_dir, "task_accuracy", "held-out task exact-match accuracy", "accuracy_vs_sparsity")
    plot_metric(df, output_dir, "wikitext_ppl", "WikiText perplexity", "perplexity_vs_sparsity")
    plot_metric(df, output_dir, "heldout_ce", "held-out CE", "heldout_ce_vs_sparsity")
    plot_metric(df, output_dir, "calibration_ce", "calibration CE", "calibration_ce_vs_sparsity")
    plot_metric(df, output_dir, "generalization_gap", "heldout CE - calibration CE", "generalization_gap_vs_sparsity")
    plot_metric(df, output_dir, "accuracy_drop", "dense accuracy - pruned accuracy", "accuracy_drop_vs_sparsity")
    plot_metric(df, output_dir, "actual_sparsity", "actual sparsity", "actual_sparsity_vs_target_sparsity")
    plot_calibration_ce_change_vs_accuracy_change(df, output_dir)
    plot_negative_score_fraction(score_dir, output_dir)
    plot_score_histograms(score_dir, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_dir", default=None)
    args = parser.parse_args()
    make_plots(args.results_csv, args.output_dir, args.score_dir)


if __name__ == "__main__":
    main()
