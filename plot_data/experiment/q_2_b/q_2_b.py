import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ============================================================
# Data
# ============================================================

data = [
    # sparsity, method, measured I, predicted I_hat
    (30, "Magnitude", 1.610, 1.510),
    (30, "WANDA",     0.330, 0.284),
    (30, "WANDA++",   0.270, 0.243),
    (30, "SparseGPT", 0.369, 0.346),
    (30, "DAP",       0.315, 0.242),

    (35, "Magnitude", 2.160, 2.008),
    (35, "WANDA",     0.455, 0.378),
    (35, "WANDA++",   0.361, 0.313),
    (35, "SparseGPT", 0.494, 0.448),
    (35, "DAP",       0.392, 0.290),

    (40, "Magnitude", 2.886, 2.631),
    (40, "WANDA",     0.646, 0.517),
    (40, "WANDA++",   0.480, 0.410),
    (40, "SparseGPT", 0.675, 0.590),
    (40, "DAP",       0.532, 0.395),

    (45, "Magnitude", 3.850, 3.358),
    (45, "WANDA",     0.902, 0.688),
    (45, "WANDA++",   0.669, 0.531),
    (45, "SparseGPT", 0.928, 0.773),
    (45, "DAP",       0.750, 0.520),

    (50, "Magnitude", 5.120, 4.224),
    (50, "WANDA",     1.305, 0.919),
    (50, "WANDA++",   0.902, 0.676),
    (50, "SparseGPT", 1.311, 1.012),
    (50, "DAP",       1.122, 0.661),

    (55, "Magnitude", 6.640, 5.113),
    (55, "WANDA",     1.862, 1.199),
    (55, "WANDA++",   1.274, 0.854),
    (55, "SparseGPT", 1.800, 1.295),
    (55, "DAP",       1.584, 0.840),
]

df = pd.DataFrame(
    data,
    columns=["Sparsity", "Method", "Measured_I", "Predicted_I"]
)


# ============================================================
# Statistics
# ============================================================

# Pooled Spearman correlation across all configurations
rho_pooled, p_pooled = spearmanr(
    df["Predicted_I"],
    df["Measured_I"]
)

# Within-sparsity Spearman correlations
within_rhos = []

for sparsity, group in df.groupby("Sparsity"):
    rho, _ = spearmanr(
        group["Predicted_I"],
        group["Measured_I"]
    )
    within_rhos.append(rho)

rho_within_mean = np.mean(within_rhos)

print(f"Pooled Spearman rho: {rho_pooled:.3f}")
print(f"Mean within-sparsity Spearman rho: {rho_within_mean:.3f}")

for sparsity, rho in zip(sorted(df["Sparsity"].unique()), within_rhos):
    print(f"{sparsity}% sparsity: rho = {rho:.3f}")


# ============================================================
# Plot configuration
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

methods = [
    "Magnitude",
    "WANDA",
    "WANDA++",
    "SparseGPT",
    "DAP",
]

# Keep the same visual convention as your existing figures.
colors = {
    "Magnitude": "#1f77b4",
    "WANDA":     "#ff7f0e",
    "WANDA++":   "#2ca02c",
    "SparseGPT": "#d62728",
    "DAP":       "#9467bd",
}

markers = {
    "Magnitude": "o",
    "WANDA":     "s",
    "WANDA++":   "D",
    "SparseGPT": "^",
    "DAP":       "P",
}


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(4.2, 3.4),
    constrained_layout=True,
)

# Plot one scatter series per method.
for method in methods:
    sub = df[df["Method"] == method].sort_values("Sparsity")

    ax.scatter(
        sub["Measured_I"],
        sub["Predicted_I"],
        s=52,
        marker=markers[method],
        color=colors[method],
        edgecolors=colors[method],
        linewidths=1.2,
        label=method,
        zorder=3,
    )


# ============================================================
# y = x reference
# ============================================================

max_val = max(
    df["Measured_I"].max(),
    df["Predicted_I"].max()
)

lim_max = max_val * 1.06

ax.plot(
    [0, lim_max],
    [0, lim_max],
    linestyle="--",
    linewidth=1.4,
    color="black",
    alpha=0.7,
    label=r"$y=x$",
    zorder=1,
)


# ============================================================
# Optional sparsity labels
# ============================================================

# Label only the DAP points to avoid clutter.
dap = df[df["Method"] == "DAP"].sort_values("Sparsity")

counter = 0
for _, row in dap.iterrows():

    add_y = 0

    if counter == 0:
        add_y = 0.05
    elif counter == 1 or counter == 2:
        add_y = 0.1
    ax.annotate(
        f"{int(row['Sparsity'])}%",
        xy=(row["Measured_I"], row["Predicted_I"] + add_y),
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    counter += 1
# ============================================================
# Correlation annotation
# ============================================================

# text = (
#     rf"Pooled Spearman $\rho={rho_pooled:.3f}$"
#     "\n"
#     rf"Within-sparsity $\bar{{\rho}}={rho_within_mean:.3f}$"
# )

ax.text(
    0.97,
    0.05,
    rf"Mean within-sparsity $\rho = {rho_within_mean:.2f}$",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.75",
        alpha=0.9,
    ),
)

# ax.text(
#     0.97,
#     0.05,
#     rf"Mean Spearman $\rho$ over sparsity levels = {rho_within_mean:.2f}",
#     transform=ax.transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=11,
# )


# ============================================================
# Axes
# ============================================================

ax.set_xlabel(
    r"Measured remaining damage $I$",
    fontsize=11,
    labelpad=7,
)

ax.set_ylabel(
    r"Predicted remaining damage $\widehat{I}$",
    fontsize=11,
    labelpad=7,
)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=9,
    width=1.1,
    length=4.5,
)

ax.grid(
    True,
    linewidth=0.8,
    alpha=0.20,
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.legend(
    frameon=False,
    loc="upper left",
    ncol=2,
    handlelength=1.6,
    columnspacing=0.9,
    labelspacing=0.35,
    markerscale=0.88,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    "dap_predicted_vs_measured_remaining_damage.pdf",
    bbox_inches="tight",
)

fig.savefig(
    "dap_predicted_vs_measured_remaining_damage.png",
    dpi=400,
    bbox_inches="tight",
)

# plt.show()
