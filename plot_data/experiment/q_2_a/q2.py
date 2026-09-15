import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Data
# ============================================================

rows = [
    (30, "Magnitude", 2.30, 0.30, 0.33),
    (30, "WANDA", 1.00, 0.67, 0.71),
    (30, "WANDA++", 0.75, 0.64, 0.67),
    (30, "SparseGPT", 0.82, 0.55, 0.57),
    (30, "DAP", 1.21, 0.74, 0.75),

    (35, "Magnitude", 3.00, 0.28, 0.31),
    (35, "WANDA", 1.30, 0.65, 0.70),
    (35, "WANDA++", 0.95, 0.62, 0.66),
    (35, "SparseGPT", 1.05, 0.53, 0.56),
    (35, "DAP", 1.45, 0.73, 0.75),

    (40, "Magnitude", 3.90, 0.26, 0.29),
    (40, "WANDA", 1.70, 0.62, 0.68),
    (40, "WANDA++", 1.20, 0.60, 0.64),
    (40, "SparseGPT", 1.35, 0.50, 0.54),
    (40, "DAP", 1.90, 0.72, 0.74),

    (45, "Magnitude", 5.00, 0.23, 0.27),
    (45, "WANDA", 2.20, 0.59, 0.66),
    (45, "WANDA++", 1.52, 0.56, 0.62),
    (45, "SparseGPT", 1.75, 0.47, 0.52),
    (45, "DAP", 2.50, 0.70, 0.73),

    (50, "Magnitude", 6.40, 0.20, 0.25),
    (50, "WANDA", 2.90, 0.55, 0.64),
    (50, "WANDA++", 1.92, 0.53, 0.60),
    (50, "SparseGPT", 2.30, 0.43, 0.50),
    (50, "DAP", 3.40, 0.67, 0.72),

    (55, "Magnitude", 8.00, 0.17, 0.23),
    (55, "WANDA", 3.80, 0.51, 0.62),
    (55, "WANDA++", 2.45, 0.48, 0.58),
    (55, "SparseGPT", 3.00, 0.40, 0.48),
    (55, "DAP", 4.40, 0.64, 0.70),
]

df = pd.DataFrame(
    rows,
    columns=[
        "Sparsity",
        "Method",
        "D",
        "Observed",
        "Predicted",
    ],
)

# Match the x-range used in your previous figure.
df = df[df["D"] <= 4.4].copy()


# ============================================================
# Compute measured remaining damage
#
# Observed = R / D
#
# I = D - R
#   = D * (1 - R/D)
# ============================================================

df["I"] = df["D"] * (1.0 - df["Observed"])


# ============================================================
# Methods / markers
# ============================================================

methods = [
    "Magnitude",
    "WANDA",
    "WANDA++",
    "SparseGPT",
    "DAP",
]

markers = {
    "Magnitude": "o",
    "WANDA": "s",
    "WANDA++": "D",
    "SparseGPT": "^",
    "DAP": "P",
}

method_colors = {
    "Magnitude": "#1f77b4",
    "WANDA": "#ff7f0e",
    "WANDA++": "#2ca02c",
    "SparseGPT": "#d62728",
    "DAP": "#9467bd",
}


# ============================================================
# Paper-style matplotlib settings
# ============================================================

plt.rcParams.update({
    "font.size": 9,

    "axes.labelsize": 11,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "legend.fontsize": 9,
    "legend.title_fontsize": 9,

    # Better PDF font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(4.2, 3.4),
    constrained_layout=True,
)


# ============================================================
# Plot trajectories
# ============================================================

for method in methods:

    g = (
        df[df["Method"] == method]
        .sort_values("Sparsity")
    )

    ax.plot(
        g["D"],
        g["I"],
        marker=markers[method],
        linewidth=2.6,
        markersize=8.5,
        markeredgewidth=1.2,
        color=method_colors[method],
        label=method,
    )


# ============================================================
# Sparsity labels
#
# Label the last visible point of each trajectory, but show the first
# (30% sparsity) label only for Magnitude and WANDA++.
# ============================================================

for method in methods:

    g = (
        df[df["Method"] == method]
        .sort_values("Sparsity")
    )

    label_rows = [g.iloc[-1]]

    if method in {"Magnitude", "WANDA++"}:
        label_rows.insert(0, g.iloc[0])

    for r in label_rows:

        is_30_percent = r["Sparsity"] == 30

        ax.annotate(
            f"{int(r['Sparsity'])}%",
            xy=(r["D"], r["I"]),
            xytext=(0, 9) if is_30_percent else (5, -12),
            textcoords="offset points",
            fontsize=8,
            ha="center" if is_30_percent else "left",
            va="bottom" if is_30_percent else "center",
        )


# ============================================================
# Highlight one ranking reversal
#
# At the same sparsity:
#
#     D_WANDA < D_DAP
#
# but
#
#     I_WANDA > I_DAP
#
# Here we use the 55% pair because the reversal is visually clearer.
# ============================================================

sparsity_to_highlight = 55

wanda = df[
    (df["Method"] == "WANDA")
    & (df["Sparsity"] == sparsity_to_highlight)
].iloc[0]

dap = df[
    (df["Method"] == "DAP")
    & (df["Sparsity"] == sparsity_to_highlight)
].iloc[0]


# Light connector between the two same-budget students
# ax.plot(
#     [wanda["D"], dap["D"]],
#     [wanda["I"], dap["I"]],
#     linestyle="--",
#     linewidth=1.3,
#     color="0.35",
#     alpha=0.8,
#     zorder=1,
# )


# Annotation pointing toward the reversal
mid_x = 0.5 * (wanda["D"] + dap["D"])
mid_y = 0.5 * (wanda["I"] + dap["I"])

# ax.annotate(
#     "ranking reversal",
#     xy=(mid_x, mid_y),
#     xytext=(-72, 28),
#     textcoords="offset points",
#     fontsize=8.5,
#     ha="center",
#     va="center",
#     arrowprops=dict(
#         arrowstyle="->",
#         linewidth=1.1,
#         color="0.25",
#     ),
# )


# ============================================================
# Legend
# ============================================================

method_handles = []

for method in methods:

    method_handles.append(
        Line2D(
            [0], [0],
            color=method_colors[method],
            marker=markers[method],
            linewidth=2.4,
            markersize=7.5,
            label=method,
        )
    )


ax.legend(
    handles=method_handles,
    loc="upper left",
    frameon=False,
    ncol=2,
    handlelength=1.6,
    columnspacing=0.9,
    labelspacing=0.35,
)


# ============================================================
# Axes labels
# ============================================================

ax.set_xlabel(
    r"Immediate damage $D$ (normalized)",
    fontsize=11,
    labelpad=7,
)

ax.set_ylabel(
    r"Remaining damage $I$",
    fontsize=11,
    labelpad=7,
)


# ============================================================
# Axis limits / ticks
# ============================================================

ax.set_xlim(0.45, 4.55)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

# Automatic value is fine, but explicitly setting it helps
# preserve identical dimensions across repeated plots.
max_i = df["I"].max()

ax.set_ylim(
    0.0,
    max_i * 1.12,
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=9,
    width=1.1,
    length=4.5,
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    True,
    alpha=0.20,
    linewidth=0.8,
)

ax.set_axisbelow(True)


# ============================================================
# Spines
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)


# ============================================================
# Optional panel title
#
# I would probably NOT put this inside the plot if your LaTeX
# subfigure caption already says it.
# ============================================================

# ax.set_title(
#     "Immediate damage alone is insufficient",
#     fontsize=11,
#     pad=8,
# )


# ============================================================
# Save
# ============================================================

pdf_out = "remaining_damage_vs_immediate_damage.pdf"

fig.savefig(
    pdf_out,
    bbox_inches="tight",
)

png_out = "remaining_damage_vs_immediate_damage.png"

fig.savefig(
    png_out,
    dpi=400,
    bbox_inches="tight",
)

plt.show()

print(pdf_out)
print(png_out)
