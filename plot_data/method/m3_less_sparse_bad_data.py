import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Data
# ============================================================

rows = [
    (30, "Magnitude", 2.30, 0.30, 0.34),
    (30, "WANDA", 1.00, 0.67, 0.72),
    (30, "WANDA++", 0.75, 0.64, 0.68),
    (30, "SparseGPT", 0.82, 0.55, 0.58),

    (35, "Magnitude", 3.00, 0.28, 0.31),
    (35, "WANDA", 1.30, 0.65, 0.70),
    (35, "WANDA++", 0.95, 0.62, 0.66),
    (35, "SparseGPT", 1.05, 0.53, 0.56),

    (40, "Magnitude", 3.90, 0.26, 0.28),
    (40, "WANDA", 1.70, 0.62, 0.67),
    (40, "WANDA++", 1.20, 0.60, 0.63),
    (40, "SparseGPT", 1.35, 0.50, 0.53),

    (45, "Magnitude", 5.00, 0.23, 0.24),
    (45, "WANDA", 2.20, 0.59, 0.64),
    (45, "WANDA++", 1.52, 0.56, 0.60),
    (45, "SparseGPT", 1.75, 0.47, 0.50),

    (50, "Magnitude", 6.40, 0.20, 0.20),
    (50, "WANDA", 2.90, 0.55, 0.60),
    (50, "WANDA++", 1.92, 0.53, 0.56),
    (50, "SparseGPT", 2.30, 0.43, 0.46),

    (55, "Magnitude", 8.00, 0.17, 0.16),
    (55, "WANDA", 3.80, 0.51, 0.55),
    (55, "WANDA++", 2.45, 0.48, 0.51),
    (55, "SparseGPT", 3.00, 0.40, 0.41),
]

df = pd.DataFrame(
    rows,
    columns=["Sparsity", "Method", "D", "Observed", "Predicted"],
)

df = df[df["D"] <= 4]


# ============================================================
# Methods / markers
# ============================================================

methods = [
    "Magnitude",
    "WANDA",
    "WANDA++",
    "SparseGPT",
]

markers = {
    "Magnitude": "o",
    "WANDA": "s",
    "WANDA++": "D",
    "SparseGPT": "^",
}


# ============================================================
# Paper-style matplotlib settings
# ============================================================

plt.rcParams.update({
    "font.size": 13.5,

    "axes.labelsize": 16,
    "axes.titlesize": 15,

    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,

    "legend.fontsize": 11.5,
    "legend.title_fontsize": 11.5,

    # Better PDF font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)


# ============================================================
# Plot trajectories
# ============================================================

for method in methods:

    g = (
        df[df["Method"] == method]
        .sort_values("Sparsity")
    )

    # ----------------------------
    # Observed
    # ----------------------------

    obs_line, = ax.plot(
        g["D"],
        g["Observed"],
        marker=markers[method],
        linewidth=2.6,
        markersize=8.5,
        markeredgewidth=1.2,
        label=method,
    )

    c = obs_line.get_color()

    # ----------------------------
    # Predicted
    # ----------------------------

    ax.plot(
        g["D"],
        g["Predicted"],
        marker=markers[method],
        linewidth=2.1,
        markersize=8.0,
        markeredgewidth=1.3,
        linestyle="--",
        markerfacecolor="white",
        markeredgecolor=c,
        color=c,
    )

    # ----------------------------
    # Correspondence connectors
    # ----------------------------

    for _, r in g.iterrows():

        ax.plot(
            [r["D"], r["D"]],
            [r["Observed"], r["Predicted"]],
            linewidth=1.0,
            alpha=0.28,
            color=c,
        )


# ============================================================
# Sparsity labels
# ============================================================

for method in methods:

    g = (
        df[df["Method"] == method]
        .sort_values("Sparsity")
    )

    for _, r in g.iloc[[0, -1]].iterrows():

        ax.annotate(
            f"{int(r['Sparsity'])}%",
            xy=(r["D"], r["Observed"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=10.0,
            ha="left",
            va="center",
        )


# ============================================================
# Method legend
# ============================================================

method_handles = []

for method in methods:

    line = next(
        l for l in ax.get_lines()
        if l.get_label() == method
    )

    method_handles.append(
        Line2D(
            [0], [0],
            color=line.get_color(),
            marker=markers[method],
            linewidth=2.4,
            markersize=7.5,
            label=method,
        )
    )


leg1 = ax.legend(
    handles=method_handles,
    loc="upper right",
    frameon=False,
    title="",
    handlelength=2.2,
    labelspacing=0.45,
)

ax.add_artist(leg1)


# ============================================================
# Style legend
# ============================================================

style_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label=r"Observed $R/D$",
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        linewidth=2.0,
        markersize=7,
        label=r"Predicted $\widehat{R}/\widehat{D}$",
    ),
]

ax.legend(
    handles=style_handles,
    loc="lower left",
    frameon=False,
    handlelength=2.4,
    labelspacing=0.45,
)


# ============================================================
# Axes labels
# ============================================================

ax.set_xlabel(
    r"Immediate damage $D$ (normalized)",
    fontsize=16,
    labelpad=7,
)

ax.set_ylabel(
    "Recoverability",
    fontsize=16,
    labelpad=7,
)


# ============================================================
# Title
# ============================================================

ax.set_title(
    "Observed vs. predicted damage–recoverability landscape",
    loc="left",
    fontsize=15,
    fontweight="bold",
    pad=9,
)


# ============================================================
# Axis limits / ticks
# ============================================================

ax.set_xlim(0.45, 4.0)
ax.set_ylim(0.12, 0.76)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=12.5,
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
# "better" annotation
# ============================================================

# ax.annotate(
#     "better",
#     xy=(0.06, -0.12),
#     xycoords="axes fraction",
#     xytext=(0.22, -0.12),
#     textcoords="axes fraction",
#     arrowprops=dict(
#         arrowstyle="<-",
#         lw=1.4,
#     ),
#     ha="center",
#     va="center",
#     fontsize=11.5,
# )


# ============================================================
# Layout
# ============================================================

fig.tight_layout()


# ============================================================
# Save
# ============================================================

# Use this in LaTeX
pdf_out = "figure_AB_overlaid_observed_predicted.pdf"

fig.savefig(
    pdf_out,
    bbox_inches="tight",
)


# Optional PNG preview
png_out = "figure_AB_overlaid_observed_predicted.png"

fig.savefig(
    png_out,
    dpi=400,
    bbox_inches="tight",
)


plt.show()

print(pdf_out)
print(png_out)
