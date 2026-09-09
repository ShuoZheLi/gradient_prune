import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


# ============================================================
# Data
# Replace these with your actual results
# ============================================================

methods = [
    "Generic\nInstruct 4B",
    "Magnitude",
    "SparseGPT",
    "WANDA",
    "WANDA++",
    "DAP",
]

# Accuracy immediately after pruning
post_pruning = np.array([
    0.393,
    0.004,
    0.325,
    0.298,
    0.345,
    0.245,
])

post_pruning_std = np.array([
    0.021,
    0.008,
    0.024,
    0.026,
    0.024,
    0.027,
])

# Accuracy after distillation
post_distillation = np.array([
    0.489,
    0.508,
    0.557,
    0.570,
    0.580,
    0.625,
])

post_distillation_std = np.array([
    0.020,
    0.017,
    0.020,
    0.020,
    0.019,
    0.018,
])

x = np.arange(len(methods))


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(4.2, 3.4),
    constrained_layout=True,
)


# ------------------------------------------------------------
# Connecting lines
# ------------------------------------------------------------

for i in range(len(methods)):
    ax.plot(
        [x[i], x[i]],
        [post_pruning[i], post_distillation[i]],
        linestyle="--",
        linewidth=1.8,
        color="0.55",
        zorder=1,
    )


# ------------------------------------------------------------
# Post-pruning points
# ------------------------------------------------------------

ax.scatter(
    x,
    post_pruning,
    s=100,
    marker="o",
    color="0.25",
    label="Post-pruning",
    zorder=3,
)

ax.errorbar(
    x,
    post_pruning,
    yerr=post_pruning_std,
    fmt="none",
    ecolor="0.25",
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    zorder=2,
)


# ------------------------------------------------------------
# Post-distillation points
# ------------------------------------------------------------

ax.scatter(
    x,
    post_distillation,
    s=110,
    marker="D",
    color="black",
    label="Post-distillation",
    zorder=3,
)

ax.errorbar(
    x,
    post_distillation,
    yerr=post_distillation_std,
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    zorder=2,
)


# ============================================================
# Axes
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9, rotation=18, ha="right")

label_shift = ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans)
for label in ax.get_xticklabels()[1:]:
    label.set_transform(label.get_transform() + label_shift)

ax.set_ylabel("Accuracy", fontsize=11)

ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9, length=0)

ax.set_xlim(-0.55, len(methods) - 0.45)

# Adjust according to your real range
ax.set_ylim(0.20, 0.68)


# ============================================================
# Styling
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    alpha=0.35,
)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper left",
)


# For LaTeX / paper
plt.savefig(
    "intro_f1_1.pdf",
)

plt.show()
