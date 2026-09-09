import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data
# Replace these with your actual results
# ============================================================

methods = ["WANDA", "DAP", "Magnitude"]

# Accuracy immediately after pruning
post_pruning = np.array([
    0.43,   # WANDA
    0.39,   # DAP
    0.36,   # Magnitude
])

# Accuracy after distillation
post_distillation = np.array([
    0.58,   # WANDA
    0.64,   # DAP
    0.60,   # Magnitude
])

x = np.arange(len(methods))


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(5.2, 4.5))


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


# ============================================================
# Axes
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13)

ax.set_ylabel("Accuracy", fontsize=14)

ax.tick_params(axis="y", labelsize=12)
ax.tick_params(axis="x", length=0)

ax.set_xlim(-0.55, len(methods) - 0.45)

# Adjust according to your real range
ax.set_ylim(0.30, 0.68)


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
    fontsize=11,
    loc="upper left",
)


plt.tight_layout()

# For LaTeX / paper
plt.savefig(
    "post_pruning_distillation.pdf",
    bbox_inches="tight",
)

plt.show()
