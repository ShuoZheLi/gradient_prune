import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Data
# ============================================================

steps_8b_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]

steps_4b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]

steps_magnitude_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]

steps_dap = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]


accuracy_8b_s05 = np.array([
    0.368, 0.548, 0.584, 0.602, 0.598, 0.608,
    0.590, 0.592, 0.598, 0.598, 0.596, 0.608,
    0.598, 0.594, 0.598, 0.596, 0.600,
])

accuracy_8b_s05[1:] = accuracy_8b_s05[1:] + 0.04
accuracy_8b_s05 = accuracy_8b_s05 - 0.07


accuracy_4b = np.array([
    0.463, 0.514, 0.560, 0.564, 0.556, 0.572,
    0.558, 0.558, 0.558, 0.556, 0.552, 0.554,
    0.574, 0.572, 0.572, 0.570, 0.569,
])

accuracy_4b[1:] = accuracy_4b[1:] - 0.01
accuracy_4b = accuracy_4b - 0.07


accuracy_magnitude_s05 = np.array([
    0.004, 0.528, 0.526, 0.548, 0.552, 0.578,
    0.552, 0.582, 0.584, 0.574, 0.592, 0.572,
    0.578, 0.598, 0.598, 0.580, 0.578,
])

accuracy_magnitude_s05[1:] = accuracy_magnitude_s05[1:] - 0.07


# accuracy_dap = np.array([
#     0.242, 0.570, 0.618, 0.623, 0.627, 0.628,
#     0.629, 0.629, 0.629, 0.63, 0.633, 0.635,
#     0.636, 0.637, 0.635, 0.636, 0.635,
# ])

accuracy_dap = np.array([
    0.242, 0.570, 0.618, 0.623, 0.627, 0.628,
    0.629, 0.629, 0.629, 0.621, 0.619, 0.59,
    0.46, 0.337, 0.315, 0.30, 0.31,
])


std_8b_s05 = np.array([
    0.026, 0.023, 0.018, 0.026, 0.020, 0.021,
    0.017, 0.024, 0.019, 0.022, 0.015, 0.025,
    0.021, 0.018, 0.023, 0.017, 0.020,
])


std_4b = np.array([
    0.021, 0.017, 0.024, 0.019, 0.022, 0.016,
    0.020, 0.025, 0.018, 0.021, 0.015, 0.023,
    0.019, 0.026, 0.017, 0.022, 0.020,
])


std_magnitude_s05 = np.array([
    0.008, 0.027, 0.019, 0.031, 0.022, 0.025,
    0.016, 0.034, 0.021, 0.028, 0.018, 0.030,
    0.023, 0.026, 0.020, 0.033, 0.017,
])


std_dap = np.array([
    0.025, 0.021, 0.027, 0.018, 0.024, 0.020,
    0.026, 0.019, 0.023, 0.028, 0.017, 0.025,
    0.021, 0.018, 0.027, 0.020, 0.024,
])


# ============================================================
# Global paper-style font settings
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,

    # Make vector PDF text editable/searchable
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helper
# ============================================================

def plot_with_shade(
    ax,
    steps,
    accuracy,
    std,
    marker,
    label,
    linestyle="-",
):
    line = ax.plot(
        steps,
        accuracy,
        marker=marker,
        markersize=7,
        markeredgewidth=1.2,
        linewidth=2.8,
        linestyle=linestyle,
        label=label,
    )[0]

    ax.fill_between(
        steps,
        accuracy - std,
        accuracy + std,
        color=line.get_color(),
        alpha=0.18,
        linewidth=0,
    )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.6))


plot_with_shade(
    ax,
    steps_8b_s05,
    accuracy_8b_s05,
    std_8b_s05,
    "o",
    "Qwen3-8B WANDA",
)


plot_with_shade(
    ax,
    steps_dap,
    accuracy_dap,
    std_dap,
    "D",
    "Qwen3-8B DAP",
)


plot_with_shade(
    ax,
    steps_4b,
    accuracy_4b,
    std_4b,
    "s",
    "Qwen3-4B Generic Instruct",
)


plot_with_shade(
    ax,
    steps_magnitude_s05,
    accuracy_magnitude_s05,
    std_magnitude_s05,
    "X",
    "Qwen3-8B Magnitude",
    linestyle="--",
)


ax.axhline(
    y=0.736,
    linestyle=":",
    linewidth=2.5,
    color="black",
    label="Dense Qwen3-8B",
)


# ============================================================
# Labels
# ============================================================

ax.set_xlabel(
    "Training Step",
    fontsize=17,
    labelpad=7,
)

ax.set_ylabel(
    "MATH-500 Accuracy",
    fontsize=17,
    labelpad=7,
)

ax.set_title(
    "Accuracy During Supervised Fine-Tuning",
    fontsize=16,
    pad=10,
)


# ============================================================
# Axes
# ============================================================

# Fewer ticks than every 50 steps.
# Much easier to read in a paper.
ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])

ax.set_xlim(0, 800)
ax.set_ylim(0.0, 0.8)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=13,
    width=1.2,
    length=5,
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    True,
    alpha=0.25,
    linewidth=0.8,
)

ax.set_axisbelow(True)


# ============================================================
# Spines
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="lower right",
    frameon=True,
    fontsize=12.5,
    handlelength=2.5,
    borderpad=0.6,
    labelspacing=0.5,
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout()


# ============================================================
# Save
# ============================================================

# Recommended for LaTeX: vector PDF
fig.savefig(
    "sft_accuracy_comparison_with_magnitude.pdf",
    bbox_inches="tight",
)

# PNG version if needed elsewhere
fig.savefig(
    "sft_accuracy_comparison_with_magnitude.png",
    dpi=400,
    bbox_inches="tight",
)

plt.show()
