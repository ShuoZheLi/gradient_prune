from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Data
# ============================================================

METHODS = [
    "WANDA",
    "Qwen3-4B",
    "Magnitude",
]

QWEN_ACCURACY = np.array([
    0.570,
    0.489,
    0.508,
])

QWEN_STD = np.array([
    0.014,
    0.020,
    0.017,
])

DEEPSEEK_ACCURACY = np.array([
    0.494,
    0.490,
    0.439,
])

DEEPSEEK_STD = np.array([
    0.017,
    0.019,
    0.018,
])


# ============================================================
# Paper-style matplotlib settings
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():

    positions = np.arange(len(METHODS))
    bar_height = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.2))


    # ========================================================
    # Bars
    # ========================================================

    ax.barh(
        positions - bar_height / 2,
        QWEN_ACCURACY,
        height=bar_height,
        xerr=QWEN_STD,
        capsize=5,
        error_kw={
            "elinewidth": 1.8,
            "capthick": 1.8,
        },
        label="Qwen3-8B",
        color="tab:blue",
        alpha=0.85,
    )

    ax.barh(
        positions + bar_height / 2,
        DEEPSEEK_ACCURACY,
        height=bar_height,
        xerr=DEEPSEEK_STD,
        capsize=5,
        error_kw={
            "elinewidth": 1.8,
            "capthick": 1.8,
        },
        label="DeepSeek-R1-Llama-8B",
        color="tab:orange",
        alpha=0.85,
    )


    # ========================================================
    # Y axis
    # ========================================================

    ax.set_yticks(positions)

    ax.set_yticklabels(
        METHODS,
        fontsize=12,
        rotation=90,
        rotation_mode="anchor",
        va="center",
        ha="center",
    )

    ax.invert_yaxis()

    ax.tick_params(
        axis="y",
        width=1.2,
        length=0,
        pad=6,
    )


    # ========================================================
    # X axis
    # ========================================================

    ax.set_xlim(0.40, 0.60)

    ax.set_xticks([
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ])

    ax.set_xlabel(
        "MATH-500 Accuracy",
        fontsize=16,
        labelpad=7,
    )

    ax.tick_params(
        axis="x",
        labelsize=13,
        width=1.2,
        length=5,
    )


    # ========================================================
    # Title
    # ========================================================

    ax.set_title(
        "MATH-500 Accuracy by Teacher and Method",
        fontsize=16,
        pad=10,
    )


    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        axis="x",
        alpha=0.25,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)


    # ========================================================
    # Spines
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="right",
        fontsize=11.5,
        frameon=True,
        handlelength=2.0,
        borderpad=0.6,
        labelspacing=0.5,
    )


    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout()


    # ========================================================
    # Save
    # ========================================================

    output_dir = Path(__file__).parent

    pdf_path = output_dir / (
        "teacher_accuracy_comparison_horizontal_bars.pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    png_path = output_dir / (
        "teacher_accuracy_comparison_horizontal_bars.png"
    )

    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()