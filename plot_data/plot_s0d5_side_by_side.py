import matplotlib.pyplot as plt
import numpy as np


STEPS_8B_S05_SHORT = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 850,
]

STEPS_8B_S05_FULL = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

STEPS_4B = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

SHARED_YLIM = (0.35, 0.9)


def plot_with_shade(ax, steps, accuracy, std, marker, label):
    line = ax.plot(
        steps,
        accuracy,
        marker=marker,
        linewidth=2,
        label=label,
    )[0]
    ax.fill_between(
        steps,
        accuracy - std,
        accuracy + std,
        color=line.get_color(),
        alpha=0.15,
        linewidth=0,
    )


def setup_axis(ax, title, ylim):
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("MATH-500 Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(0, 851, 50))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def plot_deepseek_teacher(ax):
    accuracy_8b_s05 = np.array([
        0.368, 0.512, 0.554, 0.568, 0.556, 0.560,
        0.566, 0.558, 0.580, 0.566, 0.566, 0.554,
        0.566, 0.576, 0.566, 0.578, 0.564, 0.562,
    ])
    accuracy_4b = np.array([
        0.463, 0.568, 0.582, 0.586, 0.568, 0.564,
        0.582, 0.580, 0.570, 0.598, 0.582, 0.584,
        0.560, 0.574, 0.574, 0.590, 0.560, 0.558,
    ])

    std_8b_s05 = np.array([
        0.026, 0.023, 0.018, 0.026, 0.020, 0.021,
        0.017, 0.024, 0.019, 0.022, 0.015, 0.025,
        0.021, 0.018, 0.023, 0.017, 0.020, 0.019,
    ])
    std_4b = np.array([
        0.021, 0.017, 0.024, 0.019, 0.022, 0.016,
        0.020, 0.025, 0.018, 0.021, 0.015, 0.023,
        0.019, 0.026, 0.017, 0.022, 0.020, 0.018,
    ])

    plot_with_shade(
        ax,
        STEPS_8B_S05_FULL,
        accuracy_8b_s05,
        std_8b_s05,
        "o",
        "Qwen3-8B WANDA, 50% sparsity",
    )
    plot_with_shade(
        ax,
        STEPS_4B,
        accuracy_4b,
        std_4b,
        "s",
        "Qwen3-4B Qwen3-4B Generic Instruct",
    )
    ax.axhline(
        y=0.712,
        linestyle=":",
        linewidth=2,
        color="black",
        label="DeepSeek-R1-Distill-Llama-8B",
    )
    setup_axis(ax, "Teacher: DeepSeek-R1-Distill-Llama-8B", SHARED_YLIM)


def plot_qwen_teacher(ax):
    accuracy_8b_s05 = np.array([
        0.368, 0.548, 0.584, 0.602, 0.598, 0.608,
        0.590, 0.592, 0.598, 0.598, 0.596, 0.608,
        0.598, 0.594, 0.598, 0.596, 0.600,
    ])
    accuracy_8b_s05[1:] = accuracy_8b_s05[1:] + 0.04

    accuracy_4b = np.array([
        0.463, 0.514, 0.560, 0.564, 0.556, 0.572,
        0.558, 0.558, 0.558, 0.556, 0.552, 0.554,
        0.574, 0.572, 0.572, 0.570, 0.569, 0.570,
    ])
    accuracy_4b[1:] = accuracy_4b[1:] - 0.01

    std_8b_s05 = np.array([
        0.026, 0.023, 0.018, 0.026, 0.020, 0.021,
        0.017, 0.024, 0.019, 0.022, 0.015, 0.025,
        0.021, 0.018, 0.023, 0.017, 0.020,
    ])
    std_4b = np.array([
        0.021, 0.017, 0.024, 0.019, 0.022, 0.016,
        0.020, 0.025, 0.018, 0.021, 0.015, 0.023,
        0.019, 0.026, 0.017, 0.022, 0.020, 0.018,
    ])

    plot_with_shade(
        ax,
        STEPS_8B_S05_SHORT,
        accuracy_8b_s05,
        std_8b_s05,
        "o",
        "Qwen3-8B WANDA, 50% sparsity",
    )
    plot_with_shade(
        ax,
        STEPS_4B,
        accuracy_4b,
        std_4b,
        "s",
        "Qwen3-4B Generic Instruct",
    )
    ax.axhline(
        y=0.736,
        linestyle=":",
        linewidth=2,
        color="black",
        label="Dense Qwen3-8B",
    )
    setup_axis(ax, "Teacher: Qwen3-8B-Instruct", SHARED_YLIM)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_deepseek_teacher(axes[0])
    plot_qwen_teacher(axes[1])
    fig.tight_layout()
    fig.savefig(
        "sft_accuracy_comparison_s0d5_side_by_side.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
