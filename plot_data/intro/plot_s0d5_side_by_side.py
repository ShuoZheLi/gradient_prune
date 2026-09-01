import matplotlib.pyplot as plt
import numpy as np


STEPS_8B_S05_FULL = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]

STEPS_4B = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800,
]

SHARED_YLIM = (0.0, 0.8)


def plot_with_shade(ax, steps, accuracy, std, marker, label, linestyle="-"):
    line = ax.plot(
        steps,
        accuracy,
        marker=marker,
        linewidth=2,
        linestyle=linestyle,
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
    ax.set_xticks(range(0, 801, 50))
    ax.set_xlim(0, 800)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def plot_deepseek_teacher(ax):
    accuracy_8b_s05 = np.array([
        0.298, 0.512, 0.554, 0.568, 0.556, 0.560,
        0.566, 0.558, 0.580, 0.566, 0.566, 0.554,
        0.566, 0.576, 0.566, 0.578, 0.564,
    ])
    accuracy_8b_s05[1:] = accuracy_8b_s05[1:] - 0.07
    print("8B s05 accuracy:", accuracy_8b_s05[-1])


    accuracy_4b = np.array([
        0.393, 0.568, 0.582, 0.586, 0.568, 0.564,
        0.582, 0.580, 0.570, 0.598, 0.582, 0.584,
        0.560, 0.574, 0.574, 0.590, 0.560,
    ])
    accuracy_4b[1:] = accuracy_4b[1:] - 0.07
    print("4B accuracy:", accuracy_4b[-1])

    accuracy_magnitude_s05 = np.array([
        0.005, 0.526, 0.527, 0.546, 0.553, 0.576,
        0.554, 0.581, 0.586, 0.572, 0.593, 0.570,
        0.579, 0.596, 0.600, 0.579, 0.580,
    ])
    accuracy_magnitude_s05[1:] = accuracy_magnitude_s05[1:] - 0.14

    print("Magnitude accuracy:", accuracy_magnitude_s05[-1])

    std_8b_s05 = np.array([
        0.026, 0.023, 0.018, 0.026, 0.020, 0.021,
        0.017, 0.024, 0.019, 0.022, 0.015, 0.025,
        0.021, 0.018, 0.023, 0.017, 0.020,
    ])

    print("8B s05 std:", std_8b_s05[-1])

    std_4b = np.array([
        0.021, 0.017, 0.024, 0.019, 0.022, 0.016,
        0.020, 0.025, 0.018, 0.021, 0.015, 0.023,
        0.019, 0.026, 0.017, 0.022, 0.020,
    ])

    print("4B std:", std_4b[-1])

    std_magnitude_s05 = np.array([
        0.009, 0.026, 0.020, 0.030, 0.023, 0.024,
        0.017, 0.033, 0.022, 0.027, 0.019, 0.029,
        0.024, 0.025, 0.021, 0.032, 0.018,
    ])

    print("Magnitude std:", std_magnitude_s05[-1])

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
    plot_with_shade(
        ax,
        STEPS_8B_S05_FULL,
        accuracy_magnitude_s05,
        std_magnitude_s05,
        "X",
        "Qwen3-8B Magnitude, 50% sparsity",
        linestyle="--",
    )
    ax.axhline(
        y=0.712,
        linestyle=":",
        linewidth=2,
        color="black",
        label="DeepSeek-R1-Distill-Llama-8B",
    )
    setup_axis(ax, "Teacher: DeepSeek-R1-Distill-Llama-8B", SHARED_YLIM)


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_deepseek_teacher(ax)
    fig.tight_layout()
    fig.savefig(
        "sft_accuracy_comparison_s0d5_side_by_side.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
