from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "WANDA",
    "Qwen3-4B",
    "Magnitude",
]

QWEN_ACCURACY = np.array([0.57, 0.489, 0.508])
QWEN_STD = np.array([0.014, 0.020, 0.017])

DEEPSEEK_ACCURACY = np.array([0.494, 0.490, 0.439])
DEEPSEEK_STD = np.array([0.017, 0.019, 0.018])


def main():
    positions = np.arange(len(METHODS))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        positions - bar_height / 2,
        QWEN_ACCURACY,
        height=bar_height,
        xerr=QWEN_STD,
        capsize=4,
        label="Qwen3-8B",
        color="tab:blue",
        alpha=0.85,
    )
    ax.barh(
        positions + bar_height / 2,
        DEEPSEEK_ACCURACY,
        height=bar_height,
        xerr=DEEPSEEK_STD,
        capsize=4,
        label="DeepSeek-R1-Distill-Llama-8B",
        color="tab:orange",
        alpha=0.85,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(METHODS, rotation=90, va="center", ha="center")
    ax.invert_yaxis()
    ax.set_xlim(0.4, 0.6)
    ax.set_xlabel("MATH-500 Accuracy", fontsize=13)
    ax.set_title("MATH-500 Accuracy by Teacher and Method", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_path = Path(__file__).with_name(
        "teacher_accuracy_comparison_horizontal_bars.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
