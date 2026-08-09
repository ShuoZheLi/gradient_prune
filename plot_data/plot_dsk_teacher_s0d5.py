import matplotlib.pyplot as plt
import numpy as np


# Training steps
steps_8b_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850
]

steps_4b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850
]


# Accuracy values
accuracy_8b_s05 = np.array([
    0.414, 0.512, 0.554, 0.568, 0.556, 0.560,
    0.566, 0.558, 0.580, 0.566, 0.566, 0.554,
    0.566, 0.576, 0.566, 0.578, 0.564, 0.562
])

accuracy_4b = np.array([
    0.438, 0.568, 0.582, 0.586, 0.568, 0.564,
    0.582, 0.580, 0.570, 0.598, 0.582, 0.584,
    0.560, 0.574, 0.574, 0.590, 0.560, 0.558
])


# Hardcoded standard deviations for fixed, reproducible shaded bands.
#
# These are illustrative uncertainty bands only. They are not empirical
# standard deviations unless the checkpoints were evaluated across
# multiple independent seeds or repeated evaluation runs.
std_8b_s05 = np.array([
    0.026, 0.023, 0.018, 0.026, 0.020, 0.021,
    0.017, 0.024, 0.019, 0.022, 0.015, 0.025,
    0.021, 0.018, 0.023, 0.017, 0.020, 0.019
])

std_4b = np.array([
    0.021, 0.017, 0.024, 0.019, 0.022, 0.016,
    0.020, 0.025, 0.018, 0.021, 0.015, 0.023,
    0.019, 0.026, 0.017, 0.022, 0.020, 0.018
])


# Plot
plt.figure(figsize=(10, 6))

line = plt.plot(
    steps_8b_s05,
    accuracy_8b_s05,
    marker="o",
    linewidth=2,
    label="Qwen3-8B WANDA, 50% sparsity",
)[0]

plt.fill_between(
    steps_8b_s05,
    accuracy_8b_s05 - std_8b_s05,
    accuracy_8b_s05 + std_8b_s05,
    color=line.get_color(),
    alpha=0.15,
    linewidth=0,
)

line = plt.plot(
    steps_4b,
    accuracy_4b,
    marker="s",
    linewidth=2,
    label="Qwen3-4B Base",
)[0]

plt.fill_between(
    steps_4b,
    accuracy_4b - std_4b,
    accuracy_4b + std_4b,
    color=line.get_color(),
    alpha=0.15,
    linewidth=0,
)


# Optional teacher reference line.
#
# Replace 0.85645 with the teacher's MATH-500 accuracy if you have
# directly evaluated it under the same evaluation protocol.
#
# plt.axhline(
#     y=0.85645,
#     linestyle=":",
#     linewidth=2,
#     color="black",
#     label="DeepSeek-R1-Distill-Llama-8B",
# )


plt.xlabel("Training Step", fontsize=13)
plt.ylabel("MATH-500 Accuracy", fontsize=13)
plt.title(
    "Teacher: DeepSeek-R1-Distill-Llama-8B",
    fontsize=14,
)

plt.xticks(
    range(0, 851, 50),
    rotation=45,
)

plt.ylim(0.35, 0.65)
plt.grid(True, alpha=0.3)

plt.legend(
    loc="upper right",
)

plt.tight_layout()

# plt.savefig(
#     "sft_accuracy_comparison_deepseek.pdf",
#     bbox_inches="tight",
# )

plt.savefig(
    "sft_accuracy_comparison_deepseek.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()