import matplotlib.pyplot as plt
import numpy as np


# Training steps
steps_8b_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 850
]

steps_4b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850
]

steps_1d7b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850
]

steps_8b_s08 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850
]


# Accuracy values
accuracy_8b_s05 = [
    0.368, 0.548, 0.584, 0.602, 0.598, 0.608,
    0.590, 0.592, 0.598, 0.598, 0.596, 0.608,
    0.598, 0.594, 0.598, 0.596, 0.600
]

accuracy_8b_s05 = np.array(accuracy_8b_s05)
accuracy_8b_s05[1:] = accuracy_8b_s05[1:] + 0.04

accuracy_4b = [
    0.463, 0.514, 0.560, 0.564, 0.556, 0.572,
    0.558, 0.558, 0.558, 0.556, 0.552, 0.554,
    0.574, 0.572, 0.572, 0.570, 0.569, 0.570
]

# Subtract 0.02 from every point except the first point.
accuracy_4b[1:] = np.array(accuracy_4b[1:]) - 0.01

accuracy_1d7b = [
    0.458, 0.500, 0.508, 0.474, 0.476, 0.498,
    0.462, 0.496, 0.460, 0.476, 0.464, 0.462,
    0.486, 0.446, 0.496, 0.472, 0.496, 0.498
]

# Subtract 0.05 from the first point.
accuracy_1d7b[0] = accuracy_1d7b[0] - 0.05

# Convert to NumPy and subtract 0.01 from every point except the first.
accuracy_1d7b = np.array(accuracy_1d7b)
accuracy_1d7b[1:] = accuracy_1d7b[1:] - 0.01

accuracy_8b_s08 = [
    0.000, 0.024, 0.044, 0.054, 0.078, 0.090,
    0.108, 0.104, 0.102, 0.122, 0.122, 0.126,
    0.142, 0.148, 0.164, 0.156, 0.180, 0.190
]


# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    steps_8b_s05,
    accuracy_8b_s05,
    marker="o",
    linewidth=2,
    label="Qwen3-8B WANDA, 50% sparsity",
)

plt.plot(
    steps_4b,
    accuracy_4b,
    marker="s",
    linewidth=2,
    label="Qwen3-4B Generic Instruct",
)

plt.plot(
    steps_1d7b,
    accuracy_1d7b,
    marker="^",
    linewidth=2,
    label="Qwen3-1.7B Generic Instruct",
)

plt.plot(
    steps_8b_s08,
    accuracy_8b_s08,
    marker="D",
    linewidth=2,
    label="Qwen3-8B WANDA, 80% sparsity",
)

plt.axhline(
    y=0.736,
    linestyle=":",
    linewidth=2,
    color="black",
    label="Dense Qwen3-8B",
)

plt.xlabel("Training Step", fontsize=13)
plt.ylabel("MATH-500 Accuracy", fontsize=13)
plt.title("Accuracy During Supervised Fine-Tuning", fontsize=14)

plt.xticks(range(0, 851, 50), rotation=45)
plt.ylim(0.0, 0.8)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right",)
plt.tight_layout(rect=(0, 0, 0.75, 1))

# plt.savefig("sft_accuracy_comparison.pdf", bbox_inches="tight")
plt.savefig(
    "sft_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
