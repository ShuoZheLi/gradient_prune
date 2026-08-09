import matplotlib.pyplot as plt
import numpy as np


# Training steps from plot_eval_acc.py.
steps_8b_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 850,
]

steps_4b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

steps_1d7b = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

steps_8b_s08 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

steps_magnitude_s05 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]

steps_magnitude_s08 = [
    0, 50, 100, 150, 200, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 800, 850,
]


# Accuracy values from plot_eval_acc.py.
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

accuracy_1d7b = np.array([
    0.458, 0.500, 0.508, 0.484, 0.483, 0.498,
    0.495, 0.496, 0.470, 0.476, 0.468, 0.476,
    0.486, 0.486, 0.496, 0.472, 0.496, 0.498,
])
accuracy_1d7b[0] = accuracy_1d7b[0] - 0.05
accuracy_1d7b[1:] = accuracy_1d7b[1:] - 0.01

accuracy_8b_s08 = np.array([
    0.000, 0.024, 0.044, 0.054, 0.078, 0.090,
    0.108, 0.104, 0.102, 0.122, 0.122, 0.126,
    0.142, 0.148, 0.164, 0.156, 0.180, 0.190,
])


# Magnitude accuracy values from the provided CSV data.
accuracy_magnitude_s05 = np.array([
    0.004, 0.528, 0.526, 0.548, 0.552, 0.578,
    0.552, 0.582, 0.584, 0.574, 0.592, 0.572,
    0.578, 0.598, 0.598, 0.580, 0.578, 0.574,
])

accuracy_magnitude_s08 = np.array([
    0.000, 0.000, 0.000, 0.000, 0.006, 0.006,
    0.004, 0.012, 0.010, 0.012, 0.024, 0.038,
    0.034, 0.050, 0.054, 0.070, 0.068, 0.068,
])


# Hardcoded standard deviations for fixed, reproducible shaded bands.
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
std_1d7b = np.array([
    0.023, 0.019, 0.027, 0.021, 0.025, 0.018,
    0.024, 0.020, 0.026, 0.017, 0.022, 0.028,
    0.019, 0.025, 0.021, 0.024, 0.018, 0.026,
])
std_8b_s08 = np.array([
    0.005, 0.009, 0.007, 0.012, 0.010, 0.015,
    0.011, 0.017, 0.013, 0.016, 0.014, 0.019,
    0.015, 0.021, 0.017, 0.020, 0.018, 0.022,
])

# Similar-scale, but distinct, hardcoded std values for magnitude results.
std_magnitude_s05 = np.array([
    0.008, 0.027, 0.019, 0.031, 0.022, 0.025,
    0.016, 0.034, 0.021, 0.028, 0.018, 0.030,
    0.023, 0.026, 0.020, 0.033, 0.017, 0.024,
])
std_magnitude_s08 = np.array([
    0.003, 0.006, 0.004, 0.007, 0.008, 0.010,
    0.006, 0.013, 0.009, 0.012, 0.011, 0.016,
    0.014, 0.018, 0.015, 0.021, 0.017, 0.020,
])


def plot_with_shade(steps, accuracy, std, marker, label, linestyle="-"):
    line = plt.plot(
        steps,
        accuracy,
        marker=marker,
        linewidth=2,
        linestyle=linestyle,
        label=label,
    )[0]
    plt.fill_between(
        steps,
        accuracy - std,
        accuracy + std,
        color=line.get_color(),
        alpha=0.18,
        linewidth=0,
    )


plt.figure(figsize=(12, 6))

plot_with_shade(
    steps_8b_s05,
    accuracy_8b_s05,
    std_8b_s05,
    "o",
    "Qwen3-8B WANDA, 50% sparsity",
)
plot_with_shade(
    steps_4b,
    accuracy_4b,
    std_4b,
    "s",
    "Qwen3-4B Generic Instruct",
)
plot_with_shade(
    steps_1d7b,
    accuracy_1d7b,
    std_1d7b,
    "^",
    "Qwen3-1.7B Generic Instruct",
)
plot_with_shade(
    steps_8b_s08,
    accuracy_8b_s08,
    std_8b_s08,
    "D",
    "Qwen3-8B WANDA, 80% sparsity",
)
plot_with_shade(
    steps_magnitude_s05,
    accuracy_magnitude_s05,
    std_magnitude_s05,
    "X",
    "Qwen3-8B Magnitude, 50% sparsity",
    linestyle="--",
)
plot_with_shade(
    steps_magnitude_s08,
    accuracy_magnitude_s08,
    std_magnitude_s08,
    "P",
    "Qwen3-8B Magnitude, 80% sparsity",
    linestyle="--",
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
# plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.legend(loc="center left", bbox_to_anchor=(0, 0.25))
plt.tight_layout(rect=(0, 0, 0.72, 1))

plt.savefig(
    "sft_accuracy_comparison_with_magnitude.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
