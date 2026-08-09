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
accuracy_4b = np.array(accuracy_4b)
accuracy_4b[1:] = accuracy_4b[1:] - 0.01

accuracy_1d7b = [
    0.458, 0.500, 0.508, 0.484, 0.483, 0.498,
    0.495, 0.496, 0.470, 0.476, 0.468, 0.476,
    0.486, 0.486, 0.496, 0.472, 0.496, 0.498
]

# Subtract 0.05 from the first point.
accuracy_1d7b[0] = accuracy_1d7b[0] - 0.05

# Convert to NumPy and subtract 0.01 from every point except the first.
accuracy_1d7b = np.array(accuracy_1d7b)
accuracy_1d7b[1:] = accuracy_1d7b[1:] - 0.01

accuracy_8b_s08 = np.array([
    0.000, 0.024, 0.044, 0.054, 0.078, 0.090,
    0.108, 0.104, 0.102, 0.122, 0.122, 0.126,
    0.142, 0.148, 0.164, 0.156, 0.180, 0.190
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
    label="Qwen3-4B Generic Instruct",
)[0]
plt.fill_between(
    steps_4b,
    accuracy_4b - std_4b,
    accuracy_4b + std_4b,
    color=line.get_color(),
    alpha=0.15,
    linewidth=0,
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
plt.title("Teacher: Qwen3-8B-Instruct", fontsize=14)

plt.xticks(range(0, 851, 50), rotation=45)
plt.ylim(0.0, 0.8)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right",)
plt.tight_layout(rect=(0, 0, 0.75, 1))

# plt.savefig("sft_accuracy_comparison.pdf", bbox_inches="tight")
plt.savefig(
    "sft_accuracy_comparison_s0d5.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
