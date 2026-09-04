import pandas as pd
import matplotlib.pyplot as plt

data = [
    (30, "Magnitude", 2.30, 0.34),
    (30, "WANDA", 1.00, 0.72),
    (30, "WANDA++", 0.75, 0.68),
    (30, "SparseGPT", 0.82, 0.58),

    (35, "Magnitude", 3.00, 0.31),
    (35, "WANDA", 1.30, 0.70),
    (35, "WANDA++", 0.95, 0.66),
    (35, "SparseGPT", 1.05, 0.56),

    (40, "Magnitude", 3.90, 0.28),
    (40, "WANDA", 1.70, 0.67),
    (40, "WANDA++", 1.20, 0.63),
    (40, "SparseGPT", 1.35, 0.53),

    (45, "Magnitude", 5.00, 0.24),
    (45, "WANDA", 2.20, 0.64),
    (45, "WANDA++", 1.52, 0.60),
    (45, "SparseGPT", 1.75, 0.50),

    (50, "Magnitude", 6.40, 0.20),
    (50, "WANDA", 2.90, 0.60),
    (50, "WANDA++", 1.92, 0.56),
    (50, "SparseGPT", 2.30, 0.46),

    (55, "Magnitude", 8.00, 0.16),
    (55, "WANDA", 3.80, 0.55),
    (55, "WANDA++", 2.45, 0.51),
    (55, "SparseGPT", 3.00, 0.41),
]

df = pd.DataFrame(data, columns=["Sparsity", "Method", "D", "RecoveryFraction"])

methods = ["Magnitude", "WANDA", "WANDA++", "SparseGPT"]
markers = {"Magnitude": "o", "WANDA": "s", "WANDA++": "D", "SparseGPT": "^"}

fig, ax = plt.subplots(figsize=(8.8, 5.8))

for method in methods:
    g = df[df["Method"] == method].sort_values("Sparsity")
    ax.plot(
        g["D"], g["RecoveryFraction"],
        marker=markers[method],
        linewidth=1.8,
        markersize=7,
        label=method
    )
    for _, row in g.iterrows():
        ax.annotate(
            f'{int(row["Sparsity"])}%',
            (row["D"], row["RecoveryFraction"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

ax.set_xlabel(r"Immediate damage $D$ (normalized)", fontsize=12)
ax.set_ylabel(r"Fraction recovered $R/D$", fontsize=12)
ax.set_title("Recoverability vs. immediate pruning damage", loc="left",
             fontsize=13, fontweight="bold")

ax.set_ylim(0.12, 0.76)
ax.grid(True, alpha=0.22)
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# "better ←" cue beneath x-axis
ax.annotate(
    "better",
    xy=(0.08, -0.12), xycoords="axes fraction",
    xytext=(0.25, -0.12), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="<-", lw=1.2),
    ha="center", va="center", fontsize=10
)

fig.tight_layout()

out = "recoverability_vs_normalized_damage_wandapp.png"
fig.savefig(out, dpi=240, bbox_inches="tight")
plt.show()

print(f"Saved: {out}")
