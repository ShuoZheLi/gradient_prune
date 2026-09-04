import pandas as pd
import matplotlib.pyplot as plt

# Same D for observed and predicted, as requested.
rows = [
    (30,"Magnitude",2.30,0.30,0.34),(30,"WANDA",1.00,0.67,0.72),(30,"WANDA++",0.75,0.64,0.68),(30,"SparseGPT",0.82,0.55,0.58),
    (35,"Magnitude",3.00,0.28,0.31),(35,"WANDA",1.30,0.65,0.70),(35,"WANDA++",0.95,0.62,0.66),(35,"SparseGPT",1.05,0.53,0.56),
    (40,"Magnitude",3.90,0.26,0.28),(40,"WANDA",1.70,0.62,0.67),(40,"WANDA++",1.20,0.60,0.63),(40,"SparseGPT",1.35,0.50,0.53),
    (45,"Magnitude",5.00,0.23,0.24),(45,"WANDA",2.20,0.59,0.64),(45,"WANDA++",1.52,0.56,0.60),(45,"SparseGPT",1.75,0.47,0.50),
    (50,"Magnitude",6.40,0.20,0.20),(50,"WANDA",2.90,0.55,0.60),(50,"WANDA++",1.92,0.53,0.56),(50,"SparseGPT",2.30,0.43,0.46),
    (55,"Magnitude",8.00,0.17,0.16),(55,"WANDA",3.80,0.51,0.55),(55,"WANDA++",2.45,0.48,0.51),(55,"SparseGPT",3.00,0.40,0.41),
]
df = pd.DataFrame(rows, columns=["Sparsity","Method","D","Observed","Predicted"])
df = df[df["D"] <= 4]

methods = ["Magnitude","WANDA","WANDA++","SparseGPT"]
markers = {"Magnitude":"o","WANDA":"s","WANDA++":"D","SparseGPT":"^"}

fig, ax = plt.subplots(figsize=(9.2,6.2))

# Each method has one color (automatic). Solid = observed, dashed/open = predicted.
for method in methods:
    g = df[df.Method == method].sort_values("Sparsity")
    
    # Get one automatic color from the observed trajectory.
    obs_line, = ax.plot(
        g.D, g.Observed,
        marker=markers[method], linewidth=2.0, markersize=7,
        label=method
    )
    c = obs_line.get_color()
    
    # Predicted: same method color, dashed, hollow marker.
    ax.plot(
        g.D, g.Predicted,
        marker=markers[method], linewidth=1.6, markersize=7,
        linestyle="--", markerfacecolor="white", markeredgecolor=c,
        color=c
    )
    
    # Thin vertical correspondence at identical D.
    for _, r in g.iterrows():
        ax.plot([r.D, r.D], [r.Observed, r.Predicted],
                linewidth=0.8, alpha=0.25, color=c)

# Sparsity labels only once per configuration, next to observed points.
for _, r in df.iterrows():
    ax.annotate(f"{int(r.Sparsity)}%", (r.D, r.Observed),
                xytext=(5,-11), textcoords="offset points", fontsize=7.5)

# Style legend: observed vs predicted, independent of method legend.
style_obs, = ax.plot([], [], linestyle="-", marker="o", label="Observed $R/D$")
style_pred, = ax.plot([], [], linestyle="--", marker="o",
                      markerfacecolor="white",
                      label=r"Predicted $\widehat{R}/\widehat{D}$")

method_legend = ax.legend(
    handles=ax.lines[:0], frameon=False
)

# Build method handles cleanly
from matplotlib.lines import Line2D
method_handles = []
for method in methods:
    # retrieve color from first plotted observed line for method:
    # lines are grouped with vertical connectors, so find via labels
    line = next(l for l in ax.get_lines() if l.get_label() == method)
    method_handles.append(Line2D([0],[0], color=line.get_color(),
                                 marker=markers[method], linewidth=2,
                                 label=method))
leg1 = ax.legend(handles=method_handles, loc="upper right",
                 frameon=False, title="Method")
ax.add_artist(leg1)

style_handles = [
    Line2D([0],[0], color="black", linestyle="-", marker="o",
           linewidth=1.8, label=r"Observed $R/D$"),
    Line2D([0],[0], color="black", linestyle="--", marker="o",
           markerfacecolor="white", linewidth=1.6,
           label=r"Predicted $\widehat{R}/\widehat{D}$"),
]
ax.legend(handles=style_handles, loc="lower left", frameon=False)

ax.set_xlabel(r"Immediate damage $D$ (normalized)", fontsize=12)
ax.set_ylabel("Recoverability", fontsize=12)
ax.set_title("Observed vs. theory-predicted damage–recoverability landscape",
             loc="left", fontsize=13, fontweight="bold")
ax.set_xlim(0.45, 4)
ax.set_ylim(0.12, 0.76)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate(
    "better",
    xy=(0.06,-0.11), xycoords="axes fraction",
    xytext=(0.22,-0.11), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="<-", lw=1.1),
    ha="center", va="center", fontsize=9.5
)

fig.tight_layout()
out="figure_AB_overlaid_observed_predicted.png"
fig.savefig(out, dpi=240, bbox_inches="tight")
plt.show()
print(out)
