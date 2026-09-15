from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


INK = "#20252B"
MUTED = "#646C75"
LINE = "#C9CED3"
PALE = "#F4F5F6"
AMBER = "#AA661C"
AMBER_FILL = "#F7E9D4"
TEAL = "#087F83"
TEAL_FILL = "#DFF1ED"
RED = "#B55A52"
RED_FILL = "#F8E7E4"


def toy_example():
    w = np.array([1., 1., 2., 2.])
    h = np.array([[2., 0., 0., 0.],
                  [0., 2.4, 2., 0.],
                  [0., 2., 2., 0.],
                  [0., 0., 0., 3.]])
    eta = 0.1
    damage = 0.5 * w**2 * np.diag(h)
    recovery = eta * w**2 * (np.sum(h * h.T, axis=1) - np.diag(h)**2)
    score = damage - recovery
    actual_losses = []
    updated = []
    for i in range(w.size):
        delta = np.zeros_like(w)
        delta[i] = -w[i]
        z = np.ones_like(w)
        z[i] = 0
        m = -eta * z * (h @ delta)
        residual = delta + m
        actual_losses.append(0.5 * residual @ h @ residual)
        updated.append(w + residual)
    assert np.all(np.linalg.eigvalsh(h) > 0)
    assert np.allclose(damage, [1., 1.2, 4., 6.])
    assert np.allclose(recovery, [0., 0.4, 1.6, 0.])
    assert np.allclose(score, [1., 0.8, 2.4, 6.])
    assert np.argmin(damage) == 0
    assert np.argmin(score) == np.argmin(actual_losses) == 1
    assert np.allclose(updated[1], [1., 0., 2.2, 2.])
    return w, damage, recovery, score, np.array(updated), actual_losses


def label(ax, x, y, text, size=12, color=INK, ha="left", **kw):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha,
                   va="center", **kw)


def cells(ax, x, y, values, width=0.64, height=0.53,
          fills=None, outlines=None, decimals=False):
    """Draw a four-cell row; y is its vertical center."""
    fills = fills or {}
    outlines = outlines or {}
    for i, value in enumerate(values):
        edge = outlines.get(i, LINE)
        ax.add_patch(Rectangle((x+i*width, y-height/2), width, height,
                              facecolor=fills.get(i, PALE), edgecolor=edge,
                              linewidth=1.8 if i in outlines else 0.7,
                              zorder=2))
        string = f"{value:.1f}" if decimals else f"{value:g}"
        label(ax, x+(i+0.5)*width, y, string, 14,
              color=edge if i in outlines else INK, ha="center", zorder=3)


def arrow(ax, start, end, color=MUTED, rad=0, style="-|>", lw=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=10, linewidth=lw,
                                color=color,
                                connectionstyle=f"arc3,rad={rad}"))


def draw(outdir):
    w, damage, recovery, score, updated, actual = toy_example()
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42,
                         "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10.5, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # Shared candidates and independent-removal scores. The two outgoing
    # arrows explicitly connect each criterion to its chosen pruned row.
    x, width = 2.55, 0.65
    out_x = 7.30
    label(ax, 0.20, 4.85, "Same candidates", 12, fontweight="bold")
    label(ax, 6.85, 4.85, "Same budget: remove one weight", 11, MUTED)
    for i in range(4):
        label(ax, x+(i+0.5)*width, 4.56, rf"$w_{i+1}$", 12, ha="center")
    label(ax, 0.20, 4.19, r"Original weights  $W$", 11.5)
    cells(ax, x, 4.19, w, width=width)
    ax.plot([0.20, 10.08], [3.74, 3.74], color=LINE, lw=0.7)

    yd, yr, ys = 3.18, 2.24, 1.29
    label(ax, 0.20, yd, r"Immediate damage  $\widehat D$", 11.5)
    cells(ax, x, yd, damage, width=width, fills={0:AMBER_FILL},
          outlines={0:AMBER}, decimals=True)
    label(ax, 0.20, yr, r"Predicted recovery  $\widehat R$", 11.5)
    cells(ax, x, yr, recovery, width=width, fills={1:TEAL_FILL}, decimals=True)
    label(ax, x-0.18, yr, r"$-$", 18, MUTED, ha="center")
    ax.plot([x-0.02, x+4*width+0.02], [1.76, 1.76], color=LINE, lw=0.9)
    label(ax, 0.20, ys, r"DAP score  $\widehat D-\widehat R$", 11.5)
    cells(ax, x, ys, score, width=width, fills={1:TEAL_FILL},
          outlines={1:TEAL}, decimals=True)

    # Matching rows make the control-vs-DAP comparison explicit.
    label(ax, out_x+2*width, 3.57, r"Damage-only ($\eta=0$)", 12, AMBER,
          ha="center", fontweight="bold")
    arrow(ax, (5.38, yd), (7.06, yd), AMBER, lw=1.6)
    label(ax, 6.22, yd+0.22, "prune minimum", 10, AMBER, ha="center")
    baseline_weights = w.copy()
    baseline_weights[np.argmin(damage)] = 0
    cells(ax, out_x, yd, baseline_weights, width=width,
          fills={0:AMBER_FILL}, outlines={0:AMBER})
    label(ax, out_x+2*width, yd-0.46, r"Prune $w_1$:  $\widehat D=1.0$,  $\widehat I=1.0$",
          11, AMBER, ha="center")

    label(ax, out_x+2*width, 1.68, "DAP", 12, TEAL, ha="center", fontweight="bold")
    arrow(ax, (5.38, ys), (7.06, ys), TEAL, lw=1.6)
    label(ax, 6.22, ys+0.22, "prune minimum", 10, TEAL, ha="center")
    dap_weights = w.copy()
    dap_weights[np.argmin(score)] = 0
    cells(ax, out_x, ys, dap_weights, width=width,
          fills={1:TEAL_FILL}, outlines={1:TEAL})
    label(ax, out_x+2*width, ys-0.46, r"Prune $w_2$:  $\widehat D=1.2$,  $\widehat I=0.8$",
          11, TEAL, ha="center")

    label(ax, 5.25, 0.39,
          "DAP can accept greater immediate damage when predicted recovery more than offsets it.",
          11, INK, ha="center", fontweight="bold")
    label(ax, 5.25, 0.10,
          "Illustrative single-weight removals; all damage and recovery scores are local predictions.",
          9, MUTED, ha="center")

    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(outdir / f"dap_method_figure.{suffix}", dpi=220,
                    facecolor="white", metadata={"Creator": "DAP method schematic"})
    plt.close(fig)
    print("Immediate damage:", damage.tolist())
    print("Predicted recovery:", recovery.tolist())
    print("DAP score:", score.tolist())
    print("Quadratic loss after one full step:", actual)
    print("Saved PDF, SVG and PNG to", outdir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    draw(parser.parse_args().output_dir)