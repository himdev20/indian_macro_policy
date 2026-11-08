"""Shared figure style; every figure is saved as PNG (README) and PDF (report)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIG, SHADED, SHADE

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888886", "axes.labelcolor": "#3a3a38", "xtick.color": "#5a5a58",
    "ytick.color": "#5a5a58", "legend.frameon": False, "lines.linewidth": 1.8,
})


def shade_episodes(ax, spans=SHADED, alpha=0.12):
    for d0, d1 in spans:
        ax.axvspan(pd.Timestamp(d0), pd.Timestamp(d1), color=SHADE, alpha=alpha, lw=0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fy_ticks(ax, years):
    ax.set_xticks(list(years))
    ax.set_xticklabels([f"{y-1}-{str(y)[2:]}" for y in years], rotation=45, fontsize=7)
