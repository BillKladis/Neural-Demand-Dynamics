"""Shared publication-quality plotting theme."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "truth": "#222222",
    "ude": "#2F6DB5",
    "blackbox": "#C44E52",
    "price": "#E1812C",
    "muted": "#9aa7b8",
    "grid": "#E6E6E6",
}


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.9,
    })


apply_theme()
