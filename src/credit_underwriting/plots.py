from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from credit_underwriting.metrics import calibration_table, gini

Curves = dict[str, tuple[np.ndarray, np.ndarray]]

COLORS = ["#4C6EF5", "#F59F00", "#12B886", "#E64980"]


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_roc(curves: Curves, path: Path) -> Path:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for (name, (y, score)), color in zip(curves.items(), COLORS, strict=False):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (Gini {gini(y, score):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    _style(ax, "ROC curve (test set)", "False positive rate", "True positive rate")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def plot_calibration(curves: Curves, path: Path, bins: int = 10) -> Path:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    upper = 0.0
    for (name, (y, score)), color in zip(curves.items(), COLORS, strict=False):
        table = calibration_table(y, score, bins=bins)
        ax.plot(
            table["predicted_pd"],
            table["observed_default_rate"],
            marker="o",
            color=color,
            lw=2,
            label=name,
        )
        upper = max(upper, table["predicted_pd"].max(), table["observed_default_rate"].max())
    ax.plot([0, upper], [0, upper], color="grey", lw=1, ls="--")
    _style(ax, "Calibration by risk decile", "Predicted PD", "Observed default rate")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path
