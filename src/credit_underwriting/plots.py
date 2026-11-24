from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from credit_underwriting.metrics import calibration_table, gini

Curves = dict[str, tuple[np.ndarray, np.ndarray]]

COLORS = ["#4C6EF5", "#F59F00", "#12B886", "#E64980"]
RISK_UP = "#E03131"
RISK_DOWN = "#2F9E44"


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


def _save(fig: Figure, path: Path, rect: tuple[float, ...] = (0, 0, 1, 1)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    return path


def plot_global_importance(importance: pd.Series, path: Path) -> Path:
    ordered = importance.sort_values()
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.barh(ordered.index, ordered.to_numpy(), color=COLORS[0])
    _style(ax, "Global feature importance", "Mean |SHAP value| (log-odds)", "")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_applicant_contributions(
    contributions: pd.Series, path: Path, title: str = "Why was this applicant declined?"
) -> Path:
    ordered = contributions.reindex(contributions.abs().sort_values().index)
    colors = [RISK_UP if v > 0 else RISK_DOWN for v in ordered]
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.barh(ordered.index, ordered.to_numpy(), color=colors)
    ax.axvline(0, color="grey", lw=1)
    _style(ax, title, "Contribution to risk (log-odds)", "")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_fairness(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    fig = Figure(figsize=(10, 4.2))
    axes = fig.subplots(1, len(tables), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.38
    for ax, (attribute, table) in zip(axes, tables.items(), strict=True):
        x = np.arange(len(table))
        ax.bar(x - width / 2, table["approval_rate"], width, color=COLORS[0], label="Approval rate")
        ax.bar(
            x + width / 2,
            table["good_approval_rate"],
            width,
            color=COLORS[2],
            label="Approval rate (non-defaulters)",
        )
        ax.plot(x, table["default_rate"], "o", color=RISK_UP, label="Observed default rate")
        ax.set_xticks(x, table.index)
        _style(ax, attribute.replace("_", " ").capitalize(), "", "")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncols=3, fontsize=9)
    return _save(fig, path, rect=(0, 0.08, 1, 1))
