from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from credit_underwriting.data import TARGET, load_applications
from credit_underwriting.models import MODELS
from credit_underwriting.plots import plot_calibration, plot_roc
from credit_underwriting.split import split_applications
from credit_underwriting.train import fit_and_evaluate, save_artifacts

log = logging.getLogger(__name__)

COLUMNS = ["gini", "auc", "ks", "brier", "mean_predicted_pd", "default_rate"]


def to_markdown(table: pd.DataFrame) -> str:
    header = "| model | " + " | ".join(table.columns) + " |"
    divider = "|---" * (len(table.columns) + 1) + "|"
    rows = [
        f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |"
        for name, row in table.iterrows()
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def compare(
    data_path: Path | None,
    output_dir: Path,
    figures_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    split = split_applications(load_applications(data_path), seed=seed)
    y_test = split.test[TARGET].to_numpy()

    rows = {}
    curves = {}
    for name in MODELS:
        result = fit_and_evaluate(name, split, seed)
        save_artifacts(name, result, split, output_dir)
        rows[name] = result.metrics["test"]
        curves[name] = (y_test, result.predictions["test"])

    table = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]
    table.to_csv(output_dir / "comparison.csv")
    (output_dir / "comparison.md").write_text(to_markdown(table))

    plot_roc(curves, figures_dir / "roc.png")
    plot_calibration(curves, figures_dir / "calibration.png")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models and compare them.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("docs/images"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    table = compare(args.data, args.output, args.figures, args.seed)
    print(to_markdown(table))


if __name__ == "__main__":
    main()
