from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from credit_underwriting.data import TARGET, load_applications
from credit_underwriting.metrics import calibration_table, evaluate
from credit_underwriting.models import MODELS
from credit_underwriting.models.gbm import feature_importance
from credit_underwriting.models.scorecard import coefficients
from credit_underwriting.split import DataSplit, split_applications

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    pipeline: Pipeline
    predictions: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float | int]]


def fit_and_evaluate(model_name: str, split: DataSplit, seed: int = 42) -> FitResult:
    pipeline = MODELS[model_name](seed=seed)
    pipeline.fit(split.train, split.train[TARGET])

    parts = {"valid": split.valid, "test": split.test}
    predictions = {name: pipeline.predict_proba(part)[:, 1] for name, part in parts.items()}
    metrics = {
        name: evaluate(part[TARGET], predictions[name]).to_dict() for name, part in parts.items()
    }
    return FitResult(pipeline, predictions, metrics)


def save_artifacts(model_name: str, result: FitResult, split: DataSplit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, output_dir / f"{model_name}.joblib")
    (output_dir / f"{model_name}_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    calibration_table(split.test[TARGET], result.predictions["test"]).to_csv(
        output_dir / f"{model_name}_calibration.csv"
    )
    if model_name == "scorecard":
        coefficients(result.pipeline).to_csv(
            output_dir / "scorecard_coefficients.csv", header=["coef"]
        )
    elif model_name == "gbm":
        feature_importance(result.pipeline).to_csv(
            output_dir / "gbm_feature_importance.csv", header=["gain"]
        )


def train(model_name: str, data_path: Path | None, output_dir: Path, seed: int = 42) -> dict:
    split = split_applications(load_applications(data_path), seed=seed)
    result = fit_and_evaluate(model_name, split, seed)
    save_artifacts(model_name, result, split, output_dir)
    log.info("%s test metrics: %s", model_name, result.metrics["test"])
    return result.metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a credit risk model.")
    parser.add_argument("--model", choices=sorted(MODELS), default="scorecard")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metrics = train(args.model, args.data, args.output, args.seed)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
