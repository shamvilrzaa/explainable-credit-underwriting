from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib

from credit_underwriting.data import TARGET, load_applications
from credit_underwriting.metrics import calibration_table, evaluate
from credit_underwriting.models.scorecard import build_scorecard, coefficients
from credit_underwriting.split import split_applications

log = logging.getLogger(__name__)

MODELS = {"scorecard": build_scorecard}


def train(model_name: str, data_path: Path | None, output_dir: Path, seed: int = 42) -> dict:
    df = load_applications(data_path)
    split = split_applications(df, seed=seed)

    pipeline = MODELS[model_name](seed=seed)
    pipeline.fit(split.train, split.train[TARGET])

    predictions = {
        name: pipeline.predict_proba(part)[:, 1]
        for name, part in (("valid", split.valid), ("test", split.test))
    }
    results = {
        "valid": evaluate(split.valid[TARGET], predictions["valid"]).to_dict(),
        "test": evaluate(split.test[TARGET], predictions["test"]).to_dict(),
    }
    calibration = calibration_table(split.test[TARGET], predictions["test"])

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / f"{model_name}.joblib")
    (output_dir / f"{model_name}_metrics.json").write_text(json.dumps(results, indent=2))
    calibration.to_csv(output_dir / f"{model_name}_calibration.csv")
    if model_name == "scorecard":
        coefficients(pipeline).to_csv(output_dir / "scorecard_coefficients.csv", header=["coef"])

    log.info("%s test metrics: %s", model_name, results["test"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a credit risk model.")
    parser.add_argument("--model", choices=sorted(MODELS), default="scorecard")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = train(args.model, args.data, args.output, args.seed)
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
