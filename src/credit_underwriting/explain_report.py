from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import joblib

from credit_underwriting.data import load_applications
from credit_underwriting.decision import Outcome, Underwriter
from credit_underwriting.plots import plot_applicant_contributions, plot_global_importance
from credit_underwriting.split import split_applications

log = logging.getLogger(__name__)


def explain_report(
    model_path: Path,
    data_path: Path | None,
    output_dir: Path,
    figures_dir: Path,
    seed: int = 42,
    n_examples: int = 5,
) -> dict[str, object]:
    pipeline = joblib.load(model_path)
    test = split_applications(load_applications(data_path), seed=seed).test
    underwriter = Underwriter(pipeline)

    decisions = underwriter.decide(test)
    outcomes = Counter(d.outcome.value for d in decisions)
    declined = [i for i, d in enumerate(decisions) if d.outcome is Outcome.DECLINE and d.reasons]
    reason_counts = Counter(r.label for i in declined for r in decisions[i].reasons[:1])

    plot_global_importance(
        underwriter.explainer.global_importance(test), figures_dir / "shap_importance.png"
    )
    if declined:
        example = test.iloc[[declined[0]]]
        contributions = underwriter.explainer.contributions(example).iloc[0]
        plot_applicant_contributions(contributions, figures_dir / "shap_declined_example.png")

    summary = {
        "n": len(decisions),
        "outcomes": dict(outcomes),
        "top_primary_decline_reasons": dict(reason_counts.most_common(5)),
        "examples": [decisions[i].to_dict() for i in declined[:n_examples]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decisions_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain decisions of a trained model.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/gbm.joblib"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("docs/images"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = explain_report(args.model, args.data, args.output, args.figures, args.seed)
    print(
        json.dumps({k: summary[k] for k in ("outcomes", "top_primary_decline_reasons")}, indent=2)
    )
    if summary["examples"]:
        print(json.dumps(summary["examples"][0], indent=2))


if __name__ == "__main__":
    main()
