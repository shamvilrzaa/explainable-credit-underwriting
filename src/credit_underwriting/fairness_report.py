from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from credit_underwriting.data import TARGET, load_applications
from credit_underwriting.decision import DecisionPolicy, Outcome
from credit_underwriting.fairness import audit
from credit_underwriting.plots import plot_fairness
from credit_underwriting.split import split_applications

log = logging.getLogger(__name__)

COLUMNS = [
    "n",
    "default_rate",
    "mean_predicted_pd",
    "gini",
    "approval_rate",
    "approval_ratio",
    "good_approval_rate",
]


def to_markdown(results: dict[str, dict[str, object]]) -> str:
    lines = []
    for attribute, result in results.items():
        table = result["table"][COLUMNS]
        lines.append(f"### {attribute}\n")
        lines.append("| group | " + " | ".join(COLUMNS) + " |")
        lines.append("|---" * (len(COLUMNS) + 1) + "|")
        for group, row in table.iterrows():
            values = [f"{int(row['n'])}"] + [f"{row[c]:.3f}" for c in COLUMNS[1:]]
            lines.append(f"| {group} | " + " | ".join(values) + " |")
        issues = result["issues"] or ["No issues above thresholds"]
        lines.append("")
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
    return "\n".join(lines)


def fairness_report(
    model_path: Path,
    data_path: Path | None,
    output_dir: Path,
    figures_dir: Path,
    seed: int = 42,
    policy: DecisionPolicy | None = None,
) -> dict[str, dict[str, object]]:
    policy = policy or DecisionPolicy()
    pipeline = joblib.load(model_path)
    test = split_applications(load_applications(data_path), seed=seed).test

    pd_hat = pipeline.predict_proba(test)[:, 1]
    approved = np.array([policy.outcome(p) is Outcome.APPROVE for p in pd_hat])
    results = audit(test, test[TARGET].to_numpy(), pd_hat, approved)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fairness.md").write_text(to_markdown(results))
    serialisable = {
        attribute: {"table": r["table"].to_dict(orient="index"), "issues": r["issues"]}
        for attribute, r in results.items()
    }
    (output_dir / "fairness.json").write_text(json.dumps(serialisable, indent=2))
    plot_fairness({a: r["table"] for a, r in results.items()}, figures_dir / "fairness.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model decisions across protected groups.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/gbm.joblib"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("docs/images"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = fairness_report(args.model, args.data, args.output, args.figures, args.seed)
    print(to_markdown(results))


if __name__ == "__main__":
    main()
