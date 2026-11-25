from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from credit_underwriting.api.schemas import Application, to_frame
from credit_underwriting.decision import Underwriter

COLUMNS = list(Application.model_fields)


@dataclass(frozen=True)
class BatchResult:
    decisions: pd.DataFrame
    errors: pd.DataFrame


def template() -> pd.DataFrame:
    example = Application.model_config["json_schema_extra"]["example"]
    return pd.DataFrame([example], columns=COLUMNS)


def _clean(record: dict) -> dict:
    return {k: None if pd.isna(v) else v for k, v in record.items()}


def score_batch(rows: pd.DataFrame, underwriter: Underwriter) -> BatchResult:
    missing = {"income_total", "credit_amount", "annuity"} - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    applications, errors = [], []
    known = [c for c in rows.columns if c in COLUMNS]
    for line, record in enumerate(rows[known].to_dict(orient="records"), start=2):
        try:
            applications.append(Application(**_clean(record)))
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(str(p) for p in first["loc"])
            errors.append({"row": line, "field": field, "problem": first["msg"]})

    columns = ["applicant_id", "outcome", "probability_of_default", "score", "reasons"]
    if not applications:
        return BatchResult(pd.DataFrame(columns=columns), pd.DataFrame(errors))

    decisions = underwriter.decide(to_frame(applications))
    result = pd.DataFrame(
        {
            "applicant_id": [d.applicant_id for d in decisions],
            "outcome": [d.outcome.value for d in decisions],
            "probability_of_default": [d.probability_of_default for d in decisions],
            "score": [d.score for d in decisions],
            "reasons": ["; ".join(r.label for r in d.reasons) for d in decisions],
        }
    )
    return BatchResult(result, pd.DataFrame(errors, columns=["row", "field", "problem"]))
