from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from sklearn.pipeline import Pipeline

from credit_underwriting.data.schema import ID
from credit_underwriting.explain import Reason, ReasonCodeExplainer
from credit_underwriting.models.scorecard import ScoreScale


class Outcome(StrEnum):
    APPROVE = "approve"
    REFER = "refer"
    DECLINE = "decline"


@dataclass(frozen=True)
class DecisionPolicy:
    approve_below: float = 0.05
    decline_above: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.approve_below <= self.decline_above < 1:
            raise ValueError("Require 0 < approve_below <= decline_above < 1")

    def outcome(self, pd_: float) -> Outcome:
        if pd_ < self.approve_below:
            return Outcome.APPROVE
        if pd_ > self.decline_above:
            return Outcome.DECLINE
        return Outcome.REFER


@dataclass(frozen=True)
class Decision:
    applicant_id: int | None
    probability_of_default: float
    score: int
    outcome: Outcome
    reasons: list[Reason] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "applicant_id": self.applicant_id,
            "probability_of_default": self.probability_of_default,
            "score": self.score,
            "outcome": self.outcome.value,
            "reasons": [r.to_dict() for r in self.reasons],
        }


class Underwriter:
    def __init__(
        self,
        pipeline: Pipeline,
        policy: DecisionPolicy | None = None,
        scale: ScoreScale | None = None,
        max_reasons: int = 4,
    ) -> None:
        self.pipeline = pipeline
        self.policy = policy or DecisionPolicy()
        self.scale = scale or ScoreScale()
        self.explainer = ReasonCodeExplainer(pipeline, max_reasons=max_reasons)

    def decide(self, applications: pd.DataFrame) -> list[Decision]:
        pds = self.pipeline.predict_proba(applications)[:, 1]
        scores = self.scale.to_score(pds)
        reasons = self.explainer.reasons(applications)
        ids = applications[ID] if ID in applications else [None] * len(applications)

        decisions = []
        for applicant_id, pd_, score, applicant_reasons in zip(
            ids, pds, scores, reasons, strict=True
        ):
            outcome = self.policy.outcome(float(pd_))
            decisions.append(
                Decision(
                    applicant_id=None if applicant_id is None else int(applicant_id),
                    probability_of_default=round(float(pd_), 4),
                    score=int(score),
                    outcome=outcome,
                    reasons=[] if outcome is Outcome.APPROVE else applicant_reasons,
                )
            )
        return decisions
