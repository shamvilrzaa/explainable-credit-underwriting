from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

SCALED_COLUMNS = ["AMT_CREDIT", "AMT_ANNUITY"]


@dataclass(frozen=True)
class AffordableLoan:
    requested_amount: float
    max_amount: float
    probability_of_default: float

    @property
    def reduction(self) -> float:
        return 1 - self.max_amount / self.requested_amount


def max_affordable_loan(
    pipeline: Pipeline,
    application: pd.DataFrame,
    target_pd: float,
    min_share: float = 0.1,
    steps: int = 60,
) -> AffordableLoan | None:
    """Largest loan, at the same term, whose predicted PD is below ``target_pd``."""
    if len(application) != 1:
        raise ValueError("Expected exactly one application")

    shares = np.linspace(1.0, min_share, steps)
    candidates = pd.concat([application] * steps, ignore_index=True)
    candidates[SCALED_COLUMNS] = candidates[SCALED_COLUMNS].mul(shares, axis=0)
    pds = pipeline.predict_proba(candidates)[:, 1]

    eligible = np.flatnonzero(pds < target_pd)
    if eligible.size == 0:
        return None
    best = eligible[0]
    return AffordableLoan(
        requested_amount=float(application["AMT_CREDIT"].iloc[0]),
        max_amount=float(candidates["AMT_CREDIT"].iloc[best]),
        probability_of_default=float(pds[best]),
    )
