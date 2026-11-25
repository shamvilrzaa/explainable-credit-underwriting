from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_underwriting.decision import DecisionPolicy, Outcome

SCORE_BANDS = [0, 480, 520, 560, 600, 640, 1000]
SCORE_LABELS = ["<480", "480-519", "520-559", "560-599", "600-639", "640+"]


@dataclass(frozen=True)
class PolicySummary:
    n: int
    approval_rate: float
    referral_rate: float
    decline_rate: float
    approved_default_rate: float
    approved_expected_pd: float


def apply_policy(pd_hat: np.ndarray, policy: DecisionPolicy) -> np.ndarray:
    return np.array([policy.outcome(float(p)).value for p in pd_hat])


def summarise_policy(
    pd_hat: np.ndarray, y_true: np.ndarray, policy: DecisionPolicy
) -> PolicySummary:
    outcomes = apply_policy(pd_hat, policy)
    approved = outcomes == Outcome.APPROVE.value
    y_true = np.asarray(y_true)
    return PolicySummary(
        n=len(outcomes),
        approval_rate=float(approved.mean()),
        referral_rate=float((outcomes == Outcome.REFER.value).mean()),
        decline_rate=float((outcomes == Outcome.DECLINE.value).mean()),
        approved_default_rate=float(y_true[approved].mean()) if approved.any() else float("nan"),
        approved_expected_pd=float(pd_hat[approved].mean()) if approved.any() else float("nan"),
    )


def score_band_table(scores: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    bands = pd.cut(scores, bins=SCORE_BANDS, labels=SCORE_LABELS, right=False)
    df = pd.DataFrame({"band": bands, "y": np.asarray(y_true)})
    table = df.groupby("band", observed=False).agg(
        applications=("y", "size"), default_rate=("y", "mean")
    )
    table["share"] = table["applications"] / table["applications"].sum()
    return table[["applications", "share", "default_rate"]]
