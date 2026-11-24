from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = ["<30", "30-44", "45-59", "60+"]
GENDER_LABELS = {"F": "female", "M": "male"}


def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25
    return pd.cut(years, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def protected_groups(applications: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": applications["CODE_GENDER"].map(GENDER_LABELS).fillna("unknown"),
            "age_band": age_band(applications["DAYS_BIRTH"]),
        },
        index=applications.index,
    )


@dataclass(frozen=True)
class FairnessThresholds:
    min_approval_ratio: float = 0.8
    max_good_approval_gap: float = 0.1
    max_calibration_gap: float = 0.02


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def group_metrics(
    y_true: np.ndarray,
    pd_hat: np.ndarray,
    approved: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "pd": np.asarray(pd_hat),
            "approved": np.asarray(approved, dtype=bool),
            "group": pd.Series(groups).reset_index(drop=True),
        }
    )
    rows = {}
    for name, part in df.groupby("group", sort=True, observed=True):
        good = part[part["y"] == 0]
        rows[name] = {
            "n": len(part),
            "default_rate": part["y"].mean(),
            "mean_predicted_pd": part["pd"].mean(),
            "calibration_gap": part["pd"].mean() - part["y"].mean(),
            "gini": 2 * _safe_auc(part["y"].to_numpy(), part["pd"].to_numpy()) - 1,
            "approval_rate": part["approved"].mean(),
            "good_approval_rate": good["approved"].mean() if len(good) else float("nan"),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["approval_ratio"] = table["approval_rate"] / table["approval_rate"].max()
    table["good_approval_gap"] = table["good_approval_rate"].max() - table["good_approval_rate"]
    return table.round(4)


def flag_issues(table: pd.DataFrame, thresholds: FairnessThresholds) -> list[str]:
    issues = []
    for group, row in table.iterrows():
        if row["approval_ratio"] < thresholds.min_approval_ratio:
            issues.append(
                f"{group}: approval rate is {row['approval_ratio']:.0%} of the best-treated group"
            )
        if row["good_approval_gap"] > thresholds.max_good_approval_gap:
            issues.append(
                f"{group}: creditworthy applicants approved {row['good_approval_gap']:.1%} "
                "less often than in the best-treated group"
            )
        if abs(row["calibration_gap"]) > thresholds.max_calibration_gap:
            direction = "over" if row["calibration_gap"] > 0 else "under"
            issues.append(
                f"{group}: risk is {direction}estimated by {abs(row['calibration_gap']):.1%}"
            )
    return issues


def audit(
    applications: pd.DataFrame,
    y_true: np.ndarray,
    pd_hat: np.ndarray,
    approved: np.ndarray,
    thresholds: FairnessThresholds | None = None,
) -> dict[str, dict[str, object]]:
    thresholds = thresholds or FairnessThresholds()
    groups = protected_groups(applications)
    result = {}
    for attribute in groups.columns:
        table = group_metrics(y_true, pd_hat, approved, groups[attribute])
        result[attribute] = {"table": table, "issues": flag_issues(table, thresholds)}
    return result
