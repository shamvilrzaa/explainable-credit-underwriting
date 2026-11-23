from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class EvaluationReport:
    auc: float
    gini: float
    ks: float
    brier: float
    default_rate: float
    mean_predicted_pd: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(y_score)})
    df["bucket"] = pd.qcut(df["pd"].rank(method="first"), q=bins, labels=False) + 1
    table = df.groupby("bucket").agg(
        n=("y", "size"),
        predicted_pd=("pd", "mean"),
        observed_default_rate=("y", "mean"),
    )
    table["difference"] = table["observed_default_rate"] - table["predicted_pd"]
    return table


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> EvaluationReport:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    return EvaluationReport(
        auc=round(float(auc), 4),
        gini=round(float(2 * auc - 1), 4),
        ks=round(ks_statistic(y_true, y_score), 4),
        brier=round(float(brier_score_loss(y_true, y_score)), 4),
        default_rate=round(float(y_true.mean()), 4),
        mean_predicted_pd=round(float(y_score.mean()), 4),
        n=int(len(y_true)),
    )
