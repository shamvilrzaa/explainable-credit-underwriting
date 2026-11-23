from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_underwriting.features import FeatureBuilder


def build_scorecard(C: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=1_000, random_state=seed)),
        ]
    )


def coefficients(pipeline: Pipeline) -> pd.Series:
    names = pipeline[:-1].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    return pd.Series(coefs, index=names).sort_values(key=np.abs, ascending=False)


@dataclass(frozen=True)
class ScoreScale:
    """Points scale: ``base_score`` at ``base_odds`` good:bad, +``pdo`` points doubles the odds."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def to_score(self, pd_: np.ndarray) -> np.ndarray:
        pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
        odds = (1 - pd_) / pd_
        return np.round(self.offset + self.factor * np.log(odds)).astype(int)

    def to_pd(self, score: np.ndarray) -> np.ndarray:
        odds = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1 / (1 + odds)
