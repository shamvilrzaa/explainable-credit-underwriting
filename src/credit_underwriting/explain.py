from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from credit_underwriting.data.schema import FEATURE_LABELS, FEATURES

REASON_FEATURES = [name for name, direction in FEATURES.items() if direction != 0]


@dataclass(frozen=True)
class Reason:
    feature: str
    label: str
    value: float | None
    contribution: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReasonCodeExplainer:
    """SHAP-based explanations in log-odds space for a fitted tree pipeline."""

    def __init__(self, pipeline: Pipeline, max_reasons: int = 4, min_contribution: float = 0.05):
        self.features = pipeline.named_steps["features"]
        self.model = pipeline.named_steps["model"]
        self.max_reasons = max_reasons
        self.min_contribution = min_contribution
        self._explainer = shap.TreeExplainer(self.model)

    def contributions(self, applications: pd.DataFrame) -> pd.DataFrame:
        X = self.features.transform(applications)
        values = self._explainer(X).values
        return pd.DataFrame(values, columns=X.columns, index=applications.index)

    @property
    def base_value(self) -> float:
        return float(np.ravel(self._explainer.expected_value)[0])

    def global_importance(self, applications: pd.DataFrame) -> pd.Series:
        return self.contributions(applications).abs().mean().sort_values(ascending=False)

    def reasons(self, applications: pd.DataFrame) -> list[list[Reason]]:
        X = self.features.transform(applications)
        contributions = self.contributions(applications)
        return [self._top_reasons(X.loc[idx], contributions.loc[idx]) for idx in applications.index]

    def _top_reasons(self, features: pd.Series, contributions: pd.Series) -> list[Reason]:
        eligible = contributions[REASON_FEATURES]
        adverse = eligible[eligible >= self.min_contribution]
        adverse = adverse.sort_values(ascending=False).head(self.max_reasons)
        return [
            Reason(
                feature=name,
                label=FEATURE_LABELS[name],
                value=None if pd.isna(features[name]) else round(float(features[name]), 4),
                contribution=round(float(value), 4),
            )
            for name, value in adverse.items()
        ]
