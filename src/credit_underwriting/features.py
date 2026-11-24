from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_underwriting.data.schema import FEATURES

DAYS_EMPLOYED_SENTINEL = 365243
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
REQUIRED_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    *EXT_SOURCES,
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator > 0)
    return numerator / denominator


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, max_employment_years: float = 45.0) -> None:
        self.max_employment_years = max_employment_years

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:
        self._check_columns(X)
        self.feature_names_out_ = list(FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        income = X["AMT_INCOME_TOTAL"]
        credit = X["AMT_CREDIT"]

        days_employed = X["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, 0)
        employment_years = (-days_employed / 365.25).clip(0, self.max_employment_years)

        ext = X[EXT_SOURCES]

        features = pd.DataFrame(
            {
                "debt_to_income": safe_ratio(credit, income),
                "annuity_to_income": safe_ratio(X["AMT_ANNUITY"], income),
                "credit_to_goods": safe_ratio(credit, X["AMT_GOODS_PRICE"]),
                "employment_years": employment_years,
                "ext_source_mean": ext.mean(axis=1),
                "ext_source_min": ext.min(axis=1),
                "income_log": np.log1p(income.clip(lower=0)),
                "credit_log": np.log1p(credit.clip(lower=0)),
                "cnt_children": X["CNT_CHILDREN"].astype(float),
            },
            index=X.index,
        )
        return features[list(FEATURES)]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(list(FEATURES), dtype=object)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
