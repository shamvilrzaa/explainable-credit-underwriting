from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from credit_underwriting.data.schema import FEATURES
from credit_underwriting.features import FeatureBuilder

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def build_gbm(seed: int = 42, monotone: bool = True, **params: object) -> Pipeline:
    config = {**DEFAULT_PARAMS, **params}
    if monotone:
        config["monotone_constraints"] = list(FEATURES.values())
        config["monotone_constraints_method"] = "advanced"
    model = LGBMClassifier(random_state=seed, verbose=-1, **config)
    return Pipeline([("features", FeatureBuilder()), ("model", model)])


def feature_importance(pipeline: Pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(gain, index=list(FEATURES)).sort_values(ascending=False)
