import numpy as np
import pytest

from credit_underwriting.data.schema import FEATURES, TARGET
from credit_underwriting.metrics import gini
from credit_underwriting.models.gbm import build_gbm, feature_importance
from credit_underwriting.split import split_applications


@pytest.fixture(scope="module")
def fitted(applications):
    split = split_applications(applications)
    model = build_gbm(n_estimators=150).fit(split.train, split.train[TARGET])
    return model, split


def test_gbm_beats_random(fitted):
    model, split = fitted
    pd_hat = model.predict_proba(split.test)[:, 1]
    assert gini(split.test[TARGET], pd_hat) > 0.3


@pytest.mark.parametrize("feature", [f for f, d in FEATURES.items() if d != 0])
def test_monotone_constraints_hold(fitted, feature):
    model, split = fitted
    features = model.named_steps["features"].transform(split.test.head(200))
    booster = model.named_steps["model"]
    grid = np.linspace(features[feature].quantile(0.01), features[feature].quantile(0.99), 25)

    direction = FEATURES[feature]
    for _, row in features.head(20).iterrows():
        probe = features.loc[[row.name] * len(grid)].copy()
        probe[feature] = grid
        pd_curve = booster.predict_proba(probe)[:, 1]
        steps = np.diff(pd_curve) * direction
        assert (steps >= -1e-9).all(), f"{feature} violates its monotone constraint"


def test_unconstrained_model_can_be_built():
    model = build_gbm(monotone=False)
    assert "monotone_constraints" not in model.named_steps["model"].get_params()


def test_feature_importance_covers_all_features(fitted):
    model, _ = fitted
    importance = feature_importance(model)
    assert set(importance.index) == set(FEATURES)
    assert (importance >= 0).all()
