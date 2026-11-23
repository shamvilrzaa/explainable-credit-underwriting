import json

import numpy as np
import pytest

from credit_underwriting.data.schema import TARGET
from credit_underwriting.metrics import gini
from credit_underwriting.models.scorecard import ScoreScale, build_scorecard, coefficients
from credit_underwriting.split import split_applications
from credit_underwriting.train import train


@pytest.fixture(scope="module")
def fitted(applications):
    split = split_applications(applications)
    model = build_scorecard().fit(split.train, split.train[TARGET])
    return model, split


def test_scorecard_beats_random(fitted):
    model, split = fitted
    pd_hat = model.predict_proba(split.test)[:, 1]
    assert gini(split.test[TARGET], pd_hat) > 0.3


def test_scorecard_learns_expected_directions(fitted):
    model, _ = fitted
    coefs = coefficients(model)
    assert coefs["annuity_to_income"] > 0
    assert coefs["ext_source_mean"] < 0


def test_scorecard_handles_missing_values(fitted, applications):
    model, _ = fitted
    rows = applications.head(5).copy()
    rows["EXT_SOURCE_1"] = np.nan
    rows["AMT_GOODS_PRICE"] = np.nan
    assert np.isfinite(model.predict_proba(rows)).all()


def test_score_scale_anchor_points():
    scale = ScoreScale(base_score=600, base_odds=50, pdo=20)
    assert scale.to_score(np.array([1 / 51]))[0] == 600
    assert scale.to_score(np.array([1 / 101]))[0] == 620


def test_score_scale_round_trip():
    scale = ScoreScale()
    pd_ = np.array([0.01, 0.05, 0.2, 0.5])
    assert scale.to_pd(scale.to_score(pd_)) == pytest.approx(pd_, rel=0.05)


def test_higher_risk_means_lower_score():
    scores = ScoreScale().to_score(np.array([0.01, 0.1, 0.5]))
    assert list(scores) == sorted(scores, reverse=True)


def test_train_writes_artifacts(tmp_path):
    results = train("scorecard", tmp_path / "missing.csv", tmp_path / "out")
    assert results["test"]["gini"] > 0.3
    for name in [
        "scorecard.joblib",
        "scorecard_metrics.json",
        "scorecard_calibration.csv",
        "scorecard_coefficients.csv",
    ]:
        assert (tmp_path / "out" / name).exists()
    assert json.loads((tmp_path / "out" / "scorecard_metrics.json").read_text())["valid"]["n"] > 0
