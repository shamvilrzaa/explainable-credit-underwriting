import numpy as np
import pytest

from credit_underwriting.metrics import calibration_table, evaluate, gini, ks_statistic


def test_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert gini(y, score) == pytest.approx(1.0)
    assert ks_statistic(y, score) == pytest.approx(1.0)


def test_random_ranking_is_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    score = rng.random(20_000)
    assert abs(gini(y, score)) < 0.03
    assert ks_statistic(y, score) < 0.03


def test_calibration_table_buckets():
    rng = np.random.default_rng(1)
    pd_ = rng.uniform(0, 0.3, 10_000)
    y = (rng.random(10_000) < pd_).astype(int)
    table = calibration_table(y, pd_, bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 10_000
    assert table["predicted_pd"].is_monotonic_increasing
    assert table["difference"].abs().max() < 0.03


def test_evaluate_report():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.7])
    report = evaluate(y, score)
    assert report.gini == pytest.approx(1.0)
    assert report.n == 4
    assert report.default_rate == 0.5
    assert set(report.to_dict()) >= {"auc", "gini", "ks", "brier"}
