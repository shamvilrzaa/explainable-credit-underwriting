import json

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_underwriting.data.schema import TARGET
from credit_underwriting.fairness import (
    FairnessThresholds,
    age_band,
    audit,
    flag_issues,
    group_metrics,
    protected_groups,
)
from credit_underwriting.fairness_report import fairness_report
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.split import split_applications


def test_age_band_boundaries():
    days = pd.Series([-29.9, -30.0, -44.9, -45.0, -60.0, -75.0]) * 365.25
    assert list(age_band(days).astype(str)) == ["<30", "30-44", "30-44", "45-59", "60+", "60+"]


def test_protected_groups(applications):
    groups = protected_groups(applications)
    assert set(groups["gender"]) == {"female", "male"}
    assert set(groups["age_band"]) <= {"<30", "30-44", "45-59", "60+"}


def test_group_metrics_on_known_data():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    pd_hat = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9])
    approved = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    groups = pd.Series(["a"] * 4 + ["b"] * 4)

    table = group_metrics(y, pd_hat, approved, groups)
    assert table.loc["a", "approval_rate"] == 0.5
    assert table.loc["b", "approval_rate"] == 0.25
    assert table.loc["b", "approval_ratio"] == 0.5
    assert table.loc["a", "good_approval_rate"] == 1.0
    assert table.loc["b", "good_approval_gap"] == 0.5
    assert table.loc["a", "gini"] == pytest.approx(1.0)


def test_flag_issues_detects_disparity():
    table = pd.DataFrame(
        {
            "approval_ratio": [1.0, 0.6],
            "good_approval_gap": [0.0, 0.2],
            "calibration_gap": [0.0, -0.05],
        },
        index=["a", "b"],
    )
    issues = flag_issues(table, FairnessThresholds())
    assert len(issues) == 3
    assert all(issue.startswith("b:") for issue in issues)
    assert "underestimated" in issues[2]


def test_flag_issues_clean_table():
    table = pd.DataFrame(
        {"approval_ratio": [1.0, 0.95], "good_approval_gap": [0, 0.01], "calibration_gap": [0, 0]},
        index=["a", "b"],
    )
    assert flag_issues(table, FairnessThresholds()) == []


@pytest.fixture(scope="module")
def fitted(applications):
    split = split_applications(applications)
    return build_gbm(n_estimators=150).fit(split.train, split.train[TARGET]), split.test


def test_audit_covers_both_attributes(fitted):
    model, test = fitted
    pd_hat = model.predict_proba(test)[:, 1]
    result = audit(test, test[TARGET].to_numpy(), pd_hat, pd_hat < 0.05)
    assert set(result) == {"gender", "age_band"}
    assert result["gender"]["table"]["n"].sum() == len(test)


def test_fairness_report_writes_outputs(tmp_path, fitted):
    model, _ = fitted
    joblib.dump(model, tmp_path / "gbm.joblib")
    fairness_report(tmp_path / "gbm.joblib", tmp_path / "x.csv", tmp_path / "out", tmp_path / "fig")
    data = json.loads((tmp_path / "out" / "fairness.json").read_text())
    assert set(data) == {"gender", "age_band"}
    assert "### gender" in (tmp_path / "out" / "fairness.md").read_text()
    assert (tmp_path / "fig" / "fairness.png").stat().st_size > 0
