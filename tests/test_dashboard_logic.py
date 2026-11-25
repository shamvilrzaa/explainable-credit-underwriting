import numpy as np
import pandas as pd
import pytest

from credit_underwriting.audit import AuditLog, ReviewRecord
from credit_underwriting.counterfactual import max_affordable_loan
from credit_underwriting.data.schema import TARGET
from credit_underwriting.decision import DecisionPolicy
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.portfolio import apply_policy, score_band_table, summarise_policy
from credit_underwriting.split import split_applications


@pytest.fixture(scope="module")
def fitted(applications):
    split = split_applications(applications)
    return build_gbm(n_estimators=150).fit(split.train, split.train[TARGET]), split.test


def test_affordable_loan_is_smaller_and_below_threshold(fitted):
    model, test = fitted
    pds = model.predict_proba(test)[:, 1]
    risky = test.iloc[[int(np.argmax((pds > 0.1) & (pds < 0.2)))]]
    result = max_affordable_loan(model, risky, target_pd=0.05)
    if result is not None:
        assert result.max_amount < result.requested_amount
        assert result.probability_of_default < 0.05
        assert 0 < result.reduction < 1


def test_affordable_loan_returns_none_when_unreachable(fitted):
    model, test = fitted
    assert max_affordable_loan(model, test.head(1), target_pd=1e-6) is None


def test_affordable_loan_requires_single_row(fitted):
    model, test = fitted
    with pytest.raises(ValueError):
        max_affordable_loan(model, test.head(2), target_pd=0.05)


def test_policy_summary_rates_add_up():
    pd_hat = np.array([0.01, 0.02, 0.1, 0.3])
    y = np.array([0, 1, 0, 1])
    summary = summarise_policy(pd_hat, y, DecisionPolicy(0.05, 0.15))
    assert summary.approval_rate == 0.5
    assert summary.referral_rate == 0.25
    assert summary.decline_rate == 0.25
    assert summary.approved_default_rate == 0.5
    assert list(apply_policy(pd_hat, DecisionPolicy(0.05, 0.15))) == [
        "approve",
        "approve",
        "refer",
        "decline",
    ]


def test_score_band_table():
    table = score_band_table(np.array([450, 500, 500, 650]), np.array([1, 0, 1, 0]))
    assert table["applications"].sum() == 4
    assert table.loc["480-519", "default_rate"] == 0.5
    assert table["share"].sum() == pytest.approx(1.0)


def _record(**overrides) -> ReviewRecord:
    base = {
        "applicant_id": 1,
        "model_version": "abc",
        "model_outcome": "refer",
        "probability_of_default": 0.1,
        "final_outcome": "approve",
        "reviewer": "A. Schmidt",
        "note": "Stable income verified",
    }
    return ReviewRecord(**{**base, **overrides})


def test_audit_log_appends_with_timestamp(tmp_path):
    log = AuditLog(tmp_path / "log.csv")
    assert log.read().empty
    first = log.record(_record())
    log.record(_record(applicant_id=2, final_outcome="decline"))
    entries = log.read()
    assert len(entries) == 2
    assert first.timestamp
    assert list(entries["applicant_id"]) == [1, 2]
    assert first.is_override


@pytest.mark.parametrize("change", [{"reviewer": " "}, {"final_outcome": "refer"}])
def test_audit_log_validates(tmp_path, change):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "log.csv").record(_record(**change))


def test_empty_log_has_columns(tmp_path):
    assert list(AuditLog(tmp_path / "x.csv").read().columns) == AuditLog.COLUMNS
    assert isinstance(AuditLog(tmp_path / "x.csv").read(), pd.DataFrame)
