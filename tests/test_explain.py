import json

import joblib
import numpy as np
import pytest

from credit_underwriting.data.schema import FEATURE_LABELS, FEATURES, TARGET
from credit_underwriting.decision import DecisionPolicy, Outcome, Underwriter
from credit_underwriting.explain import REASON_FEATURES, ReasonCodeExplainer
from credit_underwriting.explain_report import explain_report
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.split import split_applications


@pytest.fixture(scope="module")
def model_and_test(applications):
    split = split_applications(applications)
    model = build_gbm(n_estimators=150).fit(split.train, split.train[TARGET])
    return model, split.test


def test_contributions_add_up_to_model_output(model_and_test):
    model, test = model_and_test
    sample = test.head(50)
    explainer = ReasonCodeExplainer(model)
    contributions = explainer.contributions(sample)
    X = model.named_steps["features"].transform(sample)
    raw = model.named_steps["model"].predict_proba(X, raw_score=True)

    assert list(contributions.columns) == list(FEATURES)
    np.testing.assert_allclose(contributions.sum(axis=1) + explainer.base_value, raw, atol=1e-6)


def test_reasons_only_list_risk_increasing_features(model_and_test):
    model, test = model_and_test
    explainer = ReasonCodeExplainer(model, max_reasons=3)
    for reasons in explainer.reasons(test.head(100)):
        assert len(reasons) <= 3
        contributions = [r.contribution for r in reasons]
        assert all(c >= explainer.min_contribution for c in contributions)
        assert contributions == sorted(contributions, reverse=True)
        assert all(r.label == FEATURE_LABELS[r.feature] for r in reasons)
        assert all(r.feature in REASON_FEATURES for r in reasons)


def test_policy_outcomes():
    policy = DecisionPolicy(approve_below=0.05, decline_above=0.2)
    assert policy.outcome(0.01) is Outcome.APPROVE
    assert policy.outcome(0.1) is Outcome.REFER
    assert policy.outcome(0.5) is Outcome.DECLINE


def test_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        DecisionPolicy(approve_below=0.3, decline_above=0.1)


def test_underwriter_decisions(model_and_test):
    model, test = model_and_test
    decisions = Underwriter(model).decide(test.head(300))

    assert len(decisions) == 300
    assert {d.outcome for d in decisions} == set(Outcome)
    for d in decisions:
        if d.outcome is Outcome.APPROVE:
            assert d.reasons == []
        assert 0 <= d.probability_of_default <= 1
    declined = [d for d in decisions if d.outcome is Outcome.DECLINE]
    assert all(d.reasons for d in declined)
    assert json.dumps(declined[0].to_dict())


def test_explain_report(tmp_path, model_and_test):
    model, _ = model_and_test
    joblib.dump(model, tmp_path / "gbm.joblib")
    summary = explain_report(
        tmp_path / "gbm.joblib", tmp_path / "missing.csv", tmp_path / "out", tmp_path / "fig"
    )
    assert sum(summary["outcomes"].values()) == summary["n"]
    assert summary["examples"]
    assert (tmp_path / "out" / "decisions_summary.json").exists()
    for name in ["shap_importance.png", "shap_declined_example.png"]:
        assert (tmp_path / "fig" / name).stat().st_size > 0
