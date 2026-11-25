import json

import pytest

from credit_underwriting.api.settings import Settings
from credit_underwriting.decision import DecisionPolicy
from credit_underwriting.policy_store import PolicyStore


def test_load_returns_default_when_missing(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    assert store.load() == DecisionPolicy()
    assert store.history() == []


def test_save_and_load_round_trip(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    store.save(DecisionPolicy(0.04, 0.2), changed_by="Risk team", reason="Q4 appetite")
    store.save(DecisionPolicy(0.06, 0.18), changed_by="Risk team", reason="Loss rate review")

    assert store.load() == DecisionPolicy(0.06, 0.18)
    history = store.history()
    assert [h.approve_below for h in history] == [0.04, 0.06]
    assert history[1].reason == "Loss rate review"
    assert history[1].timestamp


@pytest.mark.parametrize("who, why", [("", "reason"), ("Risk team", "  ")])
def test_save_requires_name_and_reason(tmp_path, who, why):
    with pytest.raises(ValueError):
        PolicyStore(tmp_path / "policy.json").save(DecisionPolicy(), who, why)


def test_settings_read_stored_policy(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"approve_below": 0.03, "decline_above": 0.25}))
    monkeypatch.setenv("POLICY_PATH", str(path))
    monkeypatch.delenv("APPROVE_BELOW", raising=False)
    monkeypatch.delenv("DECLINE_ABOVE", raising=False)
    assert Settings.from_env().policy == DecisionPolicy(0.03, 0.25)


def test_environment_overrides_stored_policy(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"approve_below": 0.03, "decline_above": 0.25}))
    monkeypatch.setenv("POLICY_PATH", str(path))
    monkeypatch.setenv("DECLINE_ABOVE", "0.3")
    assert Settings.from_env().policy == DecisionPolicy(0.03, 0.3)
