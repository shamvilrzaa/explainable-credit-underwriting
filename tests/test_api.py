import pytest
from fastapi.testclient import TestClient

from credit_underwriting.api import create_app
from credit_underwriting.api.main import load_underwriter
from credit_underwriting.api.settings import Settings
from credit_underwriting.data.schema import FEATURES, TARGET
from credit_underwriting.decision import Underwriter
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.split import split_applications

GOOD = {
    "applicant_id": 1,
    "income_total": 200_000,
    "credit_amount": 150_000,
    "annuity": 9_000,
    "goods_price": 150_000,
    "employment_years": 12,
    "children": 0,
    "ext_source_2": 0.85,
    "ext_source_3": 0.8,
}
RISKY = {
    "applicant_id": 2,
    "income_total": 40_000,
    "credit_amount": 600_000,
    "annuity": 30_000,
    "goods_price": 450_000,
    "employment_years": None,
    "children": 3,
    "ext_source_2": 0.1,
    "ext_source_3": 0.15,
}


@pytest.fixture(scope="module")
def client(applications):
    split = split_applications(applications)
    model = build_gbm(n_estimators=150).fit(split.train, split.train[TARGET])
    with TestClient(create_app(Underwriter(model), model_version="abc123")) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/v1/model").json()
    assert info["model_version"] == "abc123"
    assert info["features"] == list(FEATURES)
    assert info["approve_below"] < info["decline_above"]


def test_low_risk_applicant_is_approved(client):
    body = client.post("/v1/decisions", json=GOOD).json()
    assert body["outcome"] == "approve"
    assert body["reasons"] == []
    assert body["applicant_id"] == 1
    assert body["model_version"] == "abc123"


def test_high_risk_applicant_is_declined_with_reasons(client):
    body = client.post("/v1/decisions", json=RISKY).json()
    assert body["outcome"] == "decline"
    assert body["reasons"]
    assert body["reasons"][0]["contribution"] >= body["reasons"][-1]["contribution"]


def test_batch_keeps_order_and_handles_missing_ids(client):
    no_id = {**GOOD, "applicant_id": None}
    body = client.post("/v1/decisions/batch", json={"applications": [RISKY, no_id]}).json()
    ids = [d["applicant_id"] for d in body["decisions"]]
    assert ids == [2, None]


@pytest.mark.parametrize(
    "change",
    [
        {"income_total": 0},
        {"ext_source_2": 1.5},
        {"children": -1},
        {"gender": "F"},
    ],
)
def test_invalid_input_is_rejected(client, change):
    response = client.post("/v1/decisions", json={**GOOD, **change})
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/v1/decisions/batch", json={"applications": []}).status_code == 422


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.delenv("APPROVE_BELOW", raising=False)
    monkeypatch.setenv("DECLINE_ABOVE", "0.3")
    settings = Settings.from_env()
    assert settings.model_path == tmp_path / "m.joblib"
    assert settings.decline_above == 0.3
    assert settings.approve_below == 0.05


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train-gbm"):
        load_underwriter(Settings(model_path=tmp_path / "missing.joblib"))
