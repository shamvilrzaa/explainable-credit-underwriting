from pathlib import Path

import pytest

pytest.importorskip("plotly")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
PAGES = [
    "views/queue.py",
    "views/review.py",
    "views/batch.py",
    "views/portfolio.py",
    "views/model.py",
    "views/settings.py",
]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "gbm.joblib"))
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(APP), default_timeout=120)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(app, page):
    app.run()
    app.switch_page(page).run()
    assert not app.exception
    assert app.title


def test_policy_change_is_saved_and_logged(app, tmp_path):
    app.run()
    app.switch_page("views/portfolio.py").run()
    app.slider[0].set_value(6.0).run()
    app.text_input[0].input("Head of Credit Risk")
    app.text_input[1].input("Loss rate below appetite")
    app.button[0].click().run()

    assert not app.exception
    stored = (tmp_path / "policy.json").read_text()
    assert '"approve_below": 0.06' in stored
    assert (tmp_path / "policy_history.jsonl").exists()


def test_policy_change_requires_approver(app, tmp_path):
    app.run()
    app.switch_page("views/portfolio.py").run()
    app.slider[0].set_value(6.0).run()
    app.button[0].click().run()

    assert app.error
    assert not (tmp_path / "policy.json").exists()
