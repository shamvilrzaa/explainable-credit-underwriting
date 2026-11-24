import pandas as pd

from credit_underwriting.compare import compare, to_markdown
from credit_underwriting.models import MODELS


def test_compare_writes_table_and_figures(tmp_path):
    table = compare(tmp_path / "missing.csv", tmp_path / "out", tmp_path / "figures")

    assert set(table.index) == set(MODELS)
    assert (table["gini"] > 0.3).all()
    for name in ["comparison.csv", "comparison.md", "gbm.joblib", "scorecard.joblib"]:
        assert (tmp_path / "out" / name).exists()
    for name in ["roc.png", "calibration.png"]:
        assert (tmp_path / "figures" / name).stat().st_size > 0


def test_to_markdown_format():
    table = pd.DataFrame({"gini": [0.5]}, index=["scorecard"])
    lines = to_markdown(table).strip().splitlines()
    assert lines[0] == "| model | gini |"
    assert lines[2] == "| scorecard | 0.5000 |"
