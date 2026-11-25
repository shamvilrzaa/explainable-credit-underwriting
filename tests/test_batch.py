import pandas as pd
import pytest

from credit_underwriting.batch import COLUMNS, score_batch, template
from credit_underwriting.data.schema import TARGET
from credit_underwriting.decision import Underwriter
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.split import split_applications


@pytest.fixture(scope="module")
def underwriter(applications):
    split = split_applications(applications)
    return Underwriter(build_gbm(n_estimators=100).fit(split.train, split.train[TARGET]))


def test_template_has_all_columns():
    assert list(template().columns) == COLUMNS
    assert len(template()) == 1


def test_scores_valid_rows_and_reports_invalid_ones(underwriter):
    rows = pd.concat([template()] * 3, ignore_index=True)
    rows["applicant_id"] = [1, 2, 3]
    rows.loc[1, "income_total"] = -5
    result = score_batch(rows, underwriter)

    assert list(result.decisions["applicant_id"]) == [1, 3]
    assert set(result.decisions["outcome"]) <= {"approve", "refer", "decline"}
    assert result.errors.to_dict(orient="records") == [
        {"row": 3, "field": "income_total", "problem": "Input should be greater than 0"}
    ]


def test_missing_optional_values_are_allowed(underwriter):
    rows = template().drop(columns=["ext_source_1", "goods_price", "applicant_id"])
    result = score_batch(rows, underwriter)
    assert len(result.decisions) == 1
    assert result.errors.empty


def test_missing_required_column_raises(underwriter):
    with pytest.raises(ValueError, match="annuity"):
        score_batch(template().drop(columns=["annuity"]), underwriter)


def test_all_rows_invalid(underwriter):
    rows = template().assign(credit_amount=0)
    result = score_batch(rows, underwriter)
    assert result.decisions.empty
    assert len(result.errors) == 1
