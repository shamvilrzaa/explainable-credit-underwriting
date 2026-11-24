import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from credit_underwriting.data.schema import FEATURES, PROTECTED, TARGET
from credit_underwriting.features import DAYS_EMPLOYED_SENTINEL, FeatureBuilder, safe_ratio


def _row(**overrides) -> pd.DataFrame:
    base = {
        "AMT_INCOME_TOTAL": 60_000.0,
        "AMT_CREDIT": 120_000.0,
        "AMT_ANNUITY": 12_000.0,
        "AMT_GOODS_PRICE": 100_000.0,
        "DAYS_EMPLOYED": -730,
        "CNT_CHILDREN": 1,
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": np.nan,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_output_matches_schema(applications):
    out = FeatureBuilder().fit_transform(applications)
    assert list(out.columns) == list(FEATURES)
    assert len(out) == len(applications)


def test_protected_attributes_are_not_features(applications):
    out = FeatureBuilder().fit_transform(applications)
    assert not set(PROTECTED) & set(out.columns)
    assert TARGET not in out.columns


def test_affordability_ratios():
    out = FeatureBuilder().fit_transform(_row())
    assert out.loc[0, "debt_to_income"] == pytest.approx(2.0)
    assert out.loc[0, "annuity_to_income"] == pytest.approx(0.2)
    assert out.loc[0, "credit_to_goods"] == pytest.approx(1.2)


def test_employment_years_and_sentinel():
    out = FeatureBuilder().fit_transform(_row())
    assert out.loc[0, "employment_years"] == pytest.approx(2.0, abs=0.01)

    out = FeatureBuilder().fit_transform(_row(DAYS_EMPLOYED=DAYS_EMPLOYED_SENTINEL))
    assert out.loc[0, "employment_years"] == 0


def test_employment_years_are_capped():
    out = FeatureBuilder(max_employment_years=40).fit_transform(_row(DAYS_EMPLOYED=-20_000))
    assert out.loc[0, "employment_years"] == 40


def test_ext_source_summaries_ignore_missing():
    out = FeatureBuilder().fit_transform(_row())
    assert out.loc[0, "ext_source_mean"] == pytest.approx(0.4)
    assert out.loc[0, "ext_source_min"] == pytest.approx(0.2)


def test_zero_income_gives_nan_not_inf():
    out = FeatureBuilder().fit_transform(_row(AMT_INCOME_TOTAL=0))
    assert np.isnan(out.loc[0, "debt_to_income"])
    assert np.isfinite(out.drop(columns=["debt_to_income", "annuity_to_income"])).all(axis=None)


def test_safe_ratio_handles_missing_denominator():
    result = safe_ratio(pd.Series([1.0, 1.0]), pd.Series([np.nan, 2.0]))
    assert np.isnan(result[0])
    assert result[1] == 0.5


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="AMT_CREDIT"):
        FeatureBuilder().fit(_row().drop(columns=["AMT_CREDIT"]))


def test_works_inside_pipeline(applications):
    pipe = Pipeline([("features", FeatureBuilder())])
    out = pipe.fit_transform(applications)
    assert list(pipe.get_feature_names_out()) == list(FEATURES)
    assert out.shape == (len(applications), len(FEATURES))
