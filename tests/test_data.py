from pathlib import Path

import pandas as pd

from credit_underwriting.data import load_applications, synthetic
from credit_underwriting.data.schema import RAW_COLUMNS, TARGET


def test_synthetic_has_schema_and_reasonable_default_rate():
    df = synthetic.generate(n=5_000, seed=1)
    assert list(df.columns) == RAW_COLUMNS
    assert 0.04 < df[TARGET].mean() < 0.14


def test_synthetic_is_reproducible():
    pd.testing.assert_frame_equal(synthetic.generate(500, seed=7), synthetic.generate(500, seed=7))


def test_synthetic_plants_debt_burden_signal():
    df = synthetic.generate(n=20_000, seed=3)
    burden = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    high = df[burden > burden.quantile(0.8)][TARGET].mean()
    low = df[burden < burden.quantile(0.2)][TARGET].mean()
    assert high > 1.5 * low


def test_loader_falls_back_to_synthetic(tmp_path: Path):
    df = load_applications(tmp_path / "missing.csv", synthetic_n=1_000)
    assert len(df) == 1_000
