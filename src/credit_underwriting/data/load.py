"""Load real Home Credit data if present, otherwise fall back to synthetic data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_underwriting.data import synthetic
from credit_underwriting.data.schema import RAW_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "application_train.csv"


def load_applications(path: Path | None = None, *, synthetic_n: int = 20_000) -> pd.DataFrame:
    """Return loan applications with the columns in ``schema.RAW_COLUMNS``.

    Uses the Kaggle Home Credit file when it exists at ``path`` (default:
    ``data/raw/application_train.csv``); otherwise generates synthetic data.
    """
    path = path or DEFAULT_PATH
    if path.exists():
        log.info("Loading real Home Credit data from %s", path)
        df = pd.read_csv(path, usecols=RAW_COLUMNS)
        df = df[df["CODE_GENDER"].isin(["F", "M"])]  # drop the handful of 'XNA' rows
        return df.reset_index(drop=True)

    log.warning("No data at %s - using %d synthetic applications", path, synthetic_n)
    return synthetic.generate(n=synthetic_n)
