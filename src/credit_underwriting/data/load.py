from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_underwriting.data import synthetic
from credit_underwriting.data.schema import RAW_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "application_train.csv"


def load_applications(path: Path | None = None, *, synthetic_n: int = 20_000) -> pd.DataFrame:
    """Load Home Credit applications, or synthetic ones if the file is missing."""
    path = path or DEFAULT_PATH
    if path.exists():
        log.info("Loading real Home Credit data from %s", path)
        df = pd.read_csv(path, usecols=RAW_COLUMNS)
        df = df[df["CODE_GENDER"].isin(["F", "M"])]
        return df.reset_index(drop=True)

    log.warning("No data at %s - using %d synthetic applications", path, synthetic_n)
    return synthetic.generate(n=synthetic_n)
