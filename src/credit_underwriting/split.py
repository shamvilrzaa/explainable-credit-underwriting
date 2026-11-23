from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_underwriting.data.schema import TARGET


class DataSplit(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_applications(
    df: pd.DataFrame,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    train_valid, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    relative_valid = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid,
        stratify=train_valid[TARGET],
        random_state=seed,
    )
    return DataSplit(
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
