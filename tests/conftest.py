import pandas as pd
import pytest

from credit_underwriting.data import synthetic


@pytest.fixture(scope="session")
def applications() -> pd.DataFrame:
    return synthetic.generate(n=5_000, seed=11)
