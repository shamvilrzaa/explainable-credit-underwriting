from __future__ import annotations

import numpy as np
import pandas as pd

from credit_underwriting.data.schema import ID, RAW_COLUMNS, TARGET


def generate(n: int = 20_000, seed: int = 42, default_rate: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_years = rng.uniform(21, 68, n)
    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    income = np.exp(rng.normal(11.9, 0.5, n))
    goods = np.exp(rng.normal(12.9, 0.7, n))
    credit = goods * rng.uniform(1.0, 1.35, n)
    annuity = credit / rng.uniform(8, 30, n)
    employed_years = np.clip(rng.exponential(6, n), 0, age_years - 18)
    unemployed = rng.random(n) < 0.15
    days_employed = np.where(unemployed, 365243, -(employed_years * 365.25)).astype(int)

    latent_quality = rng.normal(0, 1, n)
    ext = [1 / (1 + np.exp(-(latent_quality + rng.normal(0, 0.8, n)))) for _ in range(3)]
    ext[0] = np.where(rng.random(n) < 0.55, np.nan, ext[0])

    logit = (
        -0.9 * latent_quality
        + 1.2 * np.log(annuity / income * 12)
        + 0.6 * (credit / goods - 1)
        - 0.05 * np.where(unemployed, 0, employed_years)
        + rng.normal(0, 0.5, n)
    )
    intercept = np.quantile(logit, 1 - default_rate)
    p_default = 1 / (1 + np.exp(-(logit - intercept) * 1.5))
    target = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame(
        {
            ID: np.arange(100_000, 100_000 + n),
            TARGET: target,
            "AMT_INCOME_TOTAL": income.round(0),
            "AMT_CREDIT": credit.round(0),
            "AMT_ANNUITY": annuity.round(1),
            "AMT_GOODS_PRICE": goods.round(0),
            "DAYS_BIRTH": -(age_years * 365.25).astype(int),
            "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": rng.poisson(0.4, n),
            "EXT_SOURCE_1": ext[0],
            "EXT_SOURCE_2": ext[1],
            "EXT_SOURCE_3": ext[2],
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "NAME_INCOME_TYPE": np.where(
                unemployed, "Pensioner", rng.choice(["Working", "Commercial associate"], n)
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education", "Incomplete higher"],
                n,
                p=[0.7, 0.25, 0.05],
            ),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Rented apartment"], n, p=[0.88, 0.06, 0.06]
            ),
            "CODE_GENDER": gender,
        }
    )
    return df[RAW_COLUMNS]
