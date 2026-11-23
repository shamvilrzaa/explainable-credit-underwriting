"""Single source of truth for the columns the model uses.

The names follow the Home Credit Default Risk dataset, so the synthetic
generator and the real data go through exactly the same pipeline.
"""

from __future__ import annotations

TARGET = "TARGET"  # 1 = client had payment difficulties (default proxy)
ID = "SK_ID_CURR"

# Raw input columns taken from application_train.csv
RAW_NUMERIC = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
RAW_CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
]

# Protected attributes: kept for the fairness audit, NEVER used as model features.
PROTECTED = ["CODE_GENDER", "DAYS_BIRTH"]

RAW_COLUMNS = [ID, TARGET, *RAW_NUMERIC, *RAW_CATEGORICAL, "CODE_GENDER"]

# Engineered features the model is trained on (see features.py).
# Value = expected direction of effect on default risk, used for monotonic constraints:
#   +1 risk rises with the feature, -1 risk falls, 0 unconstrained.
FEATURES: dict[str, int] = {
    "debt_to_income": +1,
    "annuity_to_income": +1,
    "credit_to_goods": +1,
    "employment_years": -1,
    "ext_source_mean": -1,
    "ext_source_min": -1,
    "income_log": 0,
    "credit_log": 0,
    "cnt_children": 0,
}

# Plain-language labels used for adverse-action reason codes.
FEATURE_LABELS: dict[str, str] = {
    "debt_to_income": "Requested credit is high relative to income",
    "annuity_to_income": "Monthly repayment burden is high relative to income",
    "credit_to_goods": "Loan amount exceeds the value of the financed goods",
    "employment_years": "Short employment history",
    "ext_source_mean": "Weak external credit bureau scores",
    "ext_source_min": "At least one weak external credit bureau score",
    "income_log": "Income level",
    "credit_log": "Loan size",
    "cnt_children": "Number of dependants",
}
