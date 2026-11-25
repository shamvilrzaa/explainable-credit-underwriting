from __future__ import annotations

TARGET = "TARGET"
ID = "SK_ID_CURR"

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

PROTECTED = ["CODE_GENDER", "DAYS_BIRTH"]

RAW_COLUMNS = [ID, TARGET, *RAW_NUMERIC, *RAW_CATEGORICAL, "CODE_GENDER"]

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

FEATURE_LABELS: dict[str, str] = {
    "debt_to_income": "Requested credit is high relative to income",
    "annuity_to_income": "Loan repayments are high relative to income",
    "credit_to_goods": "Loan amount exceeds the value of the financed goods",
    "employment_years": "Short employment history",
    "ext_source_mean": "Weak external credit bureau scores",
    "ext_source_min": "At least one weak external credit bureau score",
    "income_log": "Income level",
    "credit_log": "Loan size",
    "cnt_children": "Number of dependants",
}

FEATURE_NAMES: dict[str, str] = {
    "debt_to_income": "Credit / income",
    "annuity_to_income": "Repayment / income",
    "credit_to_goods": "Credit / goods price",
    "employment_years": "Years employed",
    "ext_source_mean": "Bureau score (mean)",
    "ext_source_min": "Bureau score (lowest)",
    "income_log": "Income",
    "credit_log": "Loan amount",
    "cnt_children": "Children",
}
