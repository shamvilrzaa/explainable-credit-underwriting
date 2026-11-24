from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from credit_underwriting.features import DAYS_EMPLOYED_SENTINEL


class Application(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "applicant_id": 1001,
                "income_total": 54000,
                "credit_amount": 270000,
                "annuity": 18500,
                "goods_price": 250000,
                "employment_years": 3.5,
                "children": 1,
                "ext_source_1": None,
                "ext_source_2": 0.42,
                "ext_source_3": 0.35,
            }
        },
    )

    applicant_id: int | None = None
    income_total: float = Field(gt=0, description="Annual income")
    credit_amount: float = Field(gt=0)
    annuity: float = Field(gt=0, description="Annual loan repayment")
    goods_price: float | None = Field(default=None, gt=0)
    employment_years: float | None = Field(
        default=None, ge=0, le=60, description="Omit if not employed"
    )
    children: int = Field(default=0, ge=0, le=20)
    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)
    ext_source_3: float | None = Field(default=None, ge=0, le=1)


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=1000)


class ReasonOut(BaseModel):
    feature: str
    label: str
    value: float | None
    contribution: float


class DecisionOut(BaseModel):
    applicant_id: int | None
    probability_of_default: float
    score: int
    outcome: str
    reasons: list[ReasonOut]
    model_version: str


class BatchResponse(BaseModel):
    decisions: list[DecisionOut]


class ModelInfo(BaseModel):
    model_version: str
    approve_below: float
    decline_above: float
    features: list[str]


def to_frame(applications: list[Application]) -> pd.DataFrame:
    rows = []
    for a in applications:
        days_employed = (
            DAYS_EMPLOYED_SENTINEL
            if a.employment_years is None
            else -round(a.employment_years * 365.25)
        )
        rows.append(
            {
                "SK_ID_CURR": a.applicant_id,
                "AMT_INCOME_TOTAL": a.income_total,
                "AMT_CREDIT": a.credit_amount,
                "AMT_ANNUITY": a.annuity,
                "AMT_GOODS_PRICE": a.goods_price,
                "DAYS_EMPLOYED": days_employed,
                "CNT_CHILDREN": a.children,
                "EXT_SOURCE_1": a.ext_source_1,
                "EXT_SOURCE_2": a.ext_source_2,
                "EXT_SOURCE_3": a.ext_source_3,
            }
        )
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c != "SK_ID_CURR"]
    frame[numeric] = frame[numeric].astype(float)
    frame["SK_ID_CURR"] = frame["SK_ID_CURR"].astype(object)
    return frame
