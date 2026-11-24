from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.models.scorecard import ScoreScale, build_scorecard

MODELS = {"scorecard": build_scorecard, "gbm": build_gbm}

__all__ = ["MODELS", "ScoreScale", "build_gbm", "build_scorecard"]
