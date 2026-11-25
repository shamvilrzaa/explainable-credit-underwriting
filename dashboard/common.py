from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_underwriting import __version__
from credit_underwriting.api.settings import Settings
from credit_underwriting.audit import AuditLog
from credit_underwriting.data import TARGET, load_applications
from credit_underwriting.decision import DecisionPolicy, Underwriter
from credit_underwriting.explain import ReasonCodeExplainer
from credit_underwriting.models.gbm import build_gbm
from credit_underwriting.models.scorecard import ScoreScale
from credit_underwriting.policy_store import PolicyStore
from credit_underwriting.split import split_applications

PRODUCT = "Klarlend"
TAGLINE = "Explainable credit decisions"
ASSETS = Path(__file__).parent / "assets"
AUDIT_PATH = Path(os.getenv("REVIEW_LOG_PATH", "artifacts/review_log.csv"))

PALETTE = {
    "approve": "#2E9466",
    "refer": "#C38A12",
    "decline": "#D64535",
    "accent": "#4A78B5",
    "grid": "rgba(128, 128, 128, 0.15)",
    "neutral": "#8A94A0",
}

OUTCOME_LABELS = {
    "approve": "Approve",
    "refer": "Refer to underwriter",
    "decline": "Decline",
}

CSS = """
<style>
[data-testid="stMainBlockContainer"] {padding-top: 5rem !important; max-width: 1180px;}
h1 {font-size: 1.65rem !important; font-weight: 600 !important; letter-spacing: -0.01em;}
h3 {font-size: 1.02rem !important; font-weight: 600 !important; margin-top: 0.6rem;}
.eyebrow {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6;}
.decision {border-left: 4px solid; border-radius: 4px; padding: 14px 18px; margin-bottom: 18px;}
.decision .value {font-size: 1.45rem; font-weight: 600; margin-top: 2px;}
.reason {display: flex; justify-content: space-between; gap: 16px; padding: 9px 0;
         border-bottom: 1px solid rgba(128, 128, 128, 0.2); font-size: 0.93rem;}
.reason .detail {opacity: 0.6; white-space: nowrap; font-variant-numeric: tabular-nums;}
.muted {opacity: 0.68; font-size: 0.86rem;}
.footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(128, 128, 128, 0.2);
         opacity: 0.55; font-size: 0.78rem; display: flex; justify-content: space-between;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
"""


def palette() -> dict[str, str]:
    return PALETTE


def outcome_color(outcome: str) -> str:
    return PALETTE[outcome]


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, description: str | None = None) -> None:
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    if description:
        st.markdown(f"<p class='muted'>{description}</p>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model")
def load_model() -> tuple[object, str]:
    path = Settings.from_env().model_path
    if not path.exists():
        split = split_applications(load_applications())
        model = build_gbm().fit(split.train, split.train[TARGET])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return joblib.load(path), version


def policy_store() -> PolicyStore:
    return PolicyStore(Settings.from_env().policy_path)


def current_policy() -> DecisionPolicy:
    return Settings.from_env().policy


@st.cache_resource
def _underwriter(approve_below: float, decline_above: float) -> Underwriter:
    pipeline, _ = load_model()
    return Underwriter(pipeline, policy=DecisionPolicy(approve_below, decline_above))


def get_underwriter() -> Underwriter:
    policy = current_policy()
    return _underwriter(policy.approve_below, policy.decline_above)


@st.cache_data(show_spinner="Scoring applications")
def _scored_test_set() -> pd.DataFrame:
    pipeline, _ = load_model()
    test = split_applications(load_applications()).test
    scored = test.copy()
    scored["pd"] = pipeline.predict_proba(test)[:, 1].round(4)
    scored["score"] = ScoreScale().to_score(scored["pd"])
    reasons = ReasonCodeExplainer(pipeline).reasons(test)
    scored["main_reason"] = [r[0].label if r else "" for r in reasons]
    return scored


def scored_test_set() -> pd.DataFrame:
    scored = _scored_test_set().copy()
    policy = current_policy()
    scored["outcome"] = [policy.outcome(p).value for p in scored["pd"]]
    return scored


def audit_log() -> AuditLog:
    return AuditLog(AUDIT_PATH)


def eur(value: float) -> str:
    return f"€{value:,.0f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def footer() -> None:
    _, version = load_model()
    policy = current_policy()
    st.markdown(
        f"<div class='footer'><span>{PRODUCT} {__version__} · model {version}</span>"
        f"<span>Approve below {pct(policy.approve_below)} PD · "
        f"decline above {pct(policy.decline_above)} PD</span></div>",
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 320) -> go.Figure:
    colors = palette()
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=colors["grid"])
    return fig
