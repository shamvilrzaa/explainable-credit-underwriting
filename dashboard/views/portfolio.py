import plotly.graph_objects as go
import streamlit as st
from common import (
    OUTCOME_LABELS,
    current_policy,
    header,
    palette,
    pct,
    policy_store,
    scored_test_set,
    style_figure,
)

from credit_underwriting.data.schema import TARGET
from credit_underwriting.decision import DecisionPolicy
from credit_underwriting.models.scorecard import ScoreScale
from credit_underwriting.portfolio import apply_policy, score_band_table, summarise_policy

header(
    "Risk management",
    "Decision policy",
    "Test what different thresholds would mean for approval volume and credit losses before "
    "changing the live policy. Figures are based on the hold-out test set.",
)

scored = scored_test_set()
pd_hat = scored["pd"].to_numpy()
y = scored[TARGET].to_numpy()
live = current_policy()

a, b = st.columns(2)
approve_pct = a.slider(
    "Approve below PD (%)", 1.0, 20.0, live.approve_below * 100, step=0.5, format="%.1f%%"
)
decline_pct = b.slider(
    "Decline above PD (%)",
    approve_pct,
    40.0,
    max(live.decline_above * 100, approve_pct),
    step=0.5,
    format="%.1f%%",
)
candidate = DecisionPolicy(round(approve_pct / 100, 4), round(decline_pct / 100, 4))

simulated = summarise_policy(pd_hat, y, candidate)
baseline = summarise_policy(pd_hat, y, live)


def delta(new: float, old: float) -> str | None:
    return None if abs(new - old) < 1e-9 else f"{(new - old) * 100:+.1f} pp"


k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Approved", pct(simulated.approval_rate), delta(simulated.approval_rate, baseline.approval_rate)
)
k2.metric(
    "Referred",
    pct(simulated.referral_rate),
    delta(simulated.referral_rate, baseline.referral_rate),
    delta_color="inverse",
)
k3.metric(
    "Declined",
    pct(simulated.decline_rate),
    delta(simulated.decline_rate, baseline.decline_rate),
    delta_color="off",
)
k4.metric(
    "Default rate of approved",
    pct(simulated.approved_default_rate),
    delta(simulated.approved_default_rate, baseline.approved_default_rate),
    delta_color="inverse",
)

colors = palette()
scale = ScoreScale()
outcomes = apply_policy(pd_hat, candidate)

st.subheader("Score distribution")
fig = go.Figure()
for outcome, label in OUTCOME_LABELS.items():
    fig.add_trace(
        go.Histogram(
            x=scored["score"][outcomes == outcome],
            name=label,
            marker_color=colors[outcome],
            opacity=0.9,
            xbins={"size": 10},
        )
    )
for threshold in (candidate.approve_below, candidate.decline_above):
    fig.add_vline(
        x=int(scale.to_score([threshold])[0]), line_dash="dot", line_color=colors["neutral"]
    )
fig.update_layout(barmode="stack")
fig.update_xaxes(title="Score")
fig.update_yaxes(title="Applications")
st.plotly_chart(style_figure(fig, height=340), width="stretch")

st.subheader("Default rate by score band")
bands = score_band_table(scored["score"].to_numpy(), y)
st.dataframe(
    bands.reset_index().rename(
        columns={
            "band": "Score band",
            "applications": "Applications",
            "share": "Share",
            "default_rate": "Observed default rate",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Share": st.column_config.NumberColumn(format="percent"),
        "Observed default rate": st.column_config.NumberColumn(format="percent"),
    },
)

st.subheader("Change the live policy")
if candidate == live:
    st.markdown(
        "<p class='muted'>Move the sliders to propose new thresholds.</p>", unsafe_allow_html=True
    )
else:
    st.markdown(
        f"Approve below **{pct(candidate.approve_below)}** (now {pct(live.approve_below)}) and "
        f"decline above **{pct(candidate.decline_above)}** (now {pct(live.decline_above)})."
    )
    with st.form("policy-change", border=True):
        a, b = st.columns([1, 2])
        changed_by = a.text_input("Approved by")
        reason = b.text_input("Reason for the change")
        if st.form_submit_button("Apply as live policy", type="primary"):
            try:
                policy_store().save(candidate, changed_by, reason)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Live policy updated. New decisions use the new thresholds.")
                st.rerun()
