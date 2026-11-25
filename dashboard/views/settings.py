import pandas as pd
import streamlit as st
from common import PRODUCT, current_policy, header, load_model, pct, policy_store

from credit_underwriting import __version__
from credit_underwriting.api.settings import Settings
from credit_underwriting.data.schema import FEATURE_NAMES, FEATURES

header("Administration", "Settings")

settings = Settings.from_env()
_, version = load_model()
policy = current_policy()

st.subheader("Decision policy")
a, b, c = st.columns(3)
a.metric("Approve below", pct(policy.approve_below))
b.metric("Decline above", pct(policy.decline_above))
c.metric("Referral band", f"{pct(policy.approve_below)} – {pct(policy.decline_above)}")
st.markdown(
    "<p class='muted'>Change thresholds on the Policy page. Every change is recorded with the "
    "approver and a reason. Environment variables <code>APPROVE_BELOW</code> and "
    "<code>DECLINE_ABOVE</code> override the stored policy.</p>",
    unsafe_allow_html=True,
)

history = policy_store().history()
if history:
    st.markdown("**Change history**")
    st.dataframe(
        pd.DataFrame([vars(h) for h in history]).iloc[::-1],
        hide_index=True,
        width="stretch",
        column_config={
            "approve_below": st.column_config.NumberColumn("Approve below", format="percent"),
            "decline_above": st.column_config.NumberColumn("Decline above", format="percent"),
            "changed_by": "Approved by",
            "reason": "Reason",
            "timestamp": "Time (UTC)",
        },
    )

st.subheader("Model")
st.dataframe(
    pd.DataFrame(
        {
            "Property": ["Version", "File", "Algorithm", "Inputs"],
            "Value": [
                version,
                str(settings.model_path),
                "LightGBM with monotonic constraints",
                ", ".join(FEATURE_NAMES[f] for f in FEATURES),
            ],
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Appearance")
st.markdown(
    "Klarlend follows your system's light or dark mode. To switch manually, open the menu "
    "(⋮, top right) and choose **Light** or **Dark**."
)

st.subheader("About")
st.markdown(
    f"{PRODUCT} {__version__}. Explainable credit decisions with human oversight: probability of "
    "default, reasons for every adverse decision, a review workflow with an audit trail, and a "
    "REST API (`/v1/decisions`) for integration with loan origination systems."
)
