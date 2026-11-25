import plotly.graph_objects as go
import streamlit as st
from common import (
    OUTCOME_LABELS,
    audit_log,
    eur,
    get_underwriter,
    header,
    load_model,
    outcome_color,
    palette,
    pct,
    style_figure,
)
from pydantic import ValidationError

from credit_underwriting.api.schemas import Application, to_frame
from credit_underwriting.audit import ReviewRecord
from credit_underwriting.counterfactual import max_affordable_loan
from credit_underwriting.data.schema import FEATURE_NAMES

EXAMPLE = {
    "applicant_id": None,
    "income_total": 54_000.0,
    "credit_amount": 270_000.0,
    "annuity": 18_500.0,
    "goods_price": 250_000.0,
    "employment_years": 3.5,
    "children": 1,
    "ext_source_1": None,
    "ext_source_2": 0.42,
    "ext_source_3": 0.35,
}

pipeline, version = load_model()
underwriter = get_underwriter()
values = st.session_state.get("review_application", EXAMPLE)
key = str(values["applicant_id"] or "manual")

title = f"Application {values['applicant_id']}" if values["applicant_id"] else "New application"
header("Credit decision", title)

form_col, result_col = st.columns([5, 7], gap="large")

with form_col, st.form(f"application-{key}", border=False):
    st.subheader("Loan")
    a, b = st.columns(2)
    credit_amount = a.number_input(
        "Loan amount (€)",
        min_value=1_000.0,
        step=5_000.0,
        value=values["credit_amount"],
        format="%.0f",
    )
    annuity = b.number_input(
        "Annual repayment (€)", min_value=100.0, step=500.0, value=values["annuity"], format="%.0f"
    )
    goods_price = st.number_input(
        "Price of financed goods (€)",
        min_value=0.0,
        step=5_000.0,
        value=values["goods_price"],
        format="%.0f",
    )

    st.subheader("Applicant")
    a, b = st.columns(2)
    income_total = a.number_input(
        "Annual income (€)",
        min_value=1_000.0,
        step=1_000.0,
        value=values["income_total"],
        format="%.0f",
    )
    children = b.number_input("Children", min_value=0, max_value=20, value=values["children"])
    employed = st.toggle("Currently employed", value=values["employment_years"] is not None)
    employment_years = st.number_input(
        "Years with current employer",
        min_value=0.0,
        max_value=60.0,
        step=0.5,
        value=values["employment_years"] or 0.0,
        format="%.1f",
        disabled=not employed,
    )

    st.subheader("Credit bureau")
    bureau = st.columns(3)
    scores = [
        col.number_input(
            f"Score {i}",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=values[f"ext_source_{i}"],
            placeholder="n/a",
        )
        for i, col in enumerate(bureau, start=1)
    ]
    st.form_submit_button("Assess application", type="primary", width="stretch")

try:
    application = Application(
        applicant_id=values["applicant_id"],
        income_total=income_total,
        credit_amount=credit_amount,
        annuity=annuity,
        goods_price=goods_price or None,
        employment_years=employment_years if employed else None,
        children=int(children),
        ext_source_1=scores[0],
        ext_source_2=scores[1],
        ext_source_3=scores[2],
    )
except ValidationError as error:
    result_col.error(f"Please check the input: {error.errors()[0]['msg']}")
    st.stop()

frame = to_frame([application])
decision = underwriter.decide(frame)[0]
contributions = underwriter.explainer.contributions(frame).iloc[0]
color = outcome_color(decision.outcome.value)
colors = palette()

with result_col:
    st.markdown(
        f"<div class='decision' style='border-color:{color};background:{color}1A'>"
        f"<div class='eyebrow'>Recommendation</div>"
        f"<div class='value' style='color:{color}'>{OUTCOME_LABELS[decision.outcome.value]}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of default", pct(decision.probability_of_default))
    m2.metric("Score", decision.score)
    m3.metric("Repayment / income", pct(annuity / income_total, 0))

    if decision.reasons:
        st.subheader("Main risk factors")
        rows = "".join(
            f"<div class='reason'><span>{r.label}</span>"
            f"<span class='detail'>+{r.contribution:.2f}</span></div>"
            for r in decision.reasons
        )
        st.markdown(rows, unsafe_allow_html=True)

        policy = underwriter.policy
        affordable = max_affordable_loan(pipeline, frame, target_pd=policy.approve_below)
        st.subheader("What would change the outcome")
        if affordable:
            st.markdown(
                f"A loan of up to **{eur(affordable.max_amount)}** "
                f"({pct(affordable.reduction, 0)} less than requested) at the same term "
                f"would be approved, with a PD of {pct(affordable.probability_of_default)}.",
            )
        else:
            st.markdown("No smaller loan at the same term would meet the approval threshold.")

    st.subheader("How the model reached this result")
    ordered = contributions.reindex(contributions.abs().sort_values().index)
    fig = go.Figure(
        go.Bar(
            x=ordered.to_numpy(),
            y=[FEATURE_NAMES[f] for f in ordered.index],
            orientation="h",
            marker_color=[colors["decline"] if v > 0 else colors["approve"] for v in ordered],
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(
        title="Effect on risk (log-odds)", zeroline=True, zerolinecolor=colors["neutral"]
    )
    st.plotly_chart(style_figure(fig, height=300), width="stretch")
    st.caption("Red bars increase the risk estimate, green bars reduce it.")

    if decision.outcome.value != "approve":
        st.subheader("Underwriter decision")
        with st.form(f"review-{key}", border=True):
            final = st.radio(
                "Final decision", ["approve", "decline"], horizontal=True, format_func=str.title
            )
            reviewer = st.text_input("Reviewer")
            note = st.text_area("Rationale", placeholder="Documented reason for the decision")
            if st.form_submit_button("Record decision"):
                try:
                    audit_log().record(
                        ReviewRecord(
                            applicant_id=application.applicant_id,
                            model_version=version,
                            model_outcome=decision.outcome.value,
                            probability_of_default=decision.probability_of_default,
                            final_outcome=final,
                            reviewer=reviewer,
                            note=note,
                        )
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("Decision recorded in the review log.")
