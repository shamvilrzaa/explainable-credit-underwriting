import pandas as pd
import streamlit as st
from common import OUTCOME_LABELS, get_underwriter, header, pct

from credit_underwriting.batch import score_batch, template

header(
    "Operations",
    "Batch scoring",
    "Upload a CSV of applications to score them in one go. Each row gets a decision, "
    "probability of default, score and reasons. Gender and age are not accepted as inputs.",
)

left, right = st.columns([3, 1], vertical_alignment="bottom")
upload = left.file_uploader("Applications (CSV)", type="csv")
right.download_button(
    "Download template",
    template().to_csv(index=False).encode(),
    file_name="klarlend_template.csv",
    mime="text/csv",
    width="stretch",
)

if upload is None:
    st.stop()

try:
    result = score_batch(pd.read_csv(upload), get_underwriter())
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"The file could not be processed: {error}")
    st.stop()

decisions = result.decisions.assign(outcome=result.decisions["outcome"].map(OUTCOME_LABELS))
if len(decisions):
    counts = result.decisions["outcome"].value_counts(normalize=True)
    cols = st.columns(4)
    cols[0].metric("Scored", f"{len(decisions):,}")
    for col, outcome in zip(cols[1:], OUTCOME_LABELS, strict=True):
        col.metric(OUTCOME_LABELS[outcome], pct(counts.get(outcome, 0.0)))

    st.dataframe(
        decisions,
        hide_index=True,
        width="stretch",
        column_config={
            "applicant_id": st.column_config.NumberColumn("Application", format="%d"),
            "outcome": "Outcome",
            "probability_of_default": st.column_config.NumberColumn("PD", format="percent"),
            "score": "Score",
            "reasons": "Reasons",
        },
    )
    st.download_button(
        "Download decisions (CSV)",
        result.decisions.to_csv(index=False).encode(),
        file_name="klarlend_decisions.csv",
        mime="text/csv",
        type="primary",
    )

if len(result.errors):
    count = len(result.errors)
    st.subheader(f"{count} {'row' if count == 1 else 'rows'} could not be scored")
    st.dataframe(
        result.errors.rename(columns={"row": "Row", "field": "Field", "problem": "Problem"}),
        hide_index=True,
        width="stretch",
    )
