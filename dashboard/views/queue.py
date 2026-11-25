import streamlit as st
from common import audit_log, current_policy, eur, header, scored_test_set

from credit_underwriting.data.schema import ID
from credit_underwriting.features import DAYS_EMPLOYED_SENTINEL


def to_form_values(row) -> dict:
    days = row["DAYS_EMPLOYED"]
    bureau = {
        f"ext_source_{i}": None if row.isna()[f"EXT_SOURCE_{i}"] else float(row[f"EXT_SOURCE_{i}"])
        for i in (1, 2, 3)
    }
    return {
        "applicant_id": int(row[ID]),
        "income_total": float(row["AMT_INCOME_TOTAL"]),
        "credit_amount": float(row["AMT_CREDIT"]),
        "annuity": float(row["AMT_ANNUITY"]),
        "goods_price": float(row["AMT_GOODS_PRICE"]),
        "employment_years": None if days == DAYS_EMPLOYED_SENTINEL else round(-days / 365.25, 1),
        "children": int(row["CNT_CHILDREN"]),
        **bureau,
    }


title_col, action_col = st.columns([4, 1], vertical_alignment="bottom")
with title_col:
    header("Work queue", "Referrals")
with action_col:
    if st.button("New application", width="stretch"):
        st.session_state.pop("review_application", None)
        st.switch_page("views/review.py")

scored = scored_test_set()
policy = current_policy()
log = audit_log().read()
reviewed = set(log["applicant_id"].dropna().astype(int))
queue = scored[(scored["outcome"] == "refer") & ~scored[ID].isin(reviewed)]
queue = queue.sort_values("pd", ascending=False).reset_index(drop=True)
overrides = int((log["model_outcome"] != log["final_outcome"]).sum()) if len(log) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Awaiting review", f"{len(queue):,}")
c2.metric("Reviewed", f"{len(reviewed):,}")
c3.metric("Average PD in queue", f"{queue['pd'].mean():.1%}" if len(queue) else "–")
c4.metric("Model overrides", f"{overrides:,}")

st.markdown(
    "<p class='muted'>Applications between the approval and decline thresholds, highest risk "
    "first. Select a row to open it.</p>",
    unsafe_allow_html=True,
)

table = queue[[ID, "AMT_CREDIT", "AMT_INCOME_TOTAL", "pd", "score", "main_reason"]].rename(
    columns={
        ID: "Application",
        "AMT_CREDIT": "Loan amount",
        "AMT_INCOME_TOTAL": "Annual income",
        "pd": "PD",
        "score": "Score",
        "main_reason": "Main risk factor",
    }
)
selection = st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    height=420,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Application": st.column_config.NumberColumn(format="%d"),
        "Loan amount": st.column_config.NumberColumn(format="€%,d"),
        "Annual income": st.column_config.NumberColumn(format="€%,d"),
        "PD": st.column_config.ProgressColumn(
            format="percent", min_value=policy.approve_below, max_value=policy.decline_above
        ),
        "Score": st.column_config.NumberColumn(format="%d"),
    },
)

rows = selection.selection.rows
if rows:
    row = queue.iloc[rows[0]]
    st.session_state["review_application"] = to_form_values(row)
    label = f"Open application {int(row[ID])} · {eur(row['AMT_CREDIT'])}"
    if st.button(label, type="primary"):
        st.switch_page("views/review.py")

if len(log):
    st.subheader("Review log")
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
    st.download_button(
        "Download review log (CSV)",
        log.to_csv(index=False).encode(),
        file_name="klarlend_review_log.csv",
        mime="text/csv",
    )
