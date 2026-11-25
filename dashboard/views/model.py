import plotly.graph_objects as go
import streamlit as st
from common import header, load_model, palette, pct, scored_test_set, style_figure

from credit_underwriting.data.schema import FEATURE_NAMES, FEATURES, TARGET
from credit_underwriting.fairness import audit
from credit_underwriting.metrics import calibration_table, evaluate

header("Model governance", "Model monitoring")
colors = palette()

_, version = load_model()
scored = scored_test_set()
y = scored[TARGET].to_numpy()
pd_hat = scored["pd"].to_numpy()
report = evaluate(y, pd_hat)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Gini", f"{report.gini:.3f}")
m2.metric("KS", f"{report.ks:.3f}")
m3.metric("Predicted default rate", pct(report.mean_predicted_pd))
m4.metric("Observed default rate", pct(report.default_rate))
st.caption(f"Model {version} · hold-out test set of {report.n:,} applications")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Calibration")
    table = calibration_table(y, pd_hat)
    fig = go.Figure()
    upper = float(max(table["predicted_pd"].max(), table["observed_default_rate"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, upper],
            y=[0, upper],
            mode="lines",
            line={"dash": "dot", "color": colors["neutral"]},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["predicted_pd"],
            y=table["observed_default_rate"],
            mode="lines+markers",
            line={"color": colors["accent"], "width": 2},
            name="Risk deciles",
            hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted PD", tickformat=".0%")
    fig.update_yaxes(title="Observed default rate", tickformat=".0%")
    st.plotly_chart(style_figure(fig), width="stretch")

with right:
    st.subheader("Monotonic constraints")
    direction = {1: "Risk increases", -1: "Risk decreases", 0: "Unconstrained"}
    st.dataframe(
        [{"Feature": FEATURE_NAMES[f], "Constraint": direction[d]} for f, d in FEATURES.items()],
        hide_index=True,
        width="stretch",
    )

st.subheader("Fairness")
approved = scored["outcome"].eq("approve").to_numpy()
results = audit(scored, y, pd_hat, approved)
for attribute, result in results.items():
    st.markdown(f"**{attribute.replace('_', ' ').capitalize()}**")
    table = result["table"][
        ["n", "default_rate", "mean_predicted_pd", "approval_rate", "approval_ratio"]
    ]
    st.dataframe(
        table.reset_index(names="Group").rename(
            columns={
                "n": "Applications",
                "default_rate": "Observed default rate",
                "mean_predicted_pd": "Predicted PD",
                "approval_rate": "Approval rate",
                "approval_ratio": "Approval ratio",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="percent")
            for c in ["Observed default rate", "Predicted PD", "Approval rate"]
        }
        | {"Approval ratio": st.column_config.NumberColumn(format="%.2f")},
    )
    for issue in result["issues"]:
        st.warning(issue)
    if not result["issues"]:
        st.success("No issues above the audit thresholds.")
