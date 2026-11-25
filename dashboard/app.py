import streamlit as st
from common import ASSETS, PRODUCT, footer, inject_css

st.set_page_config(
    page_title=PRODUCT,
    page_icon=str(ASSETS / "mark.svg"),
    layout="wide",
    menu_items={"About": f"{PRODUCT}: explainable credit decisions with human oversight."},
)
inject_css()
st.logo(str(ASSETS / "logo.svg"), size="large")

pages = [
    st.Page("views/queue.py", title="Queue", default=True),
    st.Page("views/review.py", title="Review"),
    st.Page("views/batch.py", title="Batch scoring"),
    st.Page("views/portfolio.py", title="Policy"),
    st.Page("views/model.py", title="Monitoring"),
    st.Page("views/settings.py", title="Settings"),
]

navigation = st.navigation(pages, position="top")
navigation.run()
footer()
