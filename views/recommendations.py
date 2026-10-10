import streamlit as st

from core.recommendations import recommend

st.title("🧭 What should I do?")
st.caption("Suggestions come from the analyses you have already run (data health, leakage, models). "
           "Run more analyses to get more specific advice.")

recs = recommend(st.session_state)
SHOW = {"high": st.error, "medium": st.warning, "low": st.info}
TAG = {"high": "High priority", "medium": "Worth doing", "low": "Optional"}
for i, r in enumerate(recs, start=1):
    SHOW[r.priority](f"**{TAG[r.priority]}: {r.title}**  \n{r.why}")
    st.page_link(r.page, label=f"{i}. {r.label}")