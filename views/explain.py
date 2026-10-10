import streamlit as st

from core.explainer import TERMS, ExplainError, explain, explain_all

st.title("💬 Explain my results")
st.caption("Plain-English explanations of the numbers in your session, using your actual values.")

items = explain_all(st.session_state)
if not items:
    st.info("No results to explain yet. Run Data Health, train a model, or compare datasets first.")
    st.page_link("views/recommendations.py", label="See what to do next")
else:
    section = None
    for sec, e in items:
        if sec != section:
            st.subheader(sec)
            section = sec
        with st.container(border=True):
            st.markdown(f"**{e.term}**")
            st.write(e.meaning)
            if e.reading:
                st.info(e.reading)
            if e.caution:
                st.caption(e.caution)

st.divider()
st.subheader("Look up any term")
term = st.selectbox("Term", list(TERMS), key="ex_term")
has_value = st.checkbox("I have a value to interpret", key="ex_has")
value = st.number_input("Value", value=0.5, format="%.4f", key="ex_val") if has_value else None
try:
    e = explain(term, value)
    st.write(e.meaning)
    if e.reading:
        st.info(e.reading)
    st.caption(e.caution)
except ExplainError as err:
    st.error(err.message)
    st.info(err.suggestion)