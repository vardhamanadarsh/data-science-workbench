"""A small help button that explains a metric in plain English."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.explainer import ExplainError, explain


def help_popover(term: str, value: float | None = None, **context: Any) -> None:
    """Show a '❓' button that opens a plain-English explanation of `term`."""
    try:
        e = explain(term, value, **context)
    except ExplainError:
        return
    with st.popover("❓", help=f"What does {e.term} mean?"):
        st.markdown(f"**{e.term}**")
        st.write(e.meaning)
        if e.reading:
            st.info(e.reading)
        if e.caution:
            st.caption(e.caution)