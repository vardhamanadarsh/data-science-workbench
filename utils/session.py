"""Session helpers shared by pages that change the dataset."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import streamlit as st

from core.history import DataHistory

_STALE_WIDGET_PREFIXES = ("ml_feats|", "am_feats|", "lk_feats|", "lk_drop|", "an_cols|")


def apply_step(label: str, new_df: pl.DataFrame) -> None:
    """Record a dataset change in the undo history and make the app use it."""
    dataset = st.session_state["dataset"]
    history = st.session_state.get("history")
    if history is None:
        history = DataHistory(dataset.df)
        st.session_state["history"] = history
    history.push(label, new_df)
    st.session_state["dataset"] = replace(dataset, df=history.current.df)
    for key in ("health", "eda"):
        st.session_state.pop(key, None)
    # Column pickers remember old selections; clear them so removed columns cannot cause errors.
    for key in [k for k in st.session_state.keys() if str(k).startswith(_STALE_WIDGET_PREFIXES)]:
        del st.session_state[key]