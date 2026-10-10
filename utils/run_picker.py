"""Shared 'which results and which model?' picker for pages that use trained models."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from core.model_training import ModelResult, TrainingRun, best_result, ranked


def pick_model(prefix: str, predicate: Callable[[ModelResult], bool]) -> tuple[TrainingRun, ModelResult]:
    """Show the pickers and return (run, model result). Stops the page if nothing is available."""
    sources = {label: st.session_state.get(key)
               for label, key in (("Machine Learning", "ml_run"), ("AutoML", "automl_run"))}
    sources = {k: v for k, v in sources.items() if v is not None}
    if not sources:
        st.info("Train models first in Machine Learning or AutoML.")
        st.page_link("views/machine_learning.py", label="Go to Machine Learning")
        st.stop()
    source = (st.selectbox("Results from", list(sources), key=f"{prefix}_source")
              if len(sources) > 1 else next(iter(sources)))
    run = sources[source]
    models = [r for r in ranked(run) if r.ok and r.name != "Baseline" and predicate(r)]
    if not models:
        st.info("None of the trained models can be used on this page.")
        st.stop()
    best = best_result(run)
    names = [r.name for r in models]
    index = names.index(best.name) if best is not None and best.name in names else 0
    name = st.selectbox("Model", names, index=index, key=f"{prefix}_model|{run.created}|{source}")
    return run, next(r for r in models if r.name == name)