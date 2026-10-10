"""Streamlit caching wrappers around the UI-free core functions."""

from __future__ import annotations

import streamlit as st

from config import ProcessingMode
from core.data_loader import LoadedDataset, LoadOptions, load_dataset


@st.cache_data(show_spinner=False, max_entries=5)
def cached_load(
    path: str, delimiter: str | None, sheet: str | None, mode_value: str, display_name: str
) -> LoadedDataset:
    """Load a dataset once per (file, options, mode). Uploaded paths are content-hashed."""
    options = LoadOptions(delimiter=delimiter, sheet=sheet, mode=ProcessingMode(mode_value))
    return load_dataset(path, options, display_name=display_name)