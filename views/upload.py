from pathlib import Path

import streamlit as st

from config import ProcessingMode
from core.data_loader import (
    DELIMITERS,
    DataLoadError,
    TEXT_TYPES,
    describe_schema,
    detect_file_type,
    human_size,
    list_excel_sheets,
    save_upload,
    sniff_delimiter,
)
from core.samples import ensure_samples
from utils.caching import cached_load

st.title("📤 Data Upload")
mode: ProcessingMode = st.session_state["mode"]
st.caption(f"Processing mode: **{mode.value}**")


def load_into_session(path: Path, name: str, delimiter: str | None, sheet: str | None) -> None:
    try:
        with st.spinner("Reading file..."):
            dataset = cached_load(str(path), delimiter, sheet, mode.value, name)
    except DataLoadError as err:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
        return
    st.session_state["dataset"] = dataset
    st.session_state["target_col"] = None
    st.session_state.pop("health", None)
    st.session_state.pop("history", None)
    st.session_state.pop("eda", None)
    st.rerun()


upload_tab, sample_tab = st.tabs(["Upload a file", "Demo datasets"])

with upload_tab:
    uploaded = st.file_uploader(
        "CSV, TSV, TXT, Excel, JSON or Parquet",
        type=["csv", "tsv", "txt", "xls", "xlsx", "json", "parquet"],
    )
    if uploaded is not None:
        saved: dict = st.session_state.setdefault("_saved_uploads", {})
        if uploaded.file_id not in saved:
            try:
                with st.spinner("Saving upload..."):
                    saved[uploaded.file_id] = save_upload(uploaded.name, uploaded)
            except DataLoadError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
                st.stop()
        path: Path = saved[uploaded.file_id]

        delimiter: str | None = None
        sheet: str | None = None
        try:
            ftype = detect_file_type(path)
            if ftype in TEXT_TYPES:
                labels = list(DELIMITERS)
                guess = sniff_delimiter(path, ftype)
                default = next((i for i, lab in enumerate(labels) if DELIMITERS[lab] == guess), 0)
                choice = st.selectbox("Delimiter (auto-detected, change if wrong)", labels, index=default, key=f"delim_{uploaded.file_id}")
                delimiter = DELIMITERS[choice]
            elif ftype in {"xls", "xlsx"}:
                sheet = st.selectbox("Sheet", list_excel_sheets(path), key=f"sheet_{uploaded.file_id}")
        except DataLoadError as err:
            st.error(err.message)
            st.info(f"What you can do: {err.suggestion}")
            st.stop()

        st.caption(f"{uploaded.name} · {human_size(path.stat().st_size)}")
        if st.button("Load dataset", type="primary"):
            load_into_session(path, uploaded.name, delimiter, sheet)

with sample_tab:
    st.caption("Small synthetic datasets for trying the app.")
    for label, sample_path in ensure_samples().items():
        left, right = st.columns([4, 1])
        left.write(label)
        if right.button("Load", key=f"sample_{sample_path.name}"):
            load_into_session(sample_path, sample_path.name, None, None)

dataset = st.session_state.get("dataset")
if dataset is not None:
    st.divider()
    st.subheader(f"Loaded: {dataset.name}")
    cols = st.columns(4)
    cols[0].metric("Rows loaded", f"{dataset.n_rows_loaded:,}")
    cols[1].metric("Columns", dataset.n_cols)
    cols[2].metric("File size", human_size(dataset.file_size_bytes))
    cols[3].metric("Memory", human_size(dataset.memory_bytes))

    for message in dataset.warnings:
        st.warning(message)
    st.caption(f"Engine: **{dataset.engine}**. {dataset.engine_reason} (loaded in {dataset.load_seconds:.2f}s)")

    preview, schema = st.tabs(["Preview", "Columns and types"])
    with preview:
        st.dataframe(dataset.df.head(100))
    with schema:
        st.dataframe(describe_schema(dataset.df))
    st.page_link("views/data_health.py", label="Next: check Data Health")