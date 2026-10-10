import streamlit as st

from core.data_loader import DataLoadError, human_size, save_upload
from core.drift import (
    PSI_HIGH, PSI_MODERATE, CompareError, compare_datasets, distribution_fig, psi_fig,
)
from core.samples import ensure_samples
from utils.caching import cached_load

st.title("🔀 Compare & Drift")
st.caption("Compares two datasets: columns, missing values, summary statistics and distribution changes. "
           "Useful for checking whether new data still looks like the data a model was built on.")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()
mode = st.session_state["mode"]


def show_error(err: Exception) -> None:
    st.error(getattr(err, "message", str(err)))
    if getattr(err, "suggestion", ""):
        st.info(f"What you can do: {err.suggestion}")


source = st.radio("Compare the current dataset with", ["Another file", "A demo dataset", "Its original version (before preprocessing)"],
                  key="cd_source")
ref = new = None
ref_name = new_name = ""

if source == "Another file":
    up = st.file_uploader("Second file", type=["csv", "tsv", "txt", "xls", "xlsx", "json", "parquet"], key="cd_upload")
    if up is not None:
        saved: dict = st.session_state.setdefault("_saved_uploads", {})
        try:
            if up.file_id not in saved:
                saved[up.file_id] = save_upload(up.name, up)
            other = cached_load(str(saved[up.file_id]), None, None, mode.value, up.name)
        except DataLoadError as err:
            show_error(err)
            st.stop()
        ref, new, ref_name, new_name = dataset.df, other.df, dataset.name, up.name
        st.caption(f"Reference = the current dataset. New = {up.name} ({human_size(other.file_size_bytes)}). "
                   "The delimiter is detected automatically.")
elif source == "A demo dataset":
    samples = ensure_samples()
    label = st.selectbox("Demo dataset", list(samples), key="cd_demo")
    try:
        other = cached_load(str(samples[label]), None, None, mode.value, samples[label].name)
    except DataLoadError as err:
        show_error(err)
        st.stop()
    ref, new, ref_name, new_name = dataset.df, other.df, dataset.name, samples[label].name
    st.caption("Reference = the current dataset. New = the demo dataset.")
else:
    history = st.session_state.get("history")
    if history is None or history.index == 0:
        st.info("No preprocessing steps have been applied yet, so the current data is still the original.")
        st.page_link("views/preprocessing.py", label="Go to Preprocessing")
        st.stop()
    ref, new = history.snapshots[0].df, dataset.df
    ref_name, new_name = "Original data", "Current data"
    st.caption("Reference = the data as loaded. New = the data after your preprocessing steps.")

if ref is None:
    st.info("Choose the data to compare with.")
    st.stop()

bins = st.slider("Bins for numeric PSI", 5, 20, 10, key="cd_bins",
                 help="Bin edges come from the reference quantiles. 10 is the usual choice.")
sig = (ref_name, new_name, ref.shape, new.shape, bins)
if st.button("Compare", type="primary"):
    try:
        with st.spinner("Comparing..."):
            report = compare_datasets(ref, new, bins)
    except CompareError as err:
        show_error(err)
    except Exception as exc:
        st.error(f"The comparison failed: {exc}")
    else:
        st.session_state["compare_result"] = {"sig": sig, "report": report, "ref": ref, "new": new}

entry = st.session_state.get("compare_result")
if entry is None:
    st.stop()
if entry["sig"] != sig:
    st.info("The selection changed. Click Compare to update the results.")
    st.stop()
report, ref, new = entry["report"], entry["ref"], entry["new"]

st.divider()
SHOW = {"ok": st.success, "info": st.info, "warn": st.warning}
for level, text in report.findings:
    SHOW[level](text)
m = st.columns(4)
m[0].metric("Reference rows", f"{report.ref_rows:,}")
m[1].metric("New rows", f"{report.new_rows:,}")
m[2].metric("Reference columns", report.ref_cols)
m[3].metric("New columns", report.new_cols)

tab_schema, tab_drift, tab_missing, tab_summary, tab_cat = st.tabs(
    ["Schema", "Distribution drift", "Missing values", "Summary statistics", "Categories"])

with tab_schema:
    st.dataframe(report.schema)

with tab_drift:
    d = report.drift
    st.dataframe(d)
    if d["PSI"].notna().any():
        st.plotly_chart(psi_fig(d))
    st.caption(f"PSI below {PSI_MODERATE} is a small shift, {PSI_MODERATE} to {PSI_HIGH} moderate, above {PSI_HIGH} large "
               "(rules of thumb, not laws). The KS test is shown for reference: with many rows it flags even "
               "tiny shifts, so the level is based on PSI.")
    testable = d.loc[d["type"].isin(["numeric", "categorical"]), "column"].tolist()
    if testable:
        col = st.selectbox("Compare the distribution of", testable, key="cd_col")
        try:
            st.plotly_chart(distribution_fig(ref, new, col))
        except CompareError as err:
            show_error(err)
    st.download_button("Download drift table (CSV)", d.to_csv(index=False), file_name="drift.csv", mime="text/csv")

with tab_missing:
    st.dataframe(report.missing)
    st.caption("Change in percentage points. Uses all rows.")

with tab_summary:
    if report.summary.empty:
        st.info("No numeric columns in common.")
    else:
        st.dataframe(report.summary)

with tab_cat:
    if report.categories.empty:
        st.success("No categories appeared or disappeared in the shared categorical columns.")
    else:
        st.dataframe(report.categories)