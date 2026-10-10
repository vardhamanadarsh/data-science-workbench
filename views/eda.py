import plotly.express as px
import polars as pl
import streamlit as st

from core.data_loader import analysis_sample
from core.eda import build_html_report, correlation_matrix, run_eda
from utils.optional_imports import require

st.title("🔎 EDA")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

key = (dataset.path.as_posix(), dataset.mode.value)
entry = st.session_state.get("eda")
if entry is None or entry["key"] != key:
    st.info("EDA runs on the loaded rows. Distribution statistics use at most 200,000 of them.")
    if not st.button("Run EDA", type="primary"):
        st.stop()
    try:
        with st.spinner("Analysing..."):
            result = run_eda(dataset.df)
    except Exception as exc:
        st.error(f"EDA could not be completed: {exc}")
        st.info("Check that the dataset has rows and columns.")
        st.stop()
    st.session_state["eda"] = {"key": key, "result": result}
    st.rerun()
result = entry["result"]
o = result.overview

if dataset.sampled:
    st.warning(f"Based on a sample of {dataset.n_rows_loaded:,} of {dataset.total_rows:,} rows.")

tabs = st.tabs(["Overview", "Numeric", "Categorical", "Missing", "Correlation", "Distributions", "Report"])

with tabs[0]:
    cols = st.columns(4)
    cols[0].metric("Rows", f"{o['rows']:,}")
    cols[1].metric("Columns", o["columns"])
    cols[2].metric("Missing cells", f"{o['missing_cells_pct']}%")
    cols[3].metric("Duplicate rows", f"{o['duplicate_rows']:,}")
    st.write("Column types")
    st.dataframe(pl.DataFrame({"dtype": list(o["dtypes"]), "columns": list(o["dtypes"].values())}))

with tabs[1]:
    if result.numeric.height:
        st.dataframe(result.numeric)
    else:
        st.info("No numeric columns.")

with tabs[2]:
    if result.categorical.height:
        st.dataframe(result.categorical)
    else:
        st.info("No categorical columns.")

with tabs[3]:
    gaps = result.missing.filter(pl.col("missing") > 0)
    if gaps.height:
        st.plotly_chart(px.bar(gaps.to_pandas(), x="column", y="missing_pct",
                               labels={"missing_pct": "Missing (%)"}))
        st.dataframe(gaps)
    else:
        st.success("No missing values.")

with tabs[4]:
    method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
    with st.spinner("Computing correlations..."):
        corr = correlation_matrix(dataset.df, method=method)
    if corr is None:
        st.info("At least two numeric columns are needed.")
    else:
        st.plotly_chart(px.imshow(corr, text_auto=".2f" if corr.shape[0] <= 15 else False,
                                  zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto"))

with tabs[5]:
    numeric_cols = result.numeric["column"].to_list()
    if not numeric_cols:
        st.info("No numeric columns.")
    else:
        chosen = st.selectbox("Variable", numeric_cols)
        sample = analysis_sample(dataset.df.select(chosen), st.session_state["mode"])
        st.plotly_chart(px.histogram(sample.to_pandas(), x=chosen, nbins=40, marginal="box"))
        st.caption(f"Based on {sample.height:,} rows. Use Plot Studio for more plot types.")

with tabs[6]:
    st.download_button(
        "Download basic EDA report (HTML)",
        data=build_html_report(dataset.name, result),
        file_name="eda_report.html",
        mime="text/html",
    )
    st.divider()
    st.write("Optional: full YData Profiling report (slow; uses up to 50,000 rows).")
    if st.button("Generate YData Profiling report"):
        module = require("ydata_profiling")
        if module is not None:
            try:
                with st.spinner("Profiling..."):
                    frame = analysis_sample(dataset.df, st.session_state["mode"])
                    frame = frame.sample(n=min(frame.height, 50_000), seed=42)
                    profile = module.ProfileReport(frame.to_pandas(), title=dataset.name, minimal=True)
                    st.session_state["ydata_html"] = profile.to_html()
            except Exception as exc:
                st.error(f"Profiling failed: {exc}")
    if "ydata_html" in st.session_state:
        st.download_button("Download YData report (HTML)", st.session_state["ydata_html"],
                           file_name="ydata_report.html", mime="text/html")