import json

import streamlit as st

from core.data_quality import assess_data_health

st.title("🩺 Data Health")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

options = ["(no target)"] + dataset.df.columns
previous = st.session_state.get("target_col")
index = options.index(previous) if previous in options else 0
chosen = st.selectbox(
    "Target column (optional, enables leakage and class-imbalance checks)", options, index=index
)
target = None if chosen == "(no target)" else chosen
st.session_state["target_col"] = target

key = (dataset.path.as_posix(), dataset.mode.value, target)
entry = st.session_state.get("health")
if entry is None or entry["key"] != key:
    try:
        with st.spinner("Scanning dataset..."):
            report = assess_data_health(dataset.df, target=target)
    except Exception as exc:
        st.error(f"The health check could not be completed: {exc}")
        st.info("Check that the dataset has rows and columns, or choose a different target.")
        st.stop()
    st.session_state["health"] = {"key": key, "report": report}
    st.rerun()
report = entry["report"]

if dataset.sampled:
    st.warning(f"Based on a sample of {dataset.n_rows_loaded:,} of {dataset.total_rows:,} rows.")

left, right = st.columns([1, 3])
left.metric("Health score", f"{report.score}/100", report.label, delta_color="off")
right.progress(report.score / 100, text=report.headline)

SHOW = {"ok": st.success, "info": st.info, "warn": st.warning, "bad": st.error}

tab_findings, tab_score, tab_columns, tab_corr, tab_target = st.tabs(
    ["Findings", "Why this score", "Columns", "Correlated pairs", "Target"]
)

with tab_findings:
    for f in report.findings:
        text = f"{f.icon} **{f.title}**"
        if f.detail:
            text += f"  \n{f.detail}"
        if f.suggestion and f.severity != "ok":
            text += f"  \n*Suggestion: {f.suggestion}*"
        SHOW[f.severity](text)
    st.caption("Nothing is removed or changed automatically. You decide in Preprocessing.")

with tab_score:
    breakdown = report.breakdown()
    if breakdown.height:
        st.write("Points lost by each check (the score starts at 100):")
        st.dataframe(breakdown)
    else:
        st.success("No points were lost.")

with tab_columns:
    st.dataframe(report.columns)

with tab_corr:
    if report.correlated_pairs.height:
        st.dataframe(report.correlated_pairs)
    else:
        st.success("No highly correlated numeric pairs.")

with tab_target:
    if target is None:
        st.info("Select a target column above to see class balance and possible leakage.")
    else:
        if report.class_distribution is not None:
            st.write("Class distribution")
            st.bar_chart(report.class_distribution, x="class", y="count")
            st.dataframe(report.class_distribution)
        if report.leakage:
            st.write("Possible leakage (never certain, please investigate):")
            for c in report.leakage:
                st.warning(f"**{c.column}** ({c.strength}): {c.reason}")
        else:
            st.success("No obvious leakage signals found (this is not a proof).")

st.download_button(
    "Download health report (JSON)",
    data=json.dumps(report.to_dict(), default=str, indent=2),
    file_name="data_health_report.json",
    mime="application/json",
)