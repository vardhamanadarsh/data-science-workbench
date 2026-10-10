import json

import plotly.express as px
import streamlit as st

from core.experiments import (
    compare_hyperparameters, delete_experiment, experiments_table, list_experiments,
)

st.title("🧪 Experiments")

records = list_experiments()
if not records:
    st.info("No experiments saved yet. Train models, then use the 'Save experiment' tab in Machine Learning.")
    st.page_link("views/machine_learning.py", label="Go to Machine Learning")
    st.stop()

table = experiments_table(records)
st.dataframe(table)
st.caption("Experiments are saved as JSON files in the 'experiments' folder next to app.py.")

st.subheader("Compare")
ids = st.multiselect("Choose two or more experiments", table["id"].tolist(), key="exp_compare")
chosen = [r for r in records if r["id"] in ids]
if len(chosen) >= 2:
    sub = experiments_table(chosen)
    st.dataframe(sub)
    metrics = [c for c in sub.columns if c.startswith("Test ") and sub[c].notna().any()]
    if metrics:
        metric = st.selectbox("Metric", metrics, key="exp_metric")
        st.plotly_chart(px.bar(sub, x="id", y=metric, color="model"))
    diff = compare_hyperparameters(chosen)
    st.write("Hyperparameters that differ")
    if diff.empty:
        st.caption("The chosen experiments use identical hyperparameters.")
    else:
        st.dataframe(diff)
    for r in chosen:
        with st.expander(f"{r['id']}: preprocessing and features"):
            st.write("Preprocessing steps:", r.get("preprocessing_steps") or "none")
            st.write("Features:", ", ".join(r.get("features", [])))
elif ids:
    st.caption("Choose at least two experiments to compare.")

st.subheader("Details")
selected = st.selectbox("Experiment", table["id"].tolist(), key="exp_detail")
record = next(r for r in records if r["id"] == selected)
if record.get("note"):
    st.write(f"Note: {record['note']}")
st.json(record, expanded=False)
st.download_button("Download as JSON", json.dumps(record, indent=2, default=str),
                   file_name=f"{selected}.json", mime="application/json")
confirm = st.checkbox("I understand that deleting cannot be undone", key="exp_confirm")
if st.button("Delete this experiment", disabled=not confirm):
    delete_experiment(selected)
    st.rerun()