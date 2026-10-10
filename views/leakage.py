import streamlit as st

from core.leakage import assess_leakage, check_contamination
from core.preprocessing import PreprocessError, drop_columns
from utils.session import apply_step

st.title("🕵️ Leakage Check")
st.caption("Looks for columns that may reveal the answer to the model. Every result is only a possible "
           "leakage: you decide, because only you know when each column becomes available.")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

df = dataset.df
cols = df.columns
prev = st.session_state.get("target_col")
target = st.selectbox("Target column", cols, index=cols.index(prev) if prev in cols else None,
                      placeholder="Choose a column", key="lk_target")
if target is None:
    st.stop()
others = [c for c in cols if c != target]
features = st.multiselect("Columns to check", others, default=others, key=f"lk_feats|{dataset.name}|{target}")

flash = st.session_state.pop("lk_flash", None)
if flash:
    st.success(flash)

if st.button("Run leakage check", type="primary", disabled=not features):
    try:
        with st.spinner("Checking columns..."):
            report = assess_leakage(df, target, features)
    except ValueError as exc:
        st.error(str(exc))
        st.info("What you can do: choose a target with known values and at least one other column.")
    except Exception as exc:
        st.error(f"The check could not be completed: {exc}")
    else:
        st.session_state["leakage_report"] = {"sig": (dataset.name, df.height, target), "report": report}

entry = st.session_state.get("leakage_report")
if entry is None:
    st.stop()
report = entry["report"]
if entry["sig"] != (dataset.name, df.height, target):
    st.warning("These findings belong to an earlier selection. Run the check again.")

counts = report.counts()
a, b, c, d = st.columns(4)
a.metric("Strong signals", counts["strong"])
b.metric("Moderate", counts["moderate"])
c.metric("Weak", counts["weak"])
d.metric("Rows / columns checked", f"{report.rows_checked:,} / {report.columns_checked}")

if not report.findings:
    st.success("No leakage signals found. This is not a proof that the data is clean.")
else:
    LEVEL = {"strong": st.error, "moderate": st.warning, "weak": st.info}
    for f in report.findings[:30]:
        LEVEL[f.strength](f"**Possible leakage: {f.column}** · {f.check} ({f.strength})  \n{f.detail}")
    if len(report.findings) > 30:
        st.caption(f"Showing 30 of {len(report.findings)} findings. The table below lists all of them.")
    with st.expander("All findings (table)"):
        st.dataframe(report.table())

st.subheader("Train/test contamination")
candidates = [(label, st.session_state.get(key))
              for label, key in (("Machine Learning", "ml_run"), ("AutoML", "automl_run"))]
candidates = [(label, r) for label, r in candidates if r is not None and r.config.target == target]
if not candidates:
    st.caption("Train models for this target (Machine Learning or AutoML) to check whether test rows "
               "also appear in the training data.")
SHOW = {"ok": st.success, "moderate": st.warning, "strong": st.error}
for label, r in candidates:
    info = check_contamination(r)
    SHOW[info["level"]](
        f"{label}: {info['n']:,} of {info['rows_test']:,} test rows ({info['share']:.1%}) have exactly the "
        "same feature values as a training row. This is possible train/test contamination if they are "
        "repeats of the same record. With few features, identical rows can also be legitimate.")

st.subheader("What do you want to do?")
st.caption("Nothing is changed unless you click the button. The step can be undone in Preprocessing.")
suspects = list(dict.fromkeys(f.column for f in report.findings if f.strength == "strong"))
to_drop = st.multiselect("Columns to drop from the dataset", others,
                         default=[c for c in suspects if c in others], key=f"lk_drop|{dataset.name}|{df.width}")
if st.button("Drop selected columns", disabled=not to_drop):
    try:
        result = drop_columns(df, to_drop)
    except PreprocessError as err:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    else:
        apply_step(f"Drop {len(to_drop)} suspected leakage column(s)", result.df)
        st.session_state.pop("leakage_report", None)
        st.session_state["lk_flash"] = result.message
        st.rerun()