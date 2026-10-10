import streamlit as st

from core.automl import flaml_available, leaderboard_notes, native_automl, run_flaml
from core.data_quality import class_distribution, imbalance_level
from core.model_training import (
    IMBALANCE_OPTIONS, ML_ROW_CAPS, MLError, TrainConfig, best_result, detect_task,
    eligible_features, ranked, results_table,
)
from utils.optional_imports import install_hint

st.title("⚡ AutoML")
st.caption("Compares every available model on one split. Nothing runs until you click a button.")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

df = dataset.df
mode = st.session_state["mode"]
cols = df.columns

prev = st.session_state.get("target_col")
target = st.selectbox("Target column", cols, index=cols.index(prev) if prev in cols else None,
                      placeholder="Choose a column", key="am_target")
if target is None:
    st.stop()

sig = (dataset.name, df.height, tuple(cols), target)
cache = st.session_state.get("am_cache")
if cache is None or cache["sig"] != sig:
    feats, skipped = eligible_features(df, target)
    cache = {"sig": sig, "features": feats, "skipped": skipped}
    st.session_state["am_cache"] = cache

choice = st.radio("Problem type", ["Auto-detect", "Classification", "Regression"],
                  horizontal=True, key="am_task_choice")
task = detect_task(df[target]) if choice == "Auto-detect" else choice.lower()
if task == "regression" and not df[target].dtype.is_numeric():
    st.error("A regression target must be numeric.")
    st.info("What you can do: choose Classification, or pick a numeric target.")
    st.stop()
clf = task == "classification"
st.info(f"{'Classification' if clf else 'Regression'} problem "
        f"{'detected' if choice == 'Auto-detect' else 'selected'}.")

positive: str | None = None
if clf:
    dist = class_distribution(df[target])
    level, ratio = imbalance_level(dist)
    if level in ("severe", "moderate"):
        st.warning(f"Class imbalance detected (ratio {ratio:.1f}:1). Consider class weights below.")
    if dist.height == 2:
        positive = st.selectbox("Positive class", dist["class"].to_list(), index=1, key=f"am_pos|{target}")

features = st.multiselect("Features", cache["features"], default=cache["features"],
                          key=f"am_feats|{dataset.name}|{target}")
if cache["skipped"]:
    with st.expander(f"{len(cache['skipped'])} column(s) left out automatically"):
        for name, reason in cache["skipped"].items():
            st.write(f"**{name}**: {reason}")

with st.expander("Settings", expanded=False):
    a, b, c = st.columns(3)
    test_size = a.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="am_test")
    cv_folds = b.slider("Cross-validation folds (0 = off)", 0, 5, 3, key="am_cv")
    parallel = c.checkbox("Parallel (all CPU cores)", value=True, key="am_parallel")
    time_split = st.checkbox("Time-based split (the last rows become the test set)", key="am_time")
    imbalance = "none"
    if clf:
        imbalance = st.selectbox("Class imbalance", list(IMBALANCE_OPTIONS),
                                 format_func=IMBALANCE_OPTIONS.get, key="am_imb")

cap = ML_ROW_CAPS[mode]
if cap and df.height > cap:
    st.info(f"'{mode.value}' trains on a sample of {cap:,} rows.")

if st.button("Run native comparison", type="primary", disabled=not features):
    cfg = TrainConfig(target=target, features=features, task=task, test_size=test_size,
                      cv_folds=cv_folds, time_split=time_split, imbalance=imbalance,
                      positive_class=positive, parallel=parallel, max_rows=cap)
    bar = st.progress(0.0, text="Preparing data...")

    def on_progress(i: int, total: int, name: str) -> None:
        bar.progress(min(i / total, 1.0), text="Finished" if name == "done" else f"Training {name} ({i + 1} of {total})")

    try:
        new_run = native_automl(df, cfg, on_progress)
    except MLError as err:
        bar.empty()
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    except Exception as exc:
        bar.empty()
        st.error(f"The comparison could not be completed: {exc}")
    else:
        new_run.signature = (dataset.name, df.height, df.width)
        st.session_state["automl_run"] = new_run
        bar.empty()

run = st.session_state.get("automl_run")
if run is None:
    st.stop()

st.divider()
st.subheader("Results")
flash = st.session_state.pop("am_flash", None)
if flash:
    st.success(flash)
if run.signature != (dataset.name, df.height, df.width):
    st.warning("These results were trained on an earlier version of the data. Run the comparison again.")
for note in run.notes:
    st.caption(f"ℹ {note}")

st.dataframe(results_table(run))
for line in leaderboard_notes(run):
    st.write(f"• {line}")

ok = [r for r in ranked(run) if r.ok]
if ok:
    best = best_result(run)
    names = [r.name for r in ok if r.name != "Baseline"] or [r.name for r in ok]
    pick = st.selectbox("Choose the model to use", names,
                        index=names.index(best.name) if best is not None and best.name in names else 0,
                        key="am_pick")
    left, right = st.columns(2)
    if left.button("Use this model"):
        st.session_state["selected_model"] = pick
        st.success(f"Selected: {pick}")
    if right.button("Open these results in Machine Learning"):
        st.session_state["ml_run"] = run
        st.switch_page("views/machine_learning.py")

st.divider()
st.subheader("FLAML (optional)")
if not flaml_available():
    st.info(install_hint("flaml"))
    st.code("uv sync --extra automl", language="bash")
else:
    budget = st.slider("Time budget (seconds)", 30, 600, 60, 30, key="am_budget")
    st.caption("FLAML uses the whole time budget. It trains on the same split as the models above "
               "(at most 50,000 training rows), so the test scores are comparable.")
    if st.button("Run FLAML"):
        try:
            with st.spinner(f"FLAML is searching for up to {budget} seconds..."):
                res = run_flaml(run, budget)
        except MLError as err:
            st.error(err.message)
            if err.suggestion:
                st.info(f"What you can do: {err.suggestion}")
        except Exception as exc:
            st.error(f"FLAML failed: {exc}")
        else:
            st.session_state["am_flash"] = f"Added '{res.name}' to the leaderboard."
            st.rerun()