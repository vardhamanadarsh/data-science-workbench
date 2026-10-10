from dataclasses import asdict

import streamlit as st

from core.data_quality import class_distribution, find_leakage_candidates, imbalance_level
from core.experiments import save_experiment
from core.model_diagnostics import calibration_fig, learning_curve_fig
from core.model_evaluation import (
    actual_vs_predicted_fig, confusion_matrix_fig, importance_fig, permutation_table,
    pr_fig, residual_fig, roc_fig,
)
from core.model_training import (
    IMBALANCE_OPTIONS, ML_ROW_CAPS, SLOW_MAX_ROWS, MLError, TrainConfig, best_result,
    detect_task, eligible_features, model_specs, ranked, results_table, train_models,
)
from core.tuning import history_fig, is_tunable, param_importance_fig, tune_model
from utils.optional_imports import install_hint, is_available

st.title("🤖 Machine Learning")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

df = dataset.df
mode = st.session_state["mode"]
cols = df.columns

# ---- Target and problem type ---------------------------------------------
prev = st.session_state.get("target_col")
target = st.selectbox(
    "Target column (what you want to predict)", cols,
    index=cols.index(prev) if prev in cols else None,
    placeholder="Choose a column", key="ml_target",
)
if target is None:
    st.stop()

sig = (dataset.name, df.height, tuple(cols), target)
cache = st.session_state.get("ml_cache")
if cache is None or cache["sig"] != sig:
    feats, skipped = eligible_features(df, target)
    try:
        leaks = find_leakage_candidates(df, target)
    except Exception:
        leaks = []
    cache = {"sig": sig, "features": feats, "skipped": skipped, "leak": leaks}
    st.session_state["ml_cache"] = cache

auto_task = detect_task(df[target])
choice = st.radio("Problem type", ["Auto-detect", "Classification", "Regression"],
                  horizontal=True, key="ml_task_choice")
task = auto_task if choice == "Auto-detect" else choice.lower()
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
    with st.expander(f"Class distribution ({dist.height} classes)"):
        st.dataframe(dist.head(20))
    if level == "severe":
        st.warning(f"Severe class imbalance (ratio {ratio:.0f}:1). Consider class weights or resampling "
                   "below, and look at F1 and ROC-AUC rather than accuracy.")
    elif level == "moderate":
        st.warning(f"Class imbalance detected (ratio {ratio:.1f}:1). Class weights may help.")
    if dist.height == 2:
        names = dist["class"].to_list()
        positive = st.selectbox("Positive class (used for precision, recall and F1)", names,
                                index=1, key=f"ml_pos|{target}")

# ---- Features -------------------------------------------------------------
features = st.multiselect("Features", cache["features"], default=cache["features"],
                          key=f"ml_feats|{dataset.name}|{target}")
if cache["skipped"]:
    with st.expander(f"{len(cache['skipped'])} column(s) left out automatically"):
        for name, reason in cache["skipped"].items():
            st.write(f"**{name}**: {reason}")
for cand in cache["leak"]:
    if cand.strength == "strong" and cand.column in features:
        st.warning(f"Possible leakage: **{cand.column}**. {cand.reason} Consider removing it from the features.")

# ---- Settings -------------------------------------------------------------
with st.expander("Training settings", expanded=True):
    a, b, c = st.columns(3)
    test_size = a.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="ml_test")
    cv_folds = b.slider("Cross-validation folds (0 = off)", 0, 10, 5, key="ml_cv")
    pct = c.slider("Keep top % of features (100 = all)", 10, 100, 100, 10, key="ml_pct",
                   help="Selection is fitted on training rows only.")
    d, e, f = st.columns(3)
    scale = d.checkbox("Scale numeric features", value=True, key="ml_scale")
    num_impute = e.selectbox("Fill missing numbers with", ["median", "mean"], key="ml_impute")
    parallel = f.checkbox("Parallel (all CPU cores)", value=True, key="ml_parallel")
    time_split = st.checkbox("Time-based split (the last rows become the test set)", key="ml_time",
                             help="Use for ordered data. Rows must already be sorted by time.")
    imbalance = "none"
    if clf:
        imbalance = st.selectbox("Class imbalance", list(IMBALANCE_OPTIONS),
                                 format_func=IMBALANCE_OPTIONS.get, key="ml_imb",
                                 help="Resampling needs: uv sync --extra ml. It only touches training rows.")

specs = model_specs(task)
available = [s for s in specs if s.available]
missing_specs = [s for s in specs if not s.available]
chosen = st.multiselect("Models (a Baseline is always added for comparison)", [s.name for s in available],
                        default=[s.name for s in available if s.default], key=f"ml_models|{task}")
if missing_specs:
    st.caption(f"Not installed: {', '.join(s.name for s in missing_specs)}. Install with `uv sync --extra ml`.")
if any(s.slow and s.name in chosen for s in specs):
    st.caption(f"SVM/SVR are skipped above {SLOW_MAX_ROWS:,} training rows.")

cap = ML_ROW_CAPS[mode]
if cap and df.height > cap:
    st.info(f"'{mode.value}' trains on a sample of {cap:,} rows. Switch to Full Analysis (and reload the file) to use everything.")

# ---- Train ----------------------------------------------------------------
if st.button("Train models", type="primary", disabled=not features or not chosen):
    cfg = TrainConfig(
        target=target, features=features, task=task, test_size=test_size, cv_folds=cv_folds,
        scale=scale, num_impute=num_impute, select_percentile=pct, time_split=time_split,
        imbalance=imbalance, positive_class=positive, parallel=parallel, max_rows=cap,
    )
    bar = st.progress(0.0, text="Preparing data...")

    def on_progress(i: int, total: int, name: str) -> None:
        text = "Finished" if name == "done" else f"Training {name} ({i + 1} of {total})"
        bar.progress(min(i / total, 1.0), text=text)

    try:
        run = train_models(df, cfg, chosen, on_progress)
    except MLError as err:
        bar.empty()
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    except Exception as exc:
        bar.empty()
        st.error(f"Training could not be completed: {exc}")
        st.info("Check the target and feature columns. Nothing was changed.")
    else:
        run.signature = (dataset.name, df.height, df.width)
        st.session_state["ml_run"] = run
        bar.empty()

# ---- Results --------------------------------------------------------------
run = st.session_state.get("ml_run")
if run is None:
    st.stop()

st.divider()
st.subheader("Results")
flash = st.session_state.pop("ml_flash", None)
if flash:
    st.success(flash)
if run.signature != (dataset.name, df.height, df.width):
    st.warning("These results were trained on an earlier version of the data. Train again to update them.")
for note in run.notes:
    st.caption(f"ℹ {note}")

best = best_result(run)
ok_results = [r for r in ranked(run) if r.ok]
tab_board, tab_eval, tab_imp, tab_tune, tab_diag, tab_save, tab_detail = st.tabs(
    ["Leaderboard", "Evaluation", "Feature importance", "Tune", "Diagnostics", "Save experiment", "Details"]
)

with tab_board:
    st.dataframe(results_table(run))
    metric = "F1" if run.task == "classification" else "RMSE"
    st.caption(f"Ranked by test {metric}. CV columns show mean ± standard deviation across folds. "
               f"Trained on {len(run.X_train):,} rows, tested on {len(run.X_test):,}."
               + (f" Positive class: {run.classes[1]}." if run.task == "classification" and len(run.classes) == 2 else "")
               + (" Multiclass metrics are weighted averages." if run.task == "classification" and len(run.classes) > 2 else ""))
    if best is not None:
        st.success(f"Best model: **{best.name}**")
        if st.button(f"Use {best.name} for the next steps"):
            st.session_state["selected_model"] = best.name
        if st.session_state.get("selected_model"):
            st.caption(f"Selected model: {st.session_state['selected_model']}")
    else:
        st.error("No model could be trained. See the Status column for the reasons.")

with tab_eval:
    if not ok_results:
        st.info("No trained model to evaluate.")
    else:
        name = st.selectbox("Model", [r.name for r in ok_results], key="ml_eval_model")
        res = next(r for r in ok_results if r.name == name)
        for note in res.notes:
            st.caption(f"ℹ {note}")
        if run.task == "classification":
            norm = st.checkbox("Show percentages per actual class", key="ml_cm_norm")
            st.plotly_chart(confusion_matrix_fig(run.y_test, res.y_pred, run.classes, norm))
            left, right = st.columns(2)
            roc = roc_fig(run.y_test, res.scores, run.classes)
            pr = pr_fig(run.y_test, res.scores, run.classes)
            if roc is not None:
                left.plotly_chart(roc)
            else:
                left.info("This model gives no usable scores for an ROC curve.")
            if pr is not None:
                right.plotly_chart(pr)
        else:
            left, right = st.columns(2)
            left.plotly_chart(actual_vs_predicted_fig(run.y_test, res.y_pred))
            right.plotly_chart(residual_fig(run.y_test, res.y_pred))

with tab_imp:
    if not ok_results:
        st.info("No trained model.")
    else:
        name = st.selectbox("Model", [r.name for r in ok_results], key="ml_imp_model")
        res = next(r for r in ok_results if r.name == name)
        st.caption("Permutation importance: how much the test score drops when one column is shuffled. "
                   "Uses at most 1,000 test rows.")
        if st.button("Compute importance"):
            try:
                with st.spinner("Shuffling columns..."):
                    run.importance[name] = permutation_table(run, res)
            except Exception as exc:
                st.error(f"Importance could not be computed: {exc}")
        table = run.importance.get(name)
        if table is not None:
            st.plotly_chart(importance_fig(table))
            st.dataframe(table)

with tab_tune:
    tunable = [r for r in ok_results if is_tunable(r.name)]
    if not tunable:
        st.info("Train at least one tunable model first (the Baseline and Linear Regression have nothing to tune).")
    elif not is_available("optuna"):
        st.warning(install_hint("optuna"))
        st.code("uv sync --extra ml", language="bash")
    else:
        tune_name = st.selectbox("Model to tune", [r.name for r in tunable], key="ml_tune_model")
        a, b, c = st.columns(3)
        n_trials = a.slider("Trials", 5, 100, 20, 5, key="ml_tune_trials")
        timeout = b.slider("Time limit (seconds)", 30, 900, 120, 30, key="ml_tune_time")
        tune_folds = c.slider("CV folds", 2, 5, 3, key="ml_tune_folds")
        st.caption("Tuning uses cross-validation on training rows only (at most 20,000), "
                   "so the test score of the tuned model stays honest.")
        if st.button("Start tuning", type="primary"):
            bar = st.progress(0.0, text="Starting...")

            def on_trial(done: int, total: int) -> None:
                bar.progress(min(done / total, 1.0), text=f"Trial {done} of {total}")

            try:
                outcome = tune_model(run, tune_name, n_trials, float(timeout), tune_folds, on_trial)
            except MLError as err:
                bar.empty()
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
            except Exception as exc:
                bar.empty()
                st.error(f"Tuning failed: {exc}")
            else:
                st.session_state.setdefault("ml_tuned", {})[(run.created, tune_name)] = outcome
                st.session_state["ml_flash"] = (
                    f"'{tune_name} (tuned)' was added to the leaderboard "
                    f"({outcome.n_trials_done} trials, stopped by: {outcome.stopped_by})."
                )
                st.rerun()
        outcome = st.session_state.get("ml_tuned", {}).get((run.created, tune_name))
        if outcome is not None:
            st.write(f"Best CV {outcome.metric}: **{outcome.best_cv:.4f}** "
                     f"({outcome.n_trials_done} trials in {outcome.seconds:.0f}s)")
            st.write("Best settings")
            st.json(outcome.best_params)
            st.plotly_chart(history_fig(outcome))
            imp_fig = param_importance_fig(outcome.importances)
            if imp_fig is not None:
                st.plotly_chart(imp_fig)
            with st.expander("All trials"):
                st.dataframe(outcome.trials)

with tab_diag:
    if not ok_results:
        st.info("No trained model.")
    else:
        name = st.selectbox("Model", [r.name for r in ok_results], key="ml_diag_model")
        res = next(r for r in ok_results if r.name == name)
        if run.task == "classification" and len(run.classes) == 2:
            cal = calibration_fig(run.y_test, res.scores)
            if cal is None:
                st.info("This model gives no probabilities, so calibration cannot be shown.")
            else:
                st.plotly_chart(cal[0])
                st.caption(f"Brier score: {cal[1]:.4f} (lower is better; 0.25 is what always predicting 50% gives). "
                           "Points near the dashed line mean the predicted probabilities can be trusted.")
        elif run.task == "classification":
            st.info("Calibration is shown for two-class problems only.")
        st.write("Learning curve")
        st.caption("Shows whether more data would help. Uses at most 10,000 training rows and 3 folds.")
        lc_key = (run.created, name)
        if st.button("Compute learning curve", key="ml_lc_btn"):
            try:
                with st.spinner("Training on growing subsets..."):
                    st.session_state.setdefault("ml_lc", {})[lc_key] = learning_curve_fig(run, res)
            except MLError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
            except Exception as exc:
                st.error(f"The learning curve could not be computed: {exc}")
        lc_fig = st.session_state.get("ml_lc", {}).get(lc_key)
        if lc_fig is not None:
            st.plotly_chart(lc_fig)
            st.caption("A large gap between the lines suggests overfitting. Both lines low and close together "
                       "suggests the model is too simple or the features are weak.")

with tab_save:
    if not ok_results:
        st.info("No trained model.")
    else:
        names = [r.name for r in ok_results if r.name != "Baseline"] or [r.name for r in ok_results]
        default = names.index(best.name) if best is not None and best.name in names else 0
        save_name = st.selectbox("Model to save", names, index=default, key="ml_save_model")
        note = st.text_input("Note (optional)", key="ml_save_note")
        st.caption("Saves the settings, features, preprocessing steps, hyperparameters and metrics. "
                   "The model file itself is saved in a later phase.")
        if st.button("Save experiment", type="primary"):
            res = next(r for r in ok_results if r.name == save_name)
            history = st.session_state.get("history")
            steps = [s.label for s in history.snapshots[1:history.index + 1]] if history is not None else []
            info = {"name": dataset.name, "rows": df.height, "columns": df.width,
                    "file_type": dataset.file_type, "rows_in_file": dataset.total_rows,
                    "sampled": dataset.sampled}
            try:
                exp_id = save_experiment(run, res, info, steps, note)
            except Exception as exc:
                st.error(f"The experiment could not be saved: {exc}")
            else:
                st.success(f"Saved experiment {exp_id}.")
                st.page_link("views/experiments.py", label="Open Experiments")

with tab_detail:
    st.write("Settings used")
    st.json(asdict(run.config))
    st.write(f"Numeric features: {', '.join(run.num) or 'none'}")
    st.write(f"Categorical features: {', '.join(run.cat) or 'none'}")
    st.caption(f"Trained at {run.created}. Imputing, scaling, encoding, feature selection and resampling "
               "were all fitted on training rows only.")