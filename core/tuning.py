"""Hyperparameter tuning with Optuna (optional package, UI-free).

Tuning uses cross-validation on TRAINING rows only. The test set is touched
once, after tuning, so the tuned model's test score is honest.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score

from core.model_training import (
    SLOW_MAX_ROWS,
    MLError,
    ModelResult,
    TrainingRun,
    _make_sampler,
    _scores,
    build_pipeline,
    build_preprocessor,
    classification_metrics,
    model_specs,
    regression_metrics,
)
from utils.optional_imports import MissingDependencyError, lazy_import

TUNE_MAX_ROWS = 20_000
TUNABLE = {
    "Logistic Regression", "Ridge", "Lasso", "KNN", "Decision Tree", "Random Forest",
    "SVM", "SVR", "Gradient Boosting", "XGBoost", "LightGBM",
}


@dataclass
class TuneOutcome:
    model_name: str
    best_params: dict[str, Any]
    best_cv: float
    metric: str
    higher_is_better: bool
    trials: pd.DataFrame
    importances: dict[str, float]
    result: ModelResult
    seconds: float
    n_trials_done: int
    stopped_by: str


def is_tunable(name: str) -> bool:
    return name in TUNABLE


def _space(name: str, t: Any) -> dict[str, Any]:
    """Search space for one model. `t` is an Optuna trial."""
    if name == "Logistic Regression":
        return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "Ridge":
        return {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if name == "Lasso":
        return {"alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True)}
    if name == "KNN":
        return {
            "n_neighbors": t.suggest_int("n_neighbors", 3, 30),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if name == "Decision Tree":
        return {
            "max_depth": t.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50),
        }
    if name == "Random Forest":
        return {
            "n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": t.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        }
    if name in ("SVM", "SVR"):
        params = {
            "C": t.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": t.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
        if name == "SVR":
            params["epsilon"] = t.suggest_float("epsilon", 1e-3, 1.0, log=True)
        return params
    if name == "Gradient Boosting":
        return {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 64),
            "max_depth": t.suggest_int("max_depth", 2, 10),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }
    if name == "XGBoost":
        return {
            "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 10),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if name == "LightGBM":
        return {
            "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 128),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 60),
        }
    raise MLError(f"'{name}' has no tunable settings.", "Choose another model.")


def training_sample(run: TrainingRun, max_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Row-limited copy of the training data (evenly spaced if time-ordered)."""
    X, y = run.X_train, run.y_train
    n = len(X)
    if n <= max_rows:
        return X, y
    if run.config.time_split:
        idx = np.linspace(0, n - 1, max_rows).astype(int)
    else:
        idx = np.sort(np.random.default_rng(run.config.seed).choice(n, max_rows, replace=False))
    return X.iloc[idx].reset_index(drop=True), y[idx]


def make_cv(run: TrainingRun, y: np.ndarray, folds: int) -> Any:
    folds = max(2, folds)
    cfg = run.config
    if cfg.time_split:
        return TimeSeriesSplit(n_splits=folds)
    if run.task == "classification":
        counts = np.bincount(y)
        counts = counts[counts > 0]
        n = min(folds, int(counts.min()))
        if n < 2:
            raise MLError("A class has too few training rows for cross-validation.",
                          "Remove or merge rare classes first.")
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=cfg.seed)
    return KFold(n_splits=folds, shuffle=True, random_state=cfg.seed)


def tune_model(
    run: TrainingRun,
    model_name: str,
    n_trials: int = 20,
    timeout: float = 120.0,
    folds: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> TuneOutcome:
    """Tune one model, refit it on all training rows, add it to run.results.

    Raises:
        MLError: for setup problems or when no trial could finish.
    """
    if model_name not in TUNABLE:
        raise MLError(f"'{model_name}' has no tunable settings.", "Choose another model.")
    try:
        optuna = lazy_import("optuna")
    except MissingDependencyError as exc:
        raise MLError("Tuning needs the 'optuna' package.", str(exc)) from exc
    if n_trials < 1:
        raise MLError("Choose at least one trial.")

    cfg = run.config
    clf = run.task == "classification"
    spec = next((s for s in model_specs(run.task) if s.name == model_name), None)
    if spec is None or not spec.available:
        raise MLError(f"'{model_name}' is not available.", "Install it with: uv sync --extra ml")
    if spec.slow and len(run.X_train) > SLOW_MAX_ROWS:
        raise MLError(f"{model_name} is too slow to tune on {len(run.X_train):,} training rows.",
                      "Use Fast Preview or tune a faster model.")

    X, y = training_sample(run, TUNE_MAX_ROWS)
    cv = make_cv(run, y, folds)
    balanced = clf and cfg.imbalance == "class_weight" and spec.class_weight
    resample = clf and cfg.imbalance in ("oversample", "undersample", "smote")
    scoring = ("f1" if len(run.classes) == 2 else "f1_weighted") if clf else "neg_root_mean_squared_error"
    metric, sign = ("F1", 1) if clf else ("RMSE", -1)
    errors: list[str] = []

    def objective(trial: Any) -> float:
        try:
            estimator = spec.factory(cfg.seed, balanced)
            estimator.set_params(**_space(model_name, trial))
            sampler = _make_sampler(cfg.imbalance, run.y_train, cfg.seed) if resample else None
            pipe = build_pipeline(estimator, build_preprocessor(run.num, run.cat, cfg), cfg, sampler)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring,
                                         n_jobs=-1 if cfg.parallel else 1, error_score="raise")
            trial.set_user_attr("std", float(np.std(scores)))
            return float(np.mean(scores))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            raise optuna.TrialPruned() from exc

    def on_trial(study: Any, trial: Any) -> None:
        if progress:
            progress(len(study.trials), n_trials)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[on_trial])

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise MLError("No tuning trial could finish.", errors[-1] if errors else "Try a simpler model.")
    best = study.best_trial

    rows = []
    for t in study.trials:
        row: dict[str, Any] = {
            "trial": t.number,
            "state": t.state.name.lower(),
            metric: None if t.value is None else sign * t.value,
            "cv_std": t.user_attrs.get("std"),
        }
        row.update(t.params)
        rows.append(row)
    trials = pd.DataFrame(rows)

    importances: dict[str, float] = {}
    if len(done) >= 3:
        try:
            importances = dict(optuna.importance.get_param_importances(study))
        except Exception:
            importances = {}

    # Refit with the best settings on ALL training rows, then test once.
    estimator = spec.factory(cfg.seed, balanced)
    estimator.set_params(**best.params)
    sampler = _make_sampler(cfg.imbalance, run.y_train, cfg.seed) if resample else None
    pipe = build_pipeline(estimator, build_preprocessor(run.num, run.cat, cfg), cfg, sampler)
    result = ModelResult(name=f"{model_name} (tuned)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        pipe.fit(run.X_train, run.y_train)
        result.fit_seconds = time.perf_counter() - start
        result.y_pred = np.asarray(pipe.predict(run.X_test))
        if clf:
            result.scores = _scores(pipe, run.X_test)
    result.pipeline = pipe
    result.test = (classification_metrics(run.y_test, result.y_pred, result.scores, len(run.classes))
                   if clf else regression_metrics(run.y_test, result.y_pred))
    result.cv[metric] = (sign * best.value, float(best.user_attrs.get("std", 0.0)))
    result.notes.append(f"Tuned with Optuna: {len(done)} finished trials, {folds}-fold CV on training rows only.")

    run.results = [r for r in run.results if r.name != result.name] + [result]
    return TuneOutcome(
        model_name=model_name, best_params=dict(best.params), best_cv=sign * best.value,
        metric=metric, higher_is_better=clf, trials=trials, importances=importances, result=result,
        seconds=time.perf_counter() - started, n_trials_done=len(study.trials),
        stopped_by="all trials done" if len(study.trials) >= n_trials else "time limit",
    )


def history_fig(outcome: TuneOutcome) -> go.Figure:
    t = outcome.trials[outcome.trials["state"] == "complete"]
    values = t[outcome.metric].to_numpy(dtype=float)
    best_so_far = np.maximum.accumulate(values) if outcome.higher_is_better else np.minimum.accumulate(values)
    fig = go.Figure()
    fig.add_scatter(x=t["trial"], y=values, mode="markers", name="Trial")
    fig.add_scatter(x=t["trial"], y=best_so_far, mode="lines", name="Best so far")
    fig.update_layout(template="plotly_white", xaxis_title="Trial",
                      yaxis_title=f"CV {outcome.metric}" + ("" if outcome.higher_is_better else " (lower is better)"))
    return fig


def param_importance_fig(importances: dict[str, float]) -> go.Figure | None:
    if not importances:
        return None
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h"))
    fig.update_layout(template="plotly_white", xaxis_title="Importance of the setting", yaxis_title="",
                      height=max(260, 40 * len(items) + 120))
    return fig