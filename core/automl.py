"""AutoML: native scikit-learn comparison plus optional FLAML (UI-free)."""

from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np

from core.model_training import (
    MLError,
    ModelResult,
    TrainConfig,
    TrainingRun,
    _scores,
    classification_metrics,
    model_specs,
    ranked,
    regression_metrics,
    train_models,
)
from core.tuning import training_sample
from utils.optional_imports import MissingDependencyError, is_available, lazy_import

import polars as pl

FLAML_MAX_ROWS = 50_000


def native_automl(
    df: pl.DataFrame, cfg: TrainConfig, progress: Callable[[int, int, str], None] | None = None
) -> TrainingRun:
    """Train a Baseline plus every available model. Slow models skip themselves on big data."""
    names = [s.name for s in model_specs(cfg.task) if s.available]
    if not names:
        raise MLError("No model is available.", "Check the installation.")
    return train_models(df, cfg, names, progress)


def flaml_available() -> bool:
    return is_available("flaml")


def run_flaml(run: TrainingRun, time_budget: int = 60) -> ModelResult:
    """Search with FLAML on the SAME split as `run`, then test once. Adds the result to run.results.

    Raises:
        MLError: if FLAML is missing, fails, or finds nothing in the time budget.
    """
    try:
        flaml = lazy_import("flaml")
    except MissingDependencyError as exc:
        raise MLError("FLAML is not installed.", str(exc)) from exc

    cfg = run.config
    clf = run.task == "classification"
    estimators = ["rf", "extra_tree"]
    if is_available("lightgbm"):
        estimators.insert(0, "lgbm")
    if is_available("xgboost"):
        estimators.insert(0, "xgboost")

    X, y = training_sample(run, FLAML_MAX_ROWS)
    metric = ("f1" if len(run.classes) == 2 else "macro_f1") if clf else "rmse"
    kwargs: dict = dict(
        task="classification" if clf else "regression",
        metric=metric,
        time_budget=int(time_budget),
        estimator_list=estimators,
        seed=cfg.seed,
        verbose=0,
        n_jobs=-1 if cfg.parallel else 1,
    )
    if cfg.time_split:
        kwargs.update(eval_method="holdout", split_type="time")
    else:
        kwargs.update(eval_method="cv", n_splits=3)

    automl = flaml.AutoML()
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            automl.fit(X_train=X, y_train=y, **kwargs)
    except Exception as exc:
        raise MLError("FLAML could not finish.", f"Details: {type(exc).__name__}: {exc}") from exc
    seconds = time.perf_counter() - started

    best_name = getattr(automl, "best_estimator", None)
    if not best_name:
        raise MLError("FLAML found no model within the time budget.", "Increase the time budget.")

    res = ModelResult(name=f"FLAML ({best_name})", fit_seconds=seconds)
    try:
        pred = np.asarray(automl.predict(run.X_test))
        res.y_pred = pred.astype(int) if clf else pred.astype(float)
        if clf:
            res.scores = _scores(automl, run.X_test)
        res.test = (classification_metrics(run.y_test, res.y_pred, res.scores, len(run.classes))
                    if clf else regression_metrics(run.y_test, res.y_pred))
    except Exception as exc:
        raise MLError("The FLAML model could not be tested.", f"Details: {exc}") from exc
    res.pipeline = automl
    loss = getattr(automl, "best_loss", None)
    if loss is not None:
        value = 1 - float(loss) if clf else float(loss)
        res.notes.append(f"FLAML's own validation {metric}: {value:.4f}. Best settings: {getattr(automl, 'best_config', {})}")
    res.notes.append("FLAML does its own preprocessing. It supports the Evaluation tab but not tuning or learning curves.")

    run.results = [r for r in run.results if not r.name.startswith("FLAML (")] + [res]
    return res


def leaderboard_notes(run: TrainingRun) -> list[str]:
    """Plain-language findings computed from the actual results."""
    clf = run.task == "classification"
    key = "F1" if clf else "RMSE"
    ok = [r for r in ranked(run) if r.ok and r.test.get(key) is not None]
    real = [r for r in ok if r.name != "Baseline"]
    base = next((r for r in ok if r.name == "Baseline"), None)
    if not real:
        return ["No model finished successfully. See the Status column for the reasons."]

    best = real[0]
    value = float(best.test[key])
    notes = [f"{best.name} is the best model on test {key} ({value:.3f})."]
    if base is not None:
        b = float(base.test[key])
        if clf:
            notes.append(f"The Baseline (always guessing the most common class) scores {b:.3f}; "
                         f"{best.name} improves F1 by {value - b:+.3f}.")
        elif b > 0:
            notes.append(f"The Baseline (always predicting the mean) has RMSE {b:.3f}; "
                         f"{best.name} reduces it by {(b - value) / b * 100:.0f}%.")
    close = ([r for r in real if value - float(r.test[key]) <= 0.01] if clf
             else [r for r in real if float(r.test[key]) <= value * 1.01])
    fastest = min(close, key=lambda r: r.fit_seconds)
    if fastest is not best:
        notes.append(f"{fastest.name} is within 1% of the best and trains in {fastest.fit_seconds:.1f}s "
                     f"instead of {best.fit_seconds:.1f}s.")
    cv = best.cv.get(key)
    if cv is not None and abs(cv[0] - value) > max(0.05, 2 * cv[1]):
        notes.append(f"The cross-validation {key} ({cv[0]:.3f}) differs noticeably from the test {key} "
                     f"({value:.3f}). The test set may be small or unrepresentative.")
    return notes