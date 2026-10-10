"""Calibration and learning-curve diagnostics (UI-free)."""

from __future__ import annotations

import warnings

import numpy as np
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import learning_curve

from core.model_training import MLError, ModelResult, TrainingRun
from core.tuning import make_cv, training_sample


def calibration_fig(y_true: np.ndarray, scores: np.ndarray | None, n_bins: int = 10) -> tuple[go.Figure, float] | None:
    """Reliability diagram for a two-class model, plus the Brier score.

    Returns None when the model gives no probabilities.
    """
    if scores is None or scores.ndim != 1:
        return None
    if float(scores.min()) < 0 or float(scores.max()) > 1:
        return None
    try:
        frac, mean_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="quantile")
        brier = float(brier_score_loss(y_true, scores))
    except ValueError:
        return None
    fig = go.Figure()
    fig.add_scatter(x=mean_pred, y=frac, mode="lines+markers", name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                    line=dict(dash="dash", color="grey"))
    fig.update_layout(template="plotly_white", xaxis_title="Predicted probability",
                      yaxis_title="Observed share of the positive class")
    return fig, brier


def learning_curve_fig(run: TrainingRun, result: ModelResult, n_sizes: int = 5,
                       folds: int = 3, max_rows: int = 10_000) -> go.Figure:
    """Score against training-set size, with a band of one standard deviation."""
    if result.pipeline is None:
        raise MLError("This model has no fitted pipeline.")
    cfg = run.config
    clf = run.task == "classification"
    X, y = training_sample(run, max_rows)
    if len(X) < 40:
        raise MLError("Too few training rows for a learning curve.", "At least 40 are needed.")
    cv = make_cv(run, y, folds)
    scoring = ("f1" if len(run.classes) == 2 else "f1_weighted") if clf else "neg_root_mean_squared_error"
    sign = 1 if clf else -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train_sc, val_sc = learning_curve(
            clone(result.pipeline), X, y, train_sizes=np.linspace(0.2, 1.0, n_sizes), cv=cv,
            scoring=scoring, n_jobs=-1 if cfg.parallel else 1,
            shuffle=not cfg.time_split, random_state=cfg.seed,
        )
    fig = go.Figure()
    for label, arr, color, fill in (
        ("Training", sign * train_sc, "#1F77B4", "rgba(31,119,180,0.15)"),
        ("Cross-validation", sign * val_sc, "#D62728", "rgba(214,39,40,0.15)"),
    ):
        mean, std = arr.mean(axis=1), arr.std(axis=1)
        fig.add_scatter(x=np.concatenate([sizes, sizes[::-1]]),
                        y=np.concatenate([mean + std, (mean - std)[::-1]]),
                        fill="toself", fillcolor=fill, line=dict(width=0),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=sizes, y=mean, mode="lines+markers", name=label, line=dict(color=color))
    fig.update_layout(template="plotly_white", xaxis_title="Training rows",
                      yaxis_title="F1" if clf else "RMSE (lower is better)")
    return fig