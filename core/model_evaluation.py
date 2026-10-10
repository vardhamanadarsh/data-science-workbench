"""Evaluation plots and permutation importance (UI-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core.model_training import ModelResult, TrainingRun

TEMPLATE = "plotly_white"
MAX_POINTS = 5_000


def _sample(n: int, seed: int = 42) -> np.ndarray:
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_POINTS, replace=False))


def confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                         normalize: bool = False) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))),
                          normalize="true" if normalize else None)
    fig = px.imshow(cm, x=classes, y=classes, text_auto=".2f" if normalize else True,
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(template=TEMPLATE, xaxis_title="Predicted", yaxis_title="Actual",
                      coloraxis_showscale=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def roc_fig(y_true: np.ndarray, scores: np.ndarray | None, classes: list[str]) -> go.Figure | None:
    if scores is None:
        return None
    fig = go.Figure()
    try:
        if scores.ndim == 1:
            fpr, tpr, _ = roc_curve(y_true, scores)
            fig.add_scatter(x=fpr, y=tpr, mode="lines",
                            name=f"{classes[1]} (AUC {roc_auc_score(y_true, scores):.3f})")
        else:
            for k, name in enumerate(classes):
                yb = (y_true == k).astype(int)
                if yb.sum() in (0, len(yb)):
                    continue
                fpr, tpr, _ = roc_curve(yb, scores[:, k])
                fig.add_scatter(x=fpr, y=tpr, mode="lines",
                                name=f"{name} (AUC {roc_auc_score(yb, scores[:, k]):.3f})")
    except ValueError:
        return None
    if not fig.data:
        return None
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey"))
    fig.update_layout(template=TEMPLATE, xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def pr_fig(y_true: np.ndarray, scores: np.ndarray | None, classes: list[str]) -> go.Figure | None:
    if scores is None:
        return None
    fig = go.Figure()
    try:
        if scores.ndim == 1:
            prec, rec, _ = precision_recall_curve(y_true, scores)
            fig.add_scatter(x=rec, y=prec, mode="lines",
                            name=f"{classes[1]} (AP {average_precision_score(y_true, scores):.3f})")
            fig.add_hline(y=float(np.mean(y_true)), line_dash="dash", line_color="grey",
                          annotation_text="No-skill level")
        else:
            for k, name in enumerate(classes):
                yb = (y_true == k).astype(int)
                if yb.sum() in (0, len(yb)):
                    continue
                prec, rec, _ = precision_recall_curve(yb, scores[:, k])
                fig.add_scatter(x=rec, y=prec, mode="lines",
                                name=f"{name} (AP {average_precision_score(yb, scores[:, k]):.3f})")
    except ValueError:
        return None
    if not fig.data:
        return None
    fig.update_layout(template=TEMPLATE, xaxis_title="Recall", yaxis_title="Precision")
    return fig


def actual_vs_predicted_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    idx = _sample(len(y_true))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=y_true[idx], y=y_pred[idx], mode="markers", name="Test rows",
                               marker=dict(size=5, opacity=0.6)))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                    line=dict(dash="dash", color="grey"))
    fig.update_layout(template=TEMPLATE, xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def residual_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    idx = _sample(len(y_true))
    fig = go.Figure(go.Scattergl(x=y_pred[idx], y=(y_true - y_pred)[idx], mode="markers",
                                 name="Residual", marker=dict(size=5, opacity=0.6)))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(template=TEMPLATE, xaxis_title="Predicted", yaxis_title="Residual (actual - predicted)")
    return fig


def permutation_table(run: TrainingRun, result: ModelResult, max_rows: int = 1_000,
                      repeats: int = 5) -> pd.DataFrame:
    """How much the test score drops when one original column is shuffled."""
    X, y = run.X_test, run.y_test
    if len(X) > max_rows:
        idx = np.sort(np.random.default_rng(run.config.seed).choice(len(X), max_rows, replace=False))
        X, y = X.iloc[idx], y[idx]
    if run.task == "regression":
        scoring = "r2"
    else:
        scoring = "f1" if len(run.classes) == 2 else "f1_weighted"
    out = permutation_importance(result.pipeline, X, y, scoring=scoring, n_repeats=repeats,
                                 random_state=run.config.seed, n_jobs=1)
    return (pd.DataFrame({"feature": list(X.columns), "importance": out.importances_mean,
                          "std": out.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def importance_fig(table: pd.DataFrame, top: int = 20) -> go.Figure:
    t = table.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                           error_x=dict(type="data", array=t["std"])))
    fig.update_layout(template=TEMPLATE, xaxis_title="Drop in test score when shuffled",
                      yaxis_title="", height=max(300, 24 * len(t) + 120))
    return fig