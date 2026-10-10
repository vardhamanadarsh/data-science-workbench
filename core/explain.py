"""Model explainability with SHAP (optional package) and Plotly figures (UI-free)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.model_training import MLError, ModelResult, TrainingRun
from core.whatif import fmt_value
from utils.optional_imports import MissingDependencyError, lazy_import

TREE_MODELS = {"Random Forest", "Decision Tree", "XGBoost", "LightGBM"}
LINEAR_MODELS = {"Logistic Regression", "Linear Regression", "Ridge", "Lasso"}
PROBABILITY_TREES = {"Random Forest", "Decision Tree"}
MAX_ROWS = {"tree": 1000, "linear": 1000, "generic": 100}
BACKGROUND_ROWS = 100
TEMPLATE = "plotly_white"


def base_name(name: str) -> str:
    return name.removesuffix(" (tuned)")


def explainer_kind(name: str) -> str:
    b = base_name(name)
    return "tree" if b in TREE_MODELS else "linear" if b in LINEAR_MODELS else "generic"


def explainable(res: ModelResult) -> bool:
    """True for trained scikit-learn pipelines (not the Baseline, not FLAML)."""
    pipe = res.pipeline
    return (res.ok and res.name != "Baseline" and pipe is not None
            and hasattr(pipe, "named_steps") and "model" in pipe.named_steps)


@dataclass
class ShapResult:
    model_name: str
    kind: str
    unit: str
    class_name: str | None
    names: list[str]              # encoded feature names (after one-hot)
    values: np.ndarray            # (rows, encoded features)
    base: float                   # average prediction (in `unit`)
    Xt: np.ndarray                # transformed values, for colouring
    X_raw: pd.DataFrame           # the explained rows, original columns
    y_true: np.ndarray | None


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def transform(pipe: Any, X: pd.DataFrame) -> np.ndarray:
    """Apply every pipeline step except the sampler and the model."""
    out: Any = X
    for name, step in pipe.steps:
        if name in ("model", "sampler"):
            continue
        out = step.transform(out)
    return np.asarray(out, dtype=float)


def feature_names(pipe: Any) -> list[str]:
    names = [str(n).split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    select = pipe.named_steps.get("select")
    if select is not None:
        names = [n for n, keep in zip(names, select.get_support()) if keep]
    return names


def group_of(name: str, run: TrainingRun) -> str:
    """Original column an encoded feature came from."""
    if name in run.num:
        return name
    for col in sorted(run.cat, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


def _pick(values: Any, ci: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[ci]
    v = np.asarray(values, dtype=float)
    return v[:, :, ci] if v.ndim == 3 else v


def _base(value: Any, ci: int, generic: bool) -> float:
    arr = np.ravel(np.asarray(value, dtype=float))
    if generic or arr.size == 1:
        return float(arr.mean())
    return float(arr[min(ci, arr.size - 1)])


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #

def explain_frame(run: TrainingRun, res: ModelResult, X_raw: pd.DataFrame,
                  class_index: int = 1, y_true: np.ndarray | None = None) -> ShapResult:
    """SHAP values for the given rows (original columns)."""
    try:
        shap = lazy_import("shap")
    except MissingDependencyError as exc:
        raise MLError("SHAP is not installed.", str(exc)) from exc
    if not explainable(res):
        raise MLError("This model cannot be explained here.",
                      "Choose a model trained in Machine Learning or AutoML (FLAML models are not supported).")

    pipe = res.pipeline
    model = pipe.named_steps["model"]
    clf = run.task == "classification"
    k = len(run.classes)
    kind = explainer_kind(res.name)
    if kind == "generic" and clf and k > 2:
        raise MLError("This model type can only be explained for two-class problems.",
                      "Use Random Forest, XGBoost, LightGBM or Logistic Regression for several classes.")
    ci = 1 if (clf and k == 2) else (min(max(class_index, 0), k - 1) if clf else 0)

    X_raw = X_raw.reset_index(drop=True)
    Xt = transform(pipe, X_raw)
    names = feature_names(pipe)
    if Xt.shape[1] != len(names):
        raise MLError("Feature names could not be matched to the model input.")

    rng = np.random.default_rng(run.config.seed)
    n_bg = min(BACKGROUND_ROWS, len(run.X_train))
    bg = transform(pipe, run.X_train.iloc[np.sort(rng.choice(len(run.X_train), n_bg, replace=False))])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "tree":
                ex = shap.TreeExplainer(model)
                raw, base_raw = ex.shap_values(Xt, check_additivity=False), ex.expected_value
            elif kind == "linear":
                ex = shap.LinearExplainer(model, bg)
                raw, base_raw = ex.shap_values(Xt), ex.expected_value
            else:
                fn = (lambda a: model.predict_proba(a)[:, 1]) if clf else model.predict
                ex = shap.Explainer(fn, shap.maskers.Independent(bg, max_samples=n_bg))
                e = ex(Xt)
                raw, base_raw = e.values, e.base_values
    except Exception as exc:
        raise MLError("SHAP could not explain this model.", f"Details: {type(exc).__name__}: {exc}") from exc

    values = _pick(raw, ci)
    if values.shape != Xt.shape:
        raise MLError("SHAP returned an unexpected shape.", "Try another model.")
    unit = ("target units" if not clf
            else "probability" if base_name(res.name) in PROBABILITY_TREES or kind == "generic"
            else "log-odds")
    return ShapResult(res.name, kind, unit, run.classes[ci] if clf else None, names, values,
                      _base(base_raw, ci, kind == "generic"), Xt, X_raw, y_true)


def compute_shap(run: TrainingRun, res: ModelResult, max_rows: int = 300,
                 class_index: int = 1) -> ShapResult:
    """SHAP values for a random sample of the test rows."""
    cap = min(max_rows, MAX_ROWS[explainer_kind(res.name)])
    X, y = run.X_test, run.y_test
    if len(X) > cap:
        idx = np.sort(np.random.default_rng(run.config.seed).choice(len(X), cap, replace=False))
        X, y = X.iloc[idx], y[idx]
    return explain_frame(run, res, X, class_index, np.asarray(y))


# --------------------------------------------------------------------------- #
# Tables and figures
# --------------------------------------------------------------------------- #

def grouped_values(sr: ShapResult, run: TrainingRun) -> pd.DataFrame:
    """SHAP values summed per ORIGINAL column (one-hot columns are added together)."""
    groups = [group_of(n, run) for n in sr.names]
    return pd.DataFrame(sr.values, columns=sr.names).T.groupby(groups, sort=False).sum().T


def importance_table(sr: ShapResult, run: TrainingRun) -> pd.DataFrame:
    imp = grouped_values(sr, run).abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values})


def importance_fig(sr: ShapResult, run: TrainingRun, top: int = 15) -> go.Figure:
    t = importance_table(sr, run).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["mean_abs_shap"], y=t["feature"], orientation="h"))
    fig.update_layout(template=TEMPLATE, xaxis_title=f"Mean |SHAP value| ({sr.unit})", yaxis_title="",
                      height=max(300, 26 * len(t) + 120))
    return fig


def summary_fig(sr: ShapResult, top: int = 15, seed: int = 42) -> go.Figure:
    """Beeswarm: each dot is a row; colour is the (scaled) feature value."""
    order = np.argsort(np.abs(sr.values).mean(axis=0))[::-1][:top]
    rng = np.random.default_rng(seed)
    xs, ys, cs = [], [], []
    for pos, j in enumerate(order[::-1]):
        col = sr.Xt[:, j]
        lo, hi = np.percentile(col, [5, 95])
        if hi <= lo:
            lo, hi = float(col.min()), float(col.max())
        norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(col), 0.5)
        xs.append(sr.values[:, j])
        ys.append(pos + rng.uniform(-0.25, 0.25, len(col)))
        cs.append(norm)
    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers", hoverinfo="x",
        marker=dict(size=5, opacity=0.7, color=np.concatenate(cs),
                    colorscale=[[0, "#1F77B4"], [1, "#D62728"]], showscale=True,
                    colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"]))))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[sr.names[j] for j in order[::-1]])
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(template=TEMPLATE, xaxis_title=f"SHAP value ({sr.unit})",
                      height=max(350, 28 * len(order) + 140))
    return fig


def dependence_fig(sr: ShapResult, run: TrainingRun, feature: str, color_by: str | None = None) -> go.Figure:
    """SHAP contribution of one original column against its raw value."""
    g = grouped_values(sr, run)
    if feature not in g.columns:
        raise MLError(f"'{feature}' was not found in the explained features.")
    df = pd.DataFrame({"value": sr.X_raw[feature].to_numpy(), "shap": g[feature].to_numpy()})
    labels = {"value": feature, "shap": f"SHAP value ({sr.unit})"}
    if feature in run.num:
        if color_by and color_by in run.num and color_by != feature:
            df["color"] = sr.X_raw[color_by].to_numpy()
            fig = px.scatter(df, x="value", y="shap", color="color", color_continuous_scale="RdBu_r",
                             labels={**labels, "color": color_by})
        else:
            fig = px.scatter(df, x="value", y="shap", labels=labels)
    else:
        df["value"] = df["value"].astype(str)
        fig = px.strip(df, x="value", y="shap", labels=labels)
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(template=TEMPLATE)
    return fig


def local_contributions(sr: ShapResult, run: TrainingRun, i: int) -> pd.DataFrame:
    """Contribution of each original column to one prediction."""
    g = grouped_values(sr, run).iloc[i]
    raw = sr.X_raw.iloc[i]
    return pd.DataFrame({"feature": list(g.index),
                         "value": [fmt_value(raw[c]) if c in raw.index else "" for c in g.index],
                         "shap": g.to_numpy()})


def waterfall_fig(contrib: pd.DataFrame, base: float, unit: str, top: int = 10) -> go.Figure:
    c = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
    head, rest = c.head(top), c.iloc[top:]["shap"].sum()
    labels = ["Average prediction"] + [f"{r.feature} = {r.value}" for r in head.itertuples()]
    values = [base] + head["shap"].tolist()
    measures = ["absolute"] + ["relative"] * len(head)
    if len(c) > top:
        labels.append("Other features")
        values.append(float(rest))
        measures.append("relative")
    labels.append("This prediction")
    values.append(base + float(c["shap"].sum()))
    measures.append("total")
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measures,
        increasing=dict(marker=dict(color="#D62728")), decreasing=dict(marker=dict(color="#1F77B4")),
        totals=dict(marker=dict(color="#555555"))))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template=TEMPLATE, xaxis_title=unit.capitalize(), showlegend=False,
                      height=max(320, 32 * len(labels) + 100))
    return fig