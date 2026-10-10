"""What-if predictions and counterfactual search (UI-free, no extra packages).

A counterfactual here is a transparent search: change one or two features to
values seen in the training data and check whether the model's predicted class
flips. Results are suggestions about the MODEL, not causal advice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.model_training import MLError, ModelResult, TrainingRun

CF_GRID = 15
PAIR_GRID = 6
PAIR_FEATURES = 6


def fmt_value(v: Any) -> str:
    """Readable value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return str(v)


@dataclass
class Domain:
    """The range of values a feature takes in the training data."""

    kind: str                                   # "num" | "cat"
    low: float = 0.0
    high: float = 0.0
    median: float = 0.0
    std: float = 1.0
    is_int: bool = False
    categories: list[str] = field(default_factory=list)   # most frequent first
    grid: list[float] = field(default_factory=list)       # candidate values


@dataclass
class Counterfactual:
    changes: list[tuple[str, Any, Any]]         # (feature, old value, new value)
    prob: float                                 # probability of the desired class
    distance: float                             # size of the change (smaller = easier)

    def describe(self) -> str:
        return "; ".join(f"{f}: {fmt_value(o)} → {fmt_value(n)}" for f, o, n in self.changes)


@dataclass
class CounterfactualResult:
    current_class: int
    current_prob: float                         # current probability of the desired class
    items: list[Counterfactual]
    n_tested: int


def feature_domains(run: TrainingRun) -> dict[str, Domain]:
    out: dict[str, Domain] = {}
    X = run.X_train
    for c in run.num:
        s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        vals = s.to_numpy(dtype=float)
        is_int = bool(np.all(np.isclose(vals, np.round(vals))))
        q = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 25)))
        if is_int:
            q = np.unique(np.round(q))
        sd = float(vals.std())
        out[c] = Domain("num", float(vals.min()), float(vals.max()), float(np.median(vals)),
                        sd if sd > 0 else 1.0, is_int, [], [float(v) for v in q])
    for c in run.cat:
        s = X[c].dropna().astype(str)
        if s.empty:
            continue
        out[c] = Domain("cat", categories=s.value_counts().index.tolist())
    return out


def typical_row(run: TrainingRun, domains: dict[str, Domain]) -> pd.DataFrame:
    """Median of every numeric feature and the most common value of every category."""
    row: dict[str, list[Any]] = {}
    for c in run.X_train.columns:
        d = domains.get(c)
        row[c] = [np.nan] if d is None else [d.median] if d.kind == "num" else [d.categories[0]]
    return pd.DataFrame(row)


def predict_frame(run: TrainingRun, res: ModelResult, X: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Predictions for a frame. Classification returns class probabilities.

    The flag is False when the model gives labels only (probabilities are then 0 or 1).
    """
    pipe = res.pipeline
    if pipe is None:
        raise MLError("This model has no fitted pipeline.")
    if run.task != "classification":
        return np.asarray(pipe.predict(X), dtype=float), True
    if hasattr(pipe, "predict_proba"):
        try:
            return np.asarray(pipe.predict_proba(X), dtype=float), True
        except Exception:
            pass
    labels = np.asarray(pipe.predict(X)).astype(int)
    return np.eye(len(run.classes))[labels], False


def prediction_summary(run: TrainingRun, res: ModelResult, row: pd.DataFrame) -> str:
    p, has_proba = predict_frame(run, res, row)
    if run.task != "classification":
        return f"Prediction: {float(p[0]):,.3f}"
    pr = p[0]
    if len(run.classes) == 2 and has_proba:
        return f"Prediction: {pr[1]:.0%} probability of '{run.classes[1]}'"
    top = int(np.argmax(pr))
    suffix = f" ({pr[top]:.0%})" if has_proba else ""
    return f"Prediction: '{run.classes[top]}'{suffix}"


def apply_changes(row: pd.DataFrame, changes: list[tuple[str, Any, Any]]) -> pd.DataFrame:
    out = row.copy()
    for feature, _, new in changes:
        out[feature] = [new]
    return out


def _make_batch(row: pd.DataFrame, changes_list: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.concat([row] * len(changes_list), ignore_index=True)
    for c in {c for ch in changes_list for c in ch}:
        base = row.iloc[0][c]
        out[c] = [ch.get(c, base) for ch in changes_list]
    return out


def _candidates(dom: Domain, current: Any, n: int) -> list[Any]:
    if dom.kind == "cat":
        return [c for c in dom.categories if c != str(current)][:n]
    grid = dom.grid
    if len(grid) > n:
        grid = [grid[i] for i in np.unique(np.linspace(0, len(grid) - 1, n).astype(int))]
    cur = None if pd.isna(current) else float(current)
    return [v for v in grid if cur is None or not np.isclose(v, cur)]


def _distance(dom: Domain, old: Any, new: Any) -> float:
    if dom.kind == "cat" or pd.isna(old):
        return 1.0
    return abs(float(new) - float(old)) / dom.std


def find_counterfactuals(
    run: TrainingRun, res: ModelResult, row: pd.DataFrame, desired: int, mutable: list[str],
    max_changes: int = 2, n_results: int = 5, domains: dict[str, Domain] | None = None,
) -> CounterfactualResult:
    """Find the smallest changes (1 or 2 features) that make the model predict `desired`."""
    if run.task != "classification":
        raise MLError("Counterfactuals are available for classification problems.",
                      "Use the What-if tab to explore a regression model.")
    if not 0 <= desired < len(run.classes):
        raise MLError("Unknown desired class.")
    domains = domains or feature_domains(run)
    mutable = [c for c in mutable if c in domains]
    if not mutable:
        raise MLError("No features can be changed.", "Select at least one feature.")

    p0, _ = predict_frame(run, res, row)
    current = int(np.argmax(p0[0]))
    if current == desired:
        raise MLError("The model already predicts this class for the row.", "Choose a different desired class.")

    r0 = row.iloc[0]
    found: list[Counterfactual] = []
    tested = 0

    singles: list[tuple[str, Any]] = [
        (f, v) for f in mutable for v in _candidates(domains[f], r0[f], CF_GRID)
    ]
    if not singles:
        raise MLError("No alternative values were found.", "Choose other features.")
    proba, _ = predict_frame(run, res, _make_batch(row, [{f: v} for f, v in singles]))
    tested += len(singles)
    best_by_feature: dict[str, float] = {}
    for (f, v), p in zip(singles, proba):
        best_by_feature[f] = max(best_by_feature.get(f, 0.0), float(p[desired]))
        if int(np.argmax(p)) == desired:
            found.append(Counterfactual([(f, r0[f], v)], float(p[desired]), _distance(domains[f], r0[f], v)))

    if max_changes >= 2:
        flips_alone = {cf.changes[0][0] for cf in found}
        pool = [f for f in sorted(best_by_feature, key=best_by_feature.get, reverse=True)
                if f not in flips_alone][:PAIR_FEATURES]
        pairs: list[tuple[str, Any, str, Any]] = []
        for a, b in itertools.combinations(pool, 2):
            for x in _candidates(domains[a], r0[a], PAIR_GRID):
                for y in _candidates(domains[b], r0[b], PAIR_GRID):
                    pairs.append((a, x, b, y))
        if pairs:
            proba, _ = predict_frame(run, res, _make_batch(row, [{a: x, b: y} for a, x, b, y in pairs]))
            tested += len(pairs)
            for (a, x, b, y), p in zip(pairs, proba):
                if int(np.argmax(p)) == desired:
                    found.append(Counterfactual(
                        [(a, r0[a], x), (b, r0[b], y)], float(p[desired]),
                        _distance(domains[a], r0[a], x) + _distance(domains[b], r0[b], y)))

    found.sort(key=lambda cf: (cf.distance, -cf.prob))
    return CounterfactualResult(current, float(p0[0][desired]), found[:n_results], tested)