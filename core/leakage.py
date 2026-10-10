"""Dedicated leakage detector (UI-free).

Every finding is only a POSSIBLE leakage. Only the user knows whether a column
is really available at prediction time.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import settings
from core.data_quality import _target_as_numeric, find_leakage_candidates
from core.model_training import TrainingRun, detect_task

Strength = Literal["strong", "moderate", "weak"]
_ORDER = {"strong": 0, "moderate": 1, "weak": 2}
_THRESHOLDS = {"classification": (0.97, 0.90), "regression": (0.95, 0.85)}   # (strong, moderate)
RANK_COPY_THRESHOLD = 0.98
MAX_SINGLE_FEATURES = 80
_POST_OUTCOME = re.compile(
    r"(outcome|result|after|post_|_post|final|resolved|closure|closed|settled|cancel|refund|terminat|churn_|_churn|default|label)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    check: str
    strength: Strength
    detail: str
    score: float | None = None


@dataclass
class LeakageReport:
    target: str
    task: str
    findings: list[LeakageFinding]
    rows_checked: int
    columns_checked: int

    def counts(self) -> dict[str, int]:
        cols: dict[str, int] = {"strong": 0, "moderate": 0, "weak": 0}
        best: dict[str, str] = {}
        for f in self.findings:                      # strongest level per column
            if f.column not in best or _ORDER[f.strength] < _ORDER[best[f.column]]:
                best[f.column] = f.strength
        for level in best.values():
            cols[level] += 1
        return cols

    def table(self) -> pl.DataFrame:
        rows = [{"column": f.column, "check": f.check, "strength": f.strength,
                 "score": f.score, "detail": f.detail} for f in self.findings]
        return pl.DataFrame(rows, schema={"column": pl.String, "check": pl.String, "strength": pl.String,
                                          "score": pl.Float64, "detail": pl.String})


def _finite(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _check_name(reason: str) -> str:
    if reason.startswith("Values are identical"):
        return "Identical copy of the target"
    if reason.startswith("Correlation"):
        return "Near-perfect correlation"
    if reason.startswith("Its values almost"):
        return "Categories determine the target"
    return "Name resembles the target"


def _rank_copies(frame: pl.DataFrame, target: str, feats: list[str], skip: set[str]) -> list[LeakageFinding]:
    """Monotonic transformations of the target (e.g. log or square of it)."""
    t = _target_as_numeric(frame[target])
    if t is None:
        return []
    num = [c for c in feats if c not in skip and frame[c].dtype.is_numeric()][:200]
    if not num:
        return []
    data = frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in num]).with_columns(t.alias("__t__"))
    pdf = data.to_pandas().replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = pdf[num].corrwith(pdf["__t__"], method="spearman").abs()
    out = []
    for col, value in corr.items():
        v = _finite(value)
        if v is not None and v >= RANK_COPY_THRESHOLD:
            out.append(LeakageFinding(
                str(col), "Monotonic copy of the target", "strong",
                f"Rank correlation with the target is {v:.3f}, so it may be a transformed copy of it.", round(v, 4)))
    return out


def _as_single_feature(s: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_bool_dtype(s):
        x = s.astype(float)
    elif pd.api.types.is_datetime64_any_dtype(s):
        x = (s - s.min()).dt.total_seconds()
    elif pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        codes = pd.factorize(s.astype(str))[0].astype(float)
        return codes.reshape(-1, 1) if len(np.unique(codes)) >= 2 else None
    if x.notna().sum() < 20 or x.nunique() < 2:
        return None
    return x.fillna(x.min() - 1.0).to_numpy(dtype=float).reshape(-1, 1)


def _single_feature_findings(
    frame: pl.DataFrame, target: str, feats: list[str], task: str, seed: int, skip: set[str]
) -> list[LeakageFinding]:
    """Fit a shallow tree on ONE column at a time. Very high CV scores are suspicious."""
    pdf = frame.select([target, *feats]).to_pandas()
    if task == "classification":
        y, uniques = pd.factorize(pdf[target].astype(str))
        if len(uniques) < 2 or np.bincount(y).min() < 3:
            return []
        cv: Any = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scoring = "roc_auc" if len(uniques) == 2 else "balanced_accuracy"
    else:
        y_all = pd.to_numeric(pdf[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        pdf, y = pdf[keep], y_all[keep]
        if len(pdf) < 30:
            return []
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scoring = "r2"
    leaf = max(5, len(pdf) // 200)
    strong_t, moderate_t = _THRESHOLDS[task]
    label = {"roc_auc": "ROC-AUC", "balanced_accuracy": "balanced accuracy", "r2": "R²"}[scoring]

    out: list[LeakageFinding] = []
    checked = 0
    for col in feats:
        if col in skip:
            continue
        if checked >= MAX_SINGLE_FEATURES:
            break
        x = _as_single_feature(pdf[col])
        if x is None:
            continue
        model = (DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor)(
            max_depth=4, min_samples_leaf=leaf, random_state=seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = float(np.mean(cross_val_score(model, x, y, cv=cv, scoring=scoring)))
        except Exception as exc:
            logger.debug("Single-feature check skipped for {}: {}", col, exc)
            continue
        checked += 1
        strength = "strong" if score >= strong_t else "moderate" if score >= moderate_t else None
        if strength:
            out.append(LeakageFinding(
                col, "Single-feature predictive power", strength,
                f"This column alone predicts the target with {label} {score:.3f} in cross-validation. "
                "Check whether it is really known at prediction time.", round(score, 4)))
    return out


def assess_leakage(
    df: pl.DataFrame, target: str, features: list[str] | None = None,
    max_rows: int = 20_000, seed: int | None = None,
) -> LeakageReport:
    """Run all leakage checks. Raises ValueError for unusable input."""
    seed = settings.sample_seed if seed is None else seed
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    feats = [c for c in (features if features is not None else df.columns) if c != target and c in df.columns]
    if not feats:
        raise ValueError("There are no feature columns to check.")
    frame = df.select([target, *feats]).filter(pl.col(target).is_not_null())
    if frame.height < 30:
        raise ValueError("At least 30 rows with a known target are needed.")
    if frame.height > max_rows:
        frame = frame.sample(n=max_rows, seed=seed)
    task = detect_task(frame[target])

    findings: list[LeakageFinding] = [
        LeakageFinding(c.column, _check_name(c.reason), c.strength, c.reason, c.score)
        for c in find_leakage_candidates(frame, target, max_rows=max_rows)
    ]
    strong = lambda: {f.column for f in findings if f.strength == "strong"}

    findings += _rank_copies(frame, target, feats, strong())
    findings += _single_feature_findings(frame, target, feats, task, seed, strong())
    for col in feats:
        if col not in strong() and _POST_OUTCOME.search(col):
            findings.append(LeakageFinding(
                col, "Name suggests it is recorded after the outcome", "weak",
                "The column name contains a word that often marks information known only after the "
                "outcome. Check when this value is recorded."))

    findings.sort(key=lambda f: (_ORDER[f.strength], -(f.score or 0.0), f.column))
    return LeakageReport(target, task, findings, frame.height, len(feats))


def check_contamination(run: TrainingRun) -> dict[str, Any]:
    """How many test rows have exactly the same feature values as a training row."""
    train_hash = pd.util.hash_pandas_object(run.X_train, index=False).to_numpy()
    test_hash = pd.util.hash_pandas_object(run.X_test, index=False).to_numpy()
    n = int(np.isin(test_hash, train_hash).sum())
    share = n / max(len(test_hash), 1)
    level = "strong" if share >= 0.20 else "moderate" if share >= 0.05 else "ok"
    return {"n": n, "rows_test": len(test_hash), "share": share, "level": level}