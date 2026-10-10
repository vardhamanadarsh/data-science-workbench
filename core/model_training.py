"""Model training with leak-free scikit-learn pipelines (UI-free).

Every learned step (imputing, scaling, encoding, feature selection, resampling)
lives INSIDE the pipeline, so it is fitted on training rows only, and inside each
cross-validation fold. Nothing is learned from the test set.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import ProcessingMode
from core.data_quality import infer_kind, is_classification_target
from utils.optional_imports import MissingDependencyError, is_available, lazy_import

MAX_CATEGORIES = 50
SLOW_MAX_ROWS = 20_000
MIN_ROWS = 20
ML_ROW_CAPS: dict[ProcessingMode, int | None] = {
    ProcessingMode.FAST: 10_000,
    ProcessingMode.STANDARD: 50_000,
    ProcessingMode.FULL: None,
}
IMBALANCE_OPTIONS: dict[str, str] = {
    "none": "None",
    "class_weight": "Class weights",
    "oversample": "Random oversampling",
    "undersample": "Random undersampling",
    "smote": "SMOTE",
}


class MLError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


# --------------------------------------------------------------------------- #
# Configuration and results
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    target: str
    features: list[str]
    task: str                          # "classification" | "regression"
    test_size: float = 0.2
    cv_folds: int = 5                  # 0 or 1 = no cross-validation
    scale: bool = True
    num_impute: str = "median"         # median | mean
    select_percentile: int = 100       # 100 = keep every feature
    time_split: bool = False           # last rows become the test set
    imbalance: str = "none"
    positive_class: str | None = None  # binary classification only
    parallel: bool = True
    max_rows: int | None = None
    seed: int = 42


@dataclass
class ModelResult:
    name: str
    ok: bool = True
    error: str = ""
    pipeline: Any = None
    cv: dict[str, tuple[float, float]] = field(default_factory=dict)
    test: dict[str, float | None] = field(default_factory=dict)
    fit_seconds: float = 0.0
    y_pred: np.ndarray | None = None
    scores: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class TrainingRun:
    config: TrainConfig
    classes: list[str]
    num: list[str]
    cat: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    n_available: int
    n_used: int
    notes: list[str] = field(default_factory=list)
    results: list[ModelResult] = field(default_factory=list)
    importance: dict[str, pd.DataFrame] = field(default_factory=dict)
    signature: tuple = ()
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def task(self) -> str:
        return self.config.task


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[int, bool], Any]     # (seed, balanced) -> estimator
    import_name: str | None = None          # optional package needed
    slow: bool = False
    default: bool = False
    class_weight: bool = False

    @property
    def available(self) -> bool:
        return self.import_name is None or is_available(self.import_name)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

def _bal(flag: bool) -> str | None:
    return "balanced" if flag else None


def _xgb_c(seed: int, bal: bool) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1,
                         random_state=seed, tree_method="hist", verbosity=0)


def _xgb_r(seed: int, bal: bool) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1,
                        random_state=seed, tree_method="hist", verbosity=0)


def _lgb_c(seed: int, bal: bool) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, class_weight=_bal(bal), random_state=seed,
                          n_jobs=-1, verbose=-1)


def _lgb_r(seed: int, bal: bool) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=300, random_state=seed, n_jobs=-1, verbose=-1)


CLASSIFIERS: list[ModelSpec] = [
    ModelSpec("Logistic Regression",
              lambda s, b: LogisticRegression(max_iter=1000, class_weight=_bal(b), random_state=s),
              default=True, class_weight=True),
    ModelSpec("KNN", lambda s, b: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ModelSpec("Decision Tree",
              lambda s, b: DecisionTreeClassifier(max_depth=10, class_weight=_bal(b), random_state=s),
              default=True, class_weight=True),
    ModelSpec("Random Forest",
              lambda s, b: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight=_bal(b), random_state=s),
              default=True, class_weight=True),
    ModelSpec("SVM", lambda s, b: SVC(kernel="rbf", class_weight=_bal(b), random_state=s),
              slow=True, class_weight=True),
    ModelSpec("Gradient Boosting",
              lambda s, b: HistGradientBoostingClassifier(class_weight=_bal(b), random_state=s),
              default=True, class_weight=True),
    ModelSpec("XGBoost", _xgb_c, import_name="xgboost"),
    ModelSpec("LightGBM", _lgb_c, import_name="lightgbm", class_weight=True),
]

REGRESSORS: list[ModelSpec] = [
    ModelSpec("Linear Regression", lambda s, b: LinearRegression(), default=True),
    ModelSpec("Ridge", lambda s, b: Ridge(alpha=1.0), default=True),
    ModelSpec("Lasso", lambda s, b: Lasso(alpha=0.1, max_iter=10_000)),
    ModelSpec("KNN", lambda s, b: KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    ModelSpec("Decision Tree", lambda s, b: DecisionTreeRegressor(max_depth=10, random_state=s)),
    ModelSpec("Random Forest",
              lambda s, b: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=s), default=True),
    ModelSpec("SVR", lambda s, b: SVR(kernel="rbf"), slow=True),
    ModelSpec("Gradient Boosting", lambda s, b: HistGradientBoostingRegressor(random_state=s), default=True),
    ModelSpec("XGBoost", _xgb_r, import_name="xgboost"),
    ModelSpec("LightGBM", _lgb_r, import_name="lightgbm"),
]


def model_specs(task: str) -> list[ModelSpec]:
    return CLASSIFIERS if task == "classification" else REGRESSORS


# --------------------------------------------------------------------------- #
# Task and feature detection
# --------------------------------------------------------------------------- #

def detect_task(target: pl.Series) -> str:
    """'classification' or 'regression' from the target column."""
    if target.dtype.is_numeric():
        return "classification" if is_classification_target(target) else "regression"
    return "classification"


def eligible_features(df: pl.DataFrame, target: str) -> tuple[list[str], dict[str, str]]:
    """Columns that make sensible predictors, and why the others were left out."""
    ok: list[str] = []
    skipped: dict[str, str] = {}
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        non_null = s.drop_nulls()
        n_unique = int(non_null.n_unique()) if non_null.len() else 0
        if n_unique <= 1:
            skipped[col] = "constant or empty"
            continue
        kind = infer_kind(s, n_unique)
        if kind == "datetime":
            skipped[col] = "date column (add date parts in Preprocessing instead)"
        elif kind == "id":
            skipped[col] = "looks like an ID"
        elif kind == "text":
            skipped[col] = "free text"
        elif kind == "other":
            skipped[col] = f"unsupported type ({s.dtype})"
        elif kind == "categorical" and n_unique > MAX_CATEGORIES:
            skipped[col] = f"too many categories ({n_unique:,})"
        else:
            ok.append(col)
    return ok, skipped


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

@dataclass
class _Prepared:
    X: pd.DataFrame
    y: np.ndarray
    classes: list[str]
    num: list[str]
    cat: list[str]
    notes: list[str]
    n_available: int


def _prepare(df: pl.DataFrame, cfg: TrainConfig) -> _Prepared:
    needed = [cfg.target, *cfg.features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise MLError(f"Column(s) not found: {', '.join(missing)}.", "Choose the columns again.")

    notes: list[str] = []
    frame = df.select(list(dict.fromkeys(needed))).filter(pl.col(cfg.target).is_not_null())
    if frame.schema[cfg.target].is_float():
        frame = frame.filter(pl.col(cfg.target).is_finite())
    dropped = df.height - frame.height
    if dropped:
        notes.append(f"{dropped:,} row(s) with a missing target were left out.")
    n_available = frame.height

    if cfg.max_rows and frame.height > cfg.max_rows:
        if cfg.time_split:
            step = -(-frame.height // cfg.max_rows)
            frame = frame.gather_every(step)
        else:
            frame = frame.sample(n=cfg.max_rows, seed=cfg.seed)
        notes.append(f"Training uses {frame.height:,} of {n_available:,} rows (limit of the processing mode).")
    if frame.height < MIN_ROWS:
        raise MLError(f"Only {frame.height} usable rows.", f"At least {MIN_ROWS} are needed.")

    num = [c for c in cfg.features if frame[c].dtype.is_numeric()]
    cat = [c for c in cfg.features if c not in num]
    exprs: list[pl.Expr] = []
    for c in num:
        e = pl.col(c).cast(pl.Float64, strict=False)
        exprs.append(pl.when(e.is_finite()).then(e).otherwise(None).alias(c))
    for c in cat:
        exprs.append(pl.col(c).cast(pl.String).alias(c))
    X = frame.select(exprs).to_pandas()
    for c in cat:
        col = X[c].astype(object)
        X[c] = col.where(col.notna(), np.nan)

    empty = [c for c in X.columns if X[c].isna().all()]
    if empty:
        X = X.drop(columns=empty)
        num = [c for c in num if c not in empty]
        cat = [c for c in cat if c not in empty]
        notes.append(f"Columns with no values were ignored: {', '.join(empty)}.")
    if X.shape[1] == 0:
        raise MLError("No usable feature columns.", "Select other features.")

    raw = frame[cfg.target]
    if cfg.task == "regression":
        if not raw.dtype.is_numeric():
            raise MLError("A regression target must be numeric.", "Choose a numeric column or use classification.")
        y = raw.cast(pl.Float64).to_numpy()
        return _Prepared(X, y, [], num, cat, notes, n_available)

    classes = raw.unique().sort().cast(pl.String).to_list()
    if len(classes) < 2:
        raise MLError("The target has only one class.", "Choose another target column.")
    if len(classes) > 50:
        raise MLError(f"The target has {len(classes)} classes.", "Classification supports up to 50 classes.")
    if len(classes) == 2 and cfg.positive_class in classes:
        classes = [c for c in classes if c != cfg.positive_class] + [cfg.positive_class]
    labels = raw.cast(pl.String).to_numpy()
    y = pd.Categorical(labels, categories=classes).codes.astype(int)
    return _Prepared(X, y, classes, num, cat, notes, n_available)


def _split(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig):
    n = len(X)
    if cfg.task == "classification" and np.bincount(y).min() < 2:
        raise MLError("A class has fewer than 2 rows.", "Remove or merge rare classes first.")
    if cfg.time_split:
        cut = int(round(n * (1 - cfg.test_size)))
        if cut < 10 or n - cut < 5:
            raise MLError("The split leaves too few rows on one side.", "Change the test size.")
        return (X.iloc[:cut].reset_index(drop=True), X.iloc[cut:].reset_index(drop=True), y[:cut], y[cut:])
    stratify = y if cfg.task == "classification" else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
        )
    except ValueError as exc:
        raise MLError("The data could not be split.", f"Details: {exc}") from exc
    return X_tr.reset_index(drop=True), X_te.reset_index(drop=True), y_tr, y_te


# --------------------------------------------------------------------------- #
# Pipeline building
# --------------------------------------------------------------------------- #

def build_preprocessor(num: list[str], cat: list[str], cfg: TrainConfig) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if num:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy=cfg.num_impute))]
        if cfg.scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num))
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _make_sampler(kind: str, y_train: np.ndarray, seed: int) -> Any:
    try:
        lazy_import("imblearn")
    except MissingDependencyError as exc:
        raise MLError("Resampling needs the 'imbalanced-learn' package.", str(exc)) from exc
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if kind == "oversample":
        return RandomOverSampler(random_state=seed)
    if kind == "undersample":
        return RandomUnderSampler(random_state=seed)
    smallest = int(np.bincount(y_train).min())
    if smallest < 4:
        raise MLError("SMOTE needs at least 4 rows in the smallest class.",
                      "Use random oversampling or class weights instead.")
    return SMOTE(k_neighbors=max(1, min(5, int(smallest * 0.7) - 1)), random_state=seed)


def build_pipeline(estimator: Any, pre: ColumnTransformer, cfg: TrainConfig, sampler: Any = None) -> Any:
    steps: list[tuple[str, Any]] = [("prep", pre)]
    if cfg.select_percentile < 100:
        score = f_classif if cfg.task == "classification" else f_regression
        steps.append(("select", SelectPercentile(score, percentile=cfg.select_percentile)))
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([*steps, ("sampler", sampler), ("model", estimator)])
    return Pipeline([*steps, ("model", estimator)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _scores(pipe: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Probability of the positive class (binary) or class probabilities; else decision values."""
    if hasattr(pipe, "predict_proba"):
        try:
            p = np.asarray(pipe.predict_proba(X))
            return p[:, 1] if p.shape[1] == 2 else p
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        try:
            return np.asarray(pipe.decision_function(X))
        except Exception:
            pass
    return None


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None, n_classes: int
) -> dict[str, float | None]:
    """Binary uses the positive class; multiclass uses weighted averages."""
    avg = "binary" if n_classes == 2 else "weighted"
    out: dict[str, float | None] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "ROC-AUC": None,
    }
    if scores is not None:
        try:
            if scores.ndim == 1:
                out["ROC-AUC"] = float(roc_auc_score(y_true, scores))
            elif np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
                out["ROC-AUC"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="weighted"))
        except ValueError:
            pass
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _make_cv(cfg: TrainConfig, y_train: np.ndarray) -> tuple[Any, str]:
    k = cfg.cv_folds
    if k < 2:
        return None, ""
    if cfg.time_split:
        return TimeSeriesSplit(n_splits=k), ""
    if cfg.task == "classification":
        n = min(k, int(np.bincount(y_train).min()))
        if n < 2:
            return None, "Cross-validation was skipped: the smallest class has fewer than 2 training rows."
        note = f"Cross-validation uses {n} folds (limited by the smallest class)." if n < k else ""
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=cfg.seed), note
    return KFold(n_splits=k, shuffle=True, random_state=cfg.seed), ""


def _cv_scoring(run: TrainingRun) -> tuple[dict[str, str], dict[str, int]]:
    if run.task == "classification":
        f1 = "f1" if len(run.classes) == 2 else "f1_weighted"
        return {"Accuracy": "accuracy", "F1": f1}, {"Accuracy": 1, "F1": 1}
    return (
        {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
        {"MAE": -1, "RMSE": -1, "R2": 1},
    )


def _train_one(name: str, spec: ModelSpec | None, run: TrainingRun, cv: Any) -> ModelResult:
    cfg = run.config
    res = ModelResult(name=name)
    clf = cfg.task == "classification"
    try:
        if spec is None:
            estimator = DummyClassifier(strategy="prior") if clf else DummyRegressor(strategy="mean")
            sampler = None
        else:
            if spec.slow and len(run.X_train) > SLOW_MAX_ROWS:
                raise MLError(
                    f"Skipped: {name} is too slow for {len(run.X_train):,} training rows (limit {SLOW_MAX_ROWS:,}).",
                    "Use Fast Preview or choose faster models.",
                )
            balanced = clf and cfg.imbalance == "class_weight"
            if balanced and not spec.class_weight:
                res.notes.append("Class weights are not supported by this model.")
            estimator = spec.factory(cfg.seed, balanced and spec.class_weight)
            sampler = _make_sampler(cfg.imbalance, run.y_train, cfg.seed) if clf and cfg.imbalance in (
                "oversample", "undersample", "smote") else None

        pipe = build_pipeline(estimator, build_preprocessor(run.num, run.cat, cfg), cfg, sampler)
        scoring, sign = _cv_scoring(run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cv is not None:
                out = cross_validate(pipe, run.X_train, run.y_train, cv=cv, scoring=scoring,
                                     n_jobs=-1 if cfg.parallel else 1, error_score="raise")
                for key in scoring:
                    vals = sign[key] * np.asarray(out[f"test_{key}"], dtype=float)
                    res.cv[key] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
            start = time.perf_counter()
            pipe.fit(run.X_train, run.y_train)
            res.fit_seconds = time.perf_counter() - start
            res.y_pred = np.asarray(pipe.predict(run.X_test))
            if clf:
                res.scores = _scores(pipe, run.X_test)
        res.pipeline = pipe
        res.test = (classification_metrics(run.y_test, res.y_pred, res.scores, len(run.classes))
                    if clf else regression_metrics(run.y_test, res.y_pred))
    except MLError as err:
        res.ok, res.error = False, err.message
    except MissingDependencyError as err:
        res.ok, res.error = False, str(err)
    except Exception as exc:
        logger.warning("Model {} failed: {}", name, exc)
        res.ok, res.error = False, f"{type(exc).__name__}: {exc}"
    return res


def train_models(
    df: pl.DataFrame,
    cfg: TrainConfig,
    model_names: list[str],
    progress: Callable[[int, int, str], None] | None = None,
) -> TrainingRun:
    """Prepare data, split, then train a baseline plus the chosen models.

    Raises:
        MLError: for problems with the setup. Individual model failures are
        recorded in the results instead of stopping the run.
    """
    if cfg.task not in ("classification", "regression"):
        raise MLError(f"Unknown task '{cfg.task}'.")
    if not cfg.features:
        raise MLError("No features selected.", "Choose at least one feature.")
    if cfg.target in cfg.features:
        raise MLError("The target cannot also be a feature.", "Remove it from the features.")
    specs = {s.name: s for s in model_specs(cfg.task)}
    unknown = [n for n in model_names if n not in specs]
    if unknown:
        raise MLError(f"Unknown model(s): {', '.join(unknown)}.")

    data = _prepare(df, cfg)
    X_train, X_test, y_train, y_test = _split(data.X, data.y, cfg)
    run = TrainingRun(
        config=cfg, classes=data.classes, num=data.num, cat=data.cat,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        n_available=data.n_available, n_used=len(data.X), notes=list(data.notes),
    )
    cv, cv_note = _make_cv(cfg, y_train)
    if cv_note:
        run.notes.append(cv_note)
    if cfg.task == "classification" and cfg.imbalance in ("oversample", "undersample", "smote"):
        _make_sampler(cfg.imbalance, y_train, cfg.seed)   # fail early with a clear message
        run.notes.append("Resampling is applied to training rows only (inside each cross-validation fold).")
    if cfg.time_split:
        run.notes.append("Time-based split: the last rows are the test set. Make sure the rows are sorted by time.")

    entries: list[tuple[str, ModelSpec | None]] = [("Baseline", None)] + [(n, specs[n]) for n in model_names]
    for i, (name, spec) in enumerate(entries):
        if progress:
            progress(i, len(entries), name)
        run.results.append(_train_one(name, spec, run, cv))
    if progress:
        progress(len(entries), len(entries), "done")
    return run


# --------------------------------------------------------------------------- #
# Reading results
# --------------------------------------------------------------------------- #

def _rank_value(r: ModelResult, clf: bool) -> tuple[int, float]:
    if not r.ok:
        return 1, 0.0
    if clf:
        v = r.test.get("F1")
        return 0, -(v if v is not None else -1.0)
    v = r.test.get("RMSE")
    return 0, (v if v is not None else float("inf"))


def ranked(run: TrainingRun) -> list[ModelResult]:
    """Best first: F1 (classification) or RMSE (regression). Failed models last."""
    clf = run.task == "classification"
    return sorted(run.results, key=lambda r: _rank_value(r, clf))


def best_result(run: TrainingRun) -> ModelResult | None:
    ok = [r for r in ranked(run) if r.ok]
    real = [r for r in ok if r.name != "Baseline"]
    return (real or ok or [None])[0]


def results_table(run: TrainingRun) -> pd.DataFrame:
    clf = run.task == "classification"
    cv_keys = ["Accuracy", "F1"] if clf else ["MAE", "RMSE", "R2"]
    test_keys = ["Accuracy", "F1", "ROC-AUC", "Precision", "Recall"] if clf else ["MAE", "RMSE", "R2"]
    rows = []
    for r in ranked(run):
        row: dict[str, Any] = {"Model": r.name}
        for k in cv_keys:
            if k in r.cv:
                m, s = r.cv[k]
                row[f"CV {k}"] = f"{m:.3f} ± {s:.3f}"
        for k in test_keys:
            row[f"Test {k}"] = r.test.get(k)
        row["Train time (s)"] = round(r.fit_seconds, 2) if r.ok else None
        row["Status"] = "ok" if r.ok else r.error
        rows.append(row)
    columns = (["Model"]
               + [f"CV {k}" for k in cv_keys if any(k in r.cv for r in run.results)]
               + [f"Test {k}" for k in test_keys] + ["Train time (s)", "Status"])
    return pd.DataFrame(rows).reindex(columns=columns)