"""Saving and loading trained models with skops (safer than pickle)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from config import settings
from core.model_training import ModelResult, TrainingRun

FORMAT_VERSION = 1
_REQUIRED = {"format_version", "task", "target", "model_name", "classes", "num", "cat", "pipeline"}
_INSTALL = "uv sync --extra ml --extra automl --extra explain --extra reports"


class ModelIOError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


def can_save(res: ModelResult) -> bool:
    pipe = res.pipeline
    return (res.ok and res.name != "Baseline" and pipe is not None
            and hasattr(pipe, "named_steps") and "model" in pipe.named_steps)


def _skops() -> Any:
    try:
        import skops.io as sio
    except ImportError as exc:
        raise ModelIOError("Saving and loading models needs the 'skops' package.", f"Install it with: {_INSTALL}") from exc
    return sio


def save_model_bytes(run: TrainingRun, res: ModelResult) -> bytes:
    """Serialise the fitted pipeline plus the information needed to use it."""
    if not can_save(res):
        raise ModelIOError("This model cannot be saved.", "Choose a model trained in Machine Learning or AutoML (not the Baseline or FLAML).")
    sio = _skops()
    bundle = {
        "format_version": FORMAT_VERSION, "app_version": settings.version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": run.task, "target": run.config.target, "model_name": res.name,
        "classes": list(run.classes), "num": list(run.num), "cat": list(run.cat),
        "test_metrics": {k: float(v) for k, v in res.test.items() if v is not None},
        "pipeline": res.pipeline,
    }
    try:
        return sio.dumps(bundle)
    except Exception as exc:
        raise ModelIOError("This model could not be saved with skops.",
                           f"Details: {type(exc).__name__}: {exc}") from exc


def inspect_untrusted(data: bytes) -> list[str]:
    """Types inside the file that skops does not trust by default (nothing is executed)."""
    sio = _skops()
    try:
        return list(sio.get_untrusted_types(data=data))
    except Exception as exc:
        raise ModelIOError("This does not look like a valid model file.", f"Details: {type(exc).__name__}: {exc}") from exc


def load_model_bytes(data: bytes, trusted: list[str]) -> dict[str, Any]:
    """Load a model file. `trusted` must list every untrusted type the user approved."""
    sio = _skops()
    try:
        bundle = sio.loads(data, trusted=list(trusted))
    except Exception as exc:
        raise ModelIOError("The model file could not be loaded.",
                           f"It may be damaged, or not all of its types were trusted. Details: {type(exc).__name__}: {exc}") from exc
    if not isinstance(bundle, dict) or not _REQUIRED <= set(bundle):
        raise ModelIOError("This is not a model file made by this app.")
    return bundle


def _to_features(df: pl.DataFrame, num: list[str], cat: list[str]) -> Any:
    missing = [c for c in num + cat if c not in df.columns]
    if missing:
        raise ModelIOError(f"The data is missing column(s) the model needs: {', '.join(missing)}.",
                           "Add them, or rename the columns to match.")
    exprs: list[pl.Expr] = []
    for c in num:
        e = pl.col(c).cast(pl.Float64, strict=False)
        exprs.append(pl.when(e.is_finite()).then(e).otherwise(None).alias(c))
    for c in cat:
        exprs.append(pl.col(c).cast(pl.String).alias(c))
    X = df.select(exprs).to_pandas()
    for c in cat:
        col = X[c].astype(object)
        X[c] = col.where(col.notna(), np.nan)
    return X


def predict_from_frame(bundle: dict[str, Any], df: pl.DataFrame) -> pl.DataFrame:
    """Add prediction (and class probability) columns to the given data."""
    if df.height == 0:
        raise ModelIOError("The data has no rows.")
    X = _to_features(df, list(bundle["num"]), list(bundle["cat"]))
    pipe = bundle["pipeline"]
    try:
        pred = np.asarray(pipe.predict(X))
    except Exception as exc:
        raise ModelIOError("The model could not make predictions for this data.",
                           f"Details: {type(exc).__name__}: {exc}") from exc
    name = "prediction" if "prediction" not in df.columns else "model_prediction"
    if bundle["task"] != "classification":
        return df.with_columns(pl.Series(name, pred.astype(float)))
    classes = list(bundle["classes"])
    out = df.with_columns(pl.Series(name, [classes[int(i)] for i in pred]))
    try:
        proba = np.asarray(pipe.predict_proba(X))
    except Exception:
        proba = None
    if proba is not None and proba.ndim == 2 and proba.shape[1] == len(classes):
        out = out.with_columns([pl.Series(f"probability_{c}", proba[:, i].astype(float))
                                for i, c in enumerate(classes)])
    return out