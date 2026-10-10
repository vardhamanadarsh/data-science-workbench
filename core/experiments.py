"""Local experiment tracking: one JSON file per saved experiment (UI-free)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config import settings
from core.model_training import ModelResult, TrainingRun

_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_METRICS = ["F1", "Accuracy", "ROC-AUC", "RMSE", "MAE", "R2"]


def _folder() -> Path:
    folder = Path(settings.experiments_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _path(exp_id: str) -> Path:
    if not _ID.fullmatch(exp_id) or ".." in exp_id:
        raise ValueError("Invalid experiment id.")
    return _folder() / f"{exp_id}.json"


def _hyperparameters(pipe: Any) -> dict[str, Any]:
    model = getattr(pipe, "named_steps", {}).get("model")
    if model is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in model.get_params(deep=False).items():
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:
                pass
        if value is None or isinstance(value, (bool, int, float, str)):
            out[key] = value
    return out


def save_experiment(
    run: TrainingRun, result: ModelResult, dataset_info: dict[str, Any],
    preprocessing_steps: list[str], note: str = "",
) -> str:
    """Write one experiment to disk and return its id."""
    if not result.ok:
        raise ValueError("Only successfully trained models can be saved.")
    slug = re.sub(r"[^a-z0-9]+", "-", result.name.lower()).strip("-") or "model"
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    exp_id, n = base, 2
    while _path(exp_id).exists():
        exp_id, n = f"{base}-{n}", n + 1

    record = {
        "id": exp_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "app_version": settings.version,
        "note": note,
        "dataset": dataset_info,
        "preprocessing_steps": preprocessing_steps,
        "task": run.task,
        "target": run.config.target,
        "features": run.num + run.cat,
        "classes": run.classes,
        "model": result.name,
        "hyperparameters": _hyperparameters(result.pipeline),
        "settings": asdict(run.config),
        "rows_train": len(run.X_train),
        "rows_test": len(run.X_test),
        "metrics": {
            "test": {k: v for k, v in result.test.items() if v is not None},
            "cv": {k: {"mean": m, "std": s} for k, (m, s) in result.cv.items()},
        },
        "fit_seconds": round(result.fit_seconds, 3),
    }
    path = _path(exp_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return exp_id


def list_experiments() -> list[dict[str, Any]]:
    """All saved experiments, newest first. Unreadable files are skipped."""
    records: list[dict[str, Any]] = []
    for file in _folder().glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "id" in data:
                records.append(data)
        except Exception as exc:
            logger.warning("Skipping unreadable experiment {}: {}", file.name, exc)
    return sorted(records, key=lambda r: r.get("created", ""), reverse=True)


def load_experiment(exp_id: str) -> dict[str, Any]:
    path = _path(exp_id)
    if not path.exists():
        raise FileNotFoundError(f"Experiment '{exp_id}' was not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_experiment(exp_id: str) -> None:
    _path(exp_id).unlink(missing_ok=True)


def experiments_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        test = r.get("metrics", {}).get("test", {})
        row: dict[str, Any] = {
            "id": r.get("id"),
            "created": str(r.get("created", ""))[:16].replace("T", " "),
            "dataset": r.get("dataset", {}).get("name"),
            "task": r.get("task"),
            "target": r.get("target"),
            "model": r.get("model"),
            "features": len(r.get("features", [])),
        }
        for k in _METRICS:
            if k in test:
                row[f"Test {k}"] = test[k]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_hyperparameters(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Only the settings that differ between the chosen experiments."""
    keys = sorted({k for r in records for k in r.get("hyperparameters", {})})
    rows = []
    for k in keys:
        values = [r.get("hyperparameters", {}).get(k, "—") for r in records]
        if len({str(v) for v in values}) > 1:
            rows.append({"parameter": k, **{r["id"]: str(v) for r, v in zip(records, values)}})
    return pd.DataFrame(rows)