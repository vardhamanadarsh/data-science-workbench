"""Save and load a project file (.wbproj, a zip with data + manifest). UI-free."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from config import settings
from core.data_loader import load_dataset

FORMAT_VERSION = 1
MAX_BYTES = 2 * 1024**3   # refuse to unpack more than 2 GB
# Results that were computed from the old data and must be cleared on load
DERIVED_KEYS = ["health", "eda", "leakage_report", "anomaly_result", "fairness_result", "ml_run",
                "automl_run", "shap_cache", "history", "compare_result", "rp_files", "rp_loaded", "rp_pred"]


class ProjectError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


def save_project_bytes(state: Mapping[str, Any]) -> bytes:
    ds = state.get("dataset")
    if ds is None:
        raise ProjectError("There is no dataset to save.", "Load a dataset first.")
    history = state.get("history")
    steps = [s.label for s in history.snapshots[1:history.index + 1]] if history is not None else []
    health = (state.get("health") or {}).get("report")
    notes = {k[8:]: v for k, v in state.items() if isinstance(k, str) and k.startswith("rp_note_") and v}
    manifest = {
        "format_version": FORMAT_VERSION, "app_version": settings.version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": ds.name, "rows": ds.df.height, "columns": ds.df.width,
        "sampled_from_rows": ds.total_rows if ds.sampled else None,
        "preprocessing_steps": steps, "target": health.target if health is not None else None,
        "report_notes": notes,
    }
    buf = io.BytesIO()
    pq = io.BytesIO()
    ds.df.write_parquet(pq)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("data.parquet", pq.getvalue())
    return buf.getvalue()


def read_manifest(data: bytes) -> dict[str, Any]:
    """Read and validate the manifest without unpacking the data."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if not {"manifest.json", "data.parquet"} <= names:
                raise ProjectError("This is not a project file made by this app.")
            if sum(i.file_size for i in z.infolist()) > MAX_BYTES:
                raise ProjectError("This project file is too large to open safely.")
            manifest = json.loads(z.read("manifest.json"))
    except ProjectError:
        raise
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        raise ProjectError("This file could not be read as a project.", f"Details: {type(exc).__name__}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise ProjectError("This project file has an unsupported version.",
                           "Save it again with the version of the app that made it.")
    return manifest


def load_project(data: bytes):
    """Return (manifest, Dataset). The data is read only from 'data.parquet'."""
    manifest = read_manifest(data)
    folder = Path(tempfile.mkdtemp(prefix="wbproj_"))
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(str(manifest["dataset_name"])).stem) or "project"
    path = folder / f"{stem}.parquet"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            path.write_bytes(z.read("data.parquet"))
        pl.read_parquet(path, n_rows=1)
        return manifest, load_dataset(path)
    except Exception as exc:
        raise ProjectError("The dataset inside the project could not be loaded.",
                           f"Details: {type(exc).__name__}: {exc}") from exc


def apply_project(state: Any, manifest: dict[str, Any], dataset: Any) -> None:
    """Put the dataset into the session and clear everything computed from the old data."""
    for key in DERIVED_KEYS:
        state.pop(key, None)
    state["dataset"] = dataset
    for k, v in (manifest.get("report_notes") or {}).items():
        state[f"rp_note_{k}"] = v