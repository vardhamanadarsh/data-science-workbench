import numpy as np
import polars as pl
import pytest

from core.data_loader import load_dataset
from core.project import ProjectError, apply_project, load_project, read_manifest, save_project_bytes


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "demo.csv"
    pl.DataFrame({"a": np.arange(50, dtype=float), "b": ["x", "y"] * 25}).write_csv(path)
    return {"dataset": load_dataset(path), "rp_note_general": "my note"}


def test_roundtrip(state) -> None:
    data = save_project_bytes(state)
    manifest, ds = load_project(data)
    assert manifest["rows"] == 50 and manifest["report_notes"] == {"general": "my note"}
    assert ds.df.equals(state["dataset"].df)


def test_apply_clears_derived(state) -> None:
    manifest, ds = load_project(save_project_bytes(state))
    session = {"ml_run": object(), "health": {}, "dataset": None}
    apply_project(session, manifest, ds)
    assert "ml_run" not in session and "health" not in session
    assert session["dataset"] is ds and session["rp_note_general"] == "my note"


def test_errors() -> None:
    with pytest.raises(ProjectError):
        save_project_bytes({})
    with pytest.raises(ProjectError):
        read_manifest(b"not a zip")