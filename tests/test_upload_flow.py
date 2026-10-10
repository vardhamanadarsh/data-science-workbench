import io
from pathlib import Path

import polars as pl
import pytest

from config import settings
from core.data_loader import DataLoadError, load_dataset, save_upload
from core.data_quality import assess_data_health
from core.samples import ensure_samples


def _csv_bytes(n: int = 150) -> bytes:
    lines = ["feature,other,y"] + [f"{i * 0.5},{(i * 7) % 13},{i % 2}" for i in range(n)]
    return "\n".join(lines).encode()


def test_upload_to_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    path = save_upload("my data (1).csv", io.BytesIO(_csv_bytes()))
    assert path.exists()
    assert " " not in path.name and "(" not in path.name

    ds = load_dataset(path)
    assert ds.df.shape == (150, 3)

    report = assess_data_health(ds.df, target="y")
    assert 0 <= report.score <= 100


def test_same_upload_maps_to_same_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    a = save_upload("a.csv", io.BytesIO(_csv_bytes()))
    b = save_upload("a.csv", io.BytesIO(_csv_bytes()))
    assert a == b


def test_bad_extension_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    with pytest.raises(DataLoadError):
        save_upload("script.exe", io.BytesIO(b"MZ"))


def test_demo_datasets_load_and_assess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "samples_dir", tmp_path)
    paths = ensure_samples()
    assert len(paths) == 2
    for path in paths.values():
        ds = load_dataset(path)
        assert ds.n_rows_loaded > 0
        assess_data_health(ds.df)

    churn = load_dataset(next(p for p in paths.values() if "churn" in p.name))
    report = assess_data_health(churn.df, target="churned")
    assert isinstance(churn.df["churned"], pl.Series)
    assert report.class_distribution is not None