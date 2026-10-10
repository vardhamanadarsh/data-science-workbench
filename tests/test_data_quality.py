import numpy as np
import polars as pl
import pytest

from core.data_quality import assess_data_health


def _clean_frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "x1": rng.normal(size=300),
            "x2": rng.normal(size=300),
            "y": rng.integers(0, 2, size=300),
        }
    )


def test_clean_data_scores_high() -> None:
    report = assess_data_health(_clean_frame(), target="y")
    assert report.score >= 95
    assert report.leakage == []


def test_messy_data_is_flagged() -> None:
    rng = np.random.default_rng(1)
    x = [None] * 30 + list(rng.normal(size=80))
    df = pl.DataFrame({"x": x, "c": [1] * 110})
    report = assess_data_health(df)
    assert report.score < assess_data_health(_clean_frame(), target="y").score
    assert report.duplicate_rows > 0
    assert report.columns.filter(pl.col("constant"))["column"].to_list() == ["c"]
    assert any(f.category == "missing" and f.severity != "ok" for f in report.findings)


def test_leakage_is_detected_as_possible() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    df = pl.DataFrame(
        {"x": rng.normal(size=300), "leak": y * 3.0 + rng.normal(scale=0.01, size=300), "y": y}
    )
    report = assess_data_health(df, target="y")
    assert any(c.column == "leak" and c.strength == "strong" for c in report.leakage)


def test_class_imbalance_is_detected() -> None:
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"x": rng.normal(size=300), "y": [0] * 285 + [1] * 15})
    report = assess_data_health(df, target="y")
    assert report.class_distribution is not None
    assert any(f.category == "imbalance" and f.severity in ("warn", "bad") for f in report.findings)


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        assess_data_health(pl.DataFrame())