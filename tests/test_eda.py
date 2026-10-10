import numpy as np
import polars as pl
import pytest

from core.eda import build_html_report, correlation_matrix, run_eda


def _df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    return pl.DataFrame(
        {"a": a, "b": a * 2 + rng.normal(scale=0.1, size=100),
         "c": (["x"] * 70) + (["y"] * 30)}
    )


def test_eda_summaries() -> None:
    result = run_eda(_df())
    assert result.numeric.height == 2
    assert result.categorical["top"].to_list() == ["x"]
    assert result.overview["rows"] == 100
    corr = correlation_matrix(_df())
    assert corr is not None and corr.shape == (2, 2) and corr.loc["a", "b"] > 0.99
    assert "EDA report" in build_html_report("demo.csv", result)


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        run_eda(pl.DataFrame())