import numpy as np
import polars as pl
import pytest

from config import ProcessingMode
from visualization.plot_config import PlotError, PlotSpec, PlotStyle
from visualization.plots import build_figure, prepare_data, series_names


def _df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {"v": rng.normal(size=n), "w": rng.normal(size=n), "x": rng.normal(size=n),
         "g": rng.choice(["a", "b", "c"], n).tolist(), "k": rng.integers(0, 2, n)}
    )


def _fig(spec: PlotSpec):
    pdf = prepare_data(_df(), spec, ProcessingMode.FAST)
    return build_figure(pdf, spec, PlotStyle())


def test_kde_one_curve_per_group() -> None:
    assert sorted(series_names(_fig(PlotSpec(kind="KDE", x="v", color="g")))) == ["a", "b", "c"]


def test_regression_adds_trendline() -> None:
    assert len(_fig(PlotSpec(kind="Regression plot", x="v", y=["w"])).data) == 2


def test_100_percent_stacked_bar() -> None:
    spec = PlotSpec(kind="100% stacked bar", x="g", y=["v"], color="k", agg="mean")
    assert _fig(spec).layout.barnorm == "percent"


def test_lollipop_and_dot_plot() -> None:
    for kind in ("Lollipop", "Dot plot"):
        assert len(_fig(PlotSpec(kind=kind, x="g", y=["v"], agg="mean")).data) == 1


def test_radar_is_scaled_to_unit_range() -> None:
    spec = PlotSpec(kind="Radar", y=["v", "w", "x"], color="g")
    pdf = prepare_data(_df(), spec, ProcessingMode.FAST)
    assert pdf[["v", "w", "x"]].to_numpy().min() >= 0 and pdf[["v", "w", "x"]].to_numpy().max() <= 1
    assert len(build_figure(pdf, spec, PlotStyle()).data) == 3


def test_pair_plot_needs_two_variables() -> None:
    spec = PlotSpec(kind="Pair plot", y=["v"])
    pdf = prepare_data(_df(), spec, ProcessingMode.FAST)
    with pytest.raises(PlotError):
        build_figure(pdf, spec, PlotStyle())