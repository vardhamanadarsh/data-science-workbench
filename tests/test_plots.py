import numpy as np
import polars as pl
import pytest

from config import ProcessingMode
from visualization.plot_config import PlotError, PlotSpec, PlotStyle
from visualization.plots import apply_series_colors, build_figure, prepare_data, series_names


def _df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {"v": rng.normal(size=n), "w": rng.normal(size=n),
         "g": rng.choice(["a", "b", "c"], n).tolist(), "k": rng.integers(0, 2, n)}
    )


def _fig(spec: PlotSpec, df: pl.DataFrame | None = None):
    pdf = prepare_data(df if df is not None else _df(), spec, ProcessingMode.FAST)
    return build_figure(pdf, spec, PlotStyle())


def test_histogram_with_color_groups() -> None:
    fig = _fig(PlotSpec(kind="Histogram", x="v", color="k"))
    assert sorted(series_names(fig)) == ["0", "1"]


def test_bar_is_aggregated() -> None:
    spec = PlotSpec(kind="Bar", x="g", y=["v"], agg="mean")
    assert prepare_data(_df(), spec, ProcessingMode.FAST).shape[0] == 3
    assert _fig(spec).data[0].type == "bar"


def test_heatmap_and_qq() -> None:
    assert _fig(PlotSpec(kind="Correlation heatmap", y=["v", "w"])).data[0].type == "heatmap"
    assert len(_fig(PlotSpec(kind="Q-Q plot", x="v")).data) == 2


def test_too_many_groups_raises() -> None:
    df = pl.DataFrame({"v": [float(i) for i in range(100)], "u": [str(i) for i in range(100)]})
    with pytest.raises(PlotError):
        prepare_data(df, PlotSpec(kind="Histogram", x="v", color="u"), ProcessingMode.FAST)


def test_series_color_is_applied() -> None:
    fig = _fig(PlotSpec(kind="Histogram", x="v", color="k"))
    apply_series_colors(fig, {"0": "#112233"})
    trace = next(t for t in fig.data if t.name == "0")
    assert trace.marker.color == "#112233"