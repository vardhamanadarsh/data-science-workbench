"""Figure construction for Plot Studio (UI-free, testable)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from config import ProcessingMode
from core.data_loader import analysis_sample
from visualization.plot_config import (
    AGG_KINDS,
    BAR_KINDS,
    GROUPED_X_KINDS,
    MAX_BAR_CATEGORIES,
    MAX_COLOR_GROUPS,
    MAX_HEATMAP_COLUMNS,
    MAX_PAIR_COLUMNS,
    NEEDS_Y,
    ORDERED_KINDS,
    PAIR_PLOT_MAX_ROWS,
    SERIES_BAR_KINDS,
    PlotError,
    PlotSpec,
    PlotStyle,
)
from visualization.themes import PALETTES, THEMES

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def _group_columns(spec: PlotSpec) -> list[str]:
    cols = [spec.color] if spec.color else []
    if spec.kind in GROUPED_X_KINDS and spec.x:
        cols.append(spec.x)
    return list(dict.fromkeys(cols))


def _value_col(spec: PlotSpec) -> str:
    return "count" if spec.agg == "count" or not spec.y else spec.y[0]


def _aggregate(frame: pl.DataFrame, spec: PlotSpec) -> pl.DataFrame:
    keys = list(dict.fromkeys([spec.x, *([spec.color] if spec.color else [])]))
    if spec.agg == "count" or not spec.y:
        out = frame.group_by(keys).agg(pl.len().alias("count"))
    else:
        expr = getattr(pl.col(spec.y[0]), spec.agg)()
        out = frame.group_by(keys).agg(expr.alias(spec.y[0]))
    return out.sort(keys)


def _radar_data(frame: pl.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Group means of each variable, min-max scaled to 0..1 using the whole dataset."""
    if not 3 <= len(spec.y) <= 12:
        raise PlotError("A radar chart needs 3 to 12 variables.", "Select more (or fewer) variables.")
    frame = frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in spec.y])
    lo = {c: frame[c].min() for c in spec.y}
    hi = {c: frame[c].max() for c in spec.y}
    bad = [c for c in spec.y if lo[c] is None or hi[c] is None
           or not math.isfinite(lo[c]) or not math.isfinite(hi[c])]
    if bad:
        raise PlotError(f"No finite values in: {', '.join(bad)}.", "Remove these variables.")
    means = [pl.col(c).mean() for c in spec.y]
    table = frame.group_by(spec.color).agg(means).sort(spec.color) if spec.color else frame.select(means)
    table = table.with_columns(
        [
            ((pl.col(c) - lo[c]) / (hi[c] - lo[c])).alias(c) if hi[c] > lo[c] else pl.lit(0.5).alias(c)
            for c in spec.y
        ]
    )
    return table.to_pandas()


def prepare_data(df: pl.DataFrame, spec: PlotSpec, mode: ProcessingMode) -> pd.DataFrame:
    """Select only the needed columns, limit rows by mode, return pandas for Plotly."""
    cols = list(dict.fromkeys(c for c in [spec.x, *spec.y, spec.color, spec.size] if c))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise PlotError(f"Column(s) not found: {', '.join(missing)}.", "Pick the columns again.")
    frame = df.select(cols)

    for col in _group_columns(spec):
        limit = MAX_BAR_CATEGORIES if (spec.kind in AGG_KINDS and col == spec.x) else MAX_COLOR_GROUPS
        n_unique = frame[col].n_unique()
        if n_unique > limit:
            raise PlotError(
                f"'{col}' has {n_unique:,} distinct values, too many to use as a group.",
                f"Choose a column with at most {limit} distinct values.",
            )
    groups = _group_columns(spec)
    if groups:
        frame = frame.with_columns(
            [pl.col(c).cast(pl.String).fill_null("(missing)") for c in groups]
        )

    if spec.kind == "Radar":
        return _radar_data(frame, spec)
    if spec.kind in AGG_KINDS:
        return _aggregate(frame, spec).to_pandas()
    if spec.kind in ORDERED_KINDS and spec.x:
        frame = analysis_sample(frame.drop_nulls(spec.x).sort(spec.x), mode, preserve_order=True)
    else:
        frame = analysis_sample(frame, mode)
    if spec.kind == "Pair plot" and frame.height > PAIR_PLOT_MAX_ROWS:
        frame = frame.sample(n=PAIR_PLOT_MAX_ROWS, seed=42)  # many small panels: keep it fast
    return frame.to_pandas()


# --------------------------------------------------------------------------- #
# Special figures
# --------------------------------------------------------------------------- #

def _qq(values: pd.Series, palette: list[str]) -> go.Figure:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 5:
        raise PlotError("A Q-Q plot needs at least 5 values.", "Pick another column.")
    (osm, osr), (slope, intercept, _) = stats.probplot(x.to_numpy(), dist="norm")
    fig = go.Figure()
    fig.add_scatter(x=osm, y=osr, mode="markers", name="Sample", marker=dict(color=palette[0]))
    ends = np.array([osm.min(), osm.max()])
    fig.add_scatter(
        x=ends, y=slope * ends + intercept, mode="lines", name="Normal reference",
        line=dict(color=palette[1 % len(palette)]),
    )
    fig.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
    return fig


def _kde(pdf: pd.DataFrame, spec: PlotSpec, palette: list[str]) -> go.Figure:
    """Gaussian kernel density curves (one per group). Plotly has no native KDE."""
    parts = list(pdf.groupby(spec.color)) if spec.color else [(spec.x, pdf)]
    fig = go.Figure()
    skipped = 0
    for i, (name, part) in enumerate(parts):
        x = pd.to_numeric(part[spec.x], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        values = x.to_numpy()
        if len(values) < 5 or np.ptp(values) == 0:
            skipped += 1
            continue
        if len(values) > 20_000:
            values = np.random.default_rng(42).choice(values, 20_000, replace=False)
        grid = np.linspace(values.min(), values.max(), 300)
        density = stats.gaussian_kde(values)(grid)
        fig.add_scatter(
            x=grid, y=density, mode="lines", fill="tozeroy", name=str(name),
            line=dict(color=palette[i % len(palette)]),
        )
    if not fig.data:
        raise PlotError("A density curve needs at least 5 distinct values per group.", "Pick another column.")
    fig.update_layout(xaxis_title=spec.x, yaxis_title="Density")
    return fig


def _heatmap(pdf: pd.DataFrame, spec: PlotSpec) -> go.Figure:
    cols = spec.y[:MAX_HEATMAP_COLUMNS]
    if len(cols) < 2:
        raise PlotError("Select at least two numeric columns.", "Add more variables.")
    data = pdf[cols].replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = data.corr(method=spec.corr_method)
    return px.imshow(
        corr, text_auto=".2f" if len(cols) <= 15 else False,
        zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
    )


def _radar(pdf: pd.DataFrame, spec: PlotSpec, palette: list[str]) -> go.Figure:
    axes = list(spec.y)
    closed = axes + [axes[0]]
    fig = go.Figure()
    for i, (_, row) in enumerate(pdf.iterrows()):
        values = [row[c] for c in axes]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]], theta=closed, fill="toself",
                name=str(row[spec.color]) if spec.color else "Average",
                line=dict(color=palette[i % len(palette)]),
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)))
    return fig


def _lollipop(pdf: pd.DataFrame, spec: PlotSpec, palette: list[str]) -> go.Figure:
    ycol = _value_col(spec)
    d = pdf.sort_values(ycol, ascending=False)
    vals = d[ycol].to_numpy(dtype=float)
    fig = go.Figure(
        go.Scatter(
            x=d[spec.x], y=vals, mode="markers", name=ycol,
            marker=dict(color=palette[0]),
            error_y=dict(type="data", symmetric=False, array=np.maximum(-vals, 0),
                         arrayminus=np.maximum(vals, 0), width=0, color=palette[0], visible=True),
        )
    )
    fig.update_layout(xaxis_title=spec.x, yaxis_title=ycol)
    return fig


# --------------------------------------------------------------------------- #
# Figure building
# --------------------------------------------------------------------------- #

def build_figure(pdf: pd.DataFrame, spec: PlotSpec, style: PlotStyle) -> go.Figure:
    """Create a styled Plotly figure. Raises PlotError with a helpful message."""
    palette = PALETTES[style.palette]
    kind = spec.kind
    if kind in NEEDS_Y and not spec.y:
        raise PlotError("Select at least one value column.", "Choose a numeric column for Y.")
    if kind in SERIES_BAR_KINDS and not spec.color:
        raise PlotError(f"A {kind.lower()} needs a series column.", "Choose a 'Series' column.")
    if kind == "Pair plot" and not 2 <= len(spec.y) <= MAX_PAIR_COLUMNS:
        raise PlotError(f"A pair plot needs 2 to {MAX_PAIR_COLUMNS} variables.", "Change the selection.")
    common: dict[str, Any] = {"color_discrete_sequence": palette}

    try:
        if kind == "Histogram":
            fig = px.histogram(pdf, x=spec.x, color=spec.color, nbins=spec.bins,
                               barmode="overlay", **common)
        elif kind == "KDE":
            fig = _kde(pdf, spec, palette)
        elif kind in ("Box plot", "Violin plot", "Strip plot"):
            fn = {"Box plot": px.box, "Violin plot": px.violin, "Strip plot": px.strip}[kind]
            fig = fn(pdf, x=spec.x, y=spec.y[0], color=spec.color or spec.x, **common)
            fig.update_layout(boxmode="overlay", violinmode="overlay")
        elif kind == "ECDF":
            fig = px.ecdf(pdf, x=spec.x, color=spec.color, **common)
        elif kind == "Q-Q plot":
            fig = _qq(pdf[spec.x], palette)
        elif kind == "Scatter":
            fig = px.scatter(pdf, x=spec.x, y=spec.y[0], color=spec.color, **common)
        elif kind == "Bubble":
            data = pdf[pdf[spec.size] > 0]
            if data.empty:
                raise PlotError("The size column has no positive values.", "Choose another size column.")
            fig = px.scatter(data, x=spec.x, y=spec.y[0], size=spec.size, color=spec.color,
                             size_max=40, **common)
        elif kind == "Regression plot":
            data = pdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[spec.x, spec.y[0]])
            if len(data) < 3:
                raise PlotError("A regression plot needs at least 3 complete rows.", "Pick other columns.")
            fig = px.scatter(data, x=spec.x, y=spec.y[0], color=spec.color, trendline="ols", **common)
        elif kind == "Joint plot":
            fig = px.scatter(pdf, x=spec.x, y=spec.y[0], color=spec.color,
                             marginal_x="histogram", marginal_y="histogram", **common)
        elif kind == "Pair plot":
            fig = px.scatter_matrix(pdf, dimensions=spec.y, color=spec.color, **common)
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.update_layout(height=700)
        elif kind in BAR_KINDS:
            barmode = "stack" if kind in ("Stacked bar", "100% stacked bar") else "group"
            fig = px.bar(pdf, x=spec.x, y=_value_col(spec), color=spec.color, barmode=barmode, **common)
            if kind == "100% stacked bar":
                fig.update_layout(barnorm="percent")
                fig.update_yaxes(title_text="Percent")
        elif kind == "Lollipop":
            fig = _lollipop(pdf, spec, palette)
        elif kind == "Dot plot":
            ycol = _value_col(spec)
            d = pdf.sort_values(ycol)
            fig = px.scatter(d, x=ycol, y=spec.x, **common)
            fig.update_yaxes(categoryorder="array", categoryarray=d[spec.x].tolist())
        elif kind == "Radar":
            fig = _radar(pdf, spec, palette)
        elif kind == "Line":
            fig = px.line(pdf, x=spec.x, y=spec.y, **common)
        elif kind == "Area":
            fig = px.area(pdf, x=spec.x, y=spec.y, **common)
        elif kind == "Correlation heatmap":
            fig = _heatmap(pdf, spec)
        else:
            raise PlotError(f"Unknown plot type '{kind}'.")
    except PlotError:
        raise
    except Exception as exc:
        raise PlotError("The plot could not be created.", f"Try different columns. Details: {exc}") from exc

    label = spec.y[0] if spec.y else (spec.x or "series")
    for trace in fig.data:
        if trace.type != "heatmap" and not trace.name:
            trace.name = label
    if kind in ("Line", "Area") and len(spec.y) == 1:
        fig.update_yaxes(title_text=spec.y[0])
    apply_style(fig, spec, style)
    return fig


def apply_style(fig: go.Figure, spec: PlotSpec, style: PlotStyle) -> None:
    """Apply theme, labels, grid, legend, fonts, line width, marker size and opacity."""
    fig.update_layout(
        template=THEMES[style.theme],
        font=dict(size=style.font_size),
        showlegend=style.show_legend,
    )
    if style.title:
        fig.update_layout(title_text=style.title)
    if style.x_label:
        fig.update_xaxes(title_text=style.x_label)
    if style.y_label:
        fig.update_yaxes(title_text=style.y_label)
    fig.update_xaxes(showgrid=style.show_grid)
    fig.update_yaxes(showgrid=style.show_grid)

    for trace in fig.data:
        if trace.type == "heatmap":
            continue
        if not style.show_legend:
            trace.update(showlegend=False)
        if trace.type in ("scatter", "scattergl"):
            mode = trace.mode or ""
            if "lines" in mode:
                trace.update(line=dict(width=style.line_width))
            if "markers" in mode and spec.kind != "Bubble":
                trace.update(marker=dict(size=style.marker_size))
            trace.update(opacity=style.opacity)
        elif trace.type == "scatterpolar":
            trace.update(line=dict(width=style.line_width), opacity=style.opacity)
        elif trace.type == "splom":
            trace.update(marker=dict(size=max(2, style.marker_size - 2)), opacity=style.opacity)
        elif trace.type in ("histogram", "bar", "box", "violin"):
            trace.update(opacity=style.opacity)
            if spec.kind == "Strip plot" and trace.type == "box":
                trace.update(marker=dict(size=style.marker_size))


def series_names(fig: go.Figure) -> list[str]:
    """Unique series names (heatmaps have none)."""
    names: list[str] = []
    for trace in fig.data:
        if trace.type != "heatmap" and trace.name and trace.name not in names:
            names.append(trace.name)
    return names


def apply_series_colors(fig: go.Figure, colors: dict[str, str]) -> None:
    """Give each named series its own color (HEX)."""
    for trace in fig.data:
        color = colors.get(trace.name)
        if not color or trace.type == "heatmap":
            continue
        trace.update(marker=dict(color=color))
        if trace.type in ("scatter", "scattergl", "scatterpolar", "box", "violin"):
            trace.update(line=dict(color=color))
        if trace.type in ("scatter", "scattergl", "scatterpolar") and trace.fill:
            trace.update(fillcolor=color)
        error = getattr(trace, "error_y", None)
        if error is not None and error.visible:
            trace.update(error_y=dict(color=color))


def finalize(fig: go.Figure, spec: PlotSpec, live: bool) -> None:
    """Mode-specific touches. Live mode adds a time-range slider to line/area plots."""
    if live and spec.kind in ORDERED_KINDS:
        fig.update_xaxes(rangeslider_visible=True)


def chart_config(live: bool, filename: str = "plot") -> dict[str, Any]:
    """Plotly.js config. Normal = static (no hover/zoom); Live = fully interactive."""
    config: dict[str, Any] = {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
    }
    if live:
        config["scrollZoom"] = True
    else:
        config["staticPlot"] = True
    return config


def export_figure(fig: go.Figure, fmt: str) -> bytes:
    """Export to png / svg / pdf (needs Kaleido) or a self-contained html file."""
    if fmt == "html":
        return fig.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")
    try:
        return fig.to_image(format=fmt, scale=2)
    except Exception as exc:
        raise PlotError(
            f"Static export to {fmt.upper()} failed.",
            "This needs the Kaleido package (and, for Kaleido 1.x, Chrome). "
            "HTML export always works. See the notes in the README. "
            f"Details: {exc}",
        ) from exc