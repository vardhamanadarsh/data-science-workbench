"""Automated EDA: UI-free summaries that feed the EDA page and report."""

from __future__ import annotations

import html
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger

from config import settings

NUMERIC_SCHEMA: dict[str, Any] = {
    "column": pl.String, "count": pl.Int64, "missing": pl.Int64,
    "mean": pl.Float64, "std": pl.Float64, "min": pl.Float64, "p25": pl.Float64,
    "median": pl.Float64, "p75": pl.Float64, "max": pl.Float64,
    "skew": pl.Float64, "kurtosis": pl.Float64,
}
CATEGORICAL_SCHEMA: dict[str, Any] = {
    "column": pl.String, "unique": pl.Int64, "missing": pl.Int64, "missing_pct": pl.Float64,
    "top": pl.String, "top_count": pl.Int64, "top_pct": pl.Float64,
}
MISSING_SCHEMA: dict[str, Any] = {"column": pl.String, "missing": pl.Int64, "missing_pct": pl.Float64}


@dataclass
class EdaResult:
    overview: dict[str, Any]
    numeric: pl.DataFrame
    categorical: pl.DataFrame
    missing: pl.DataFrame
    stats_rows: int


def _num(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sample(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    return df if df.height <= max_rows else df.sample(n=max_rows, seed=settings.sample_seed)


def numeric_summary(df: pl.DataFrame, max_rows: int = 200_000) -> pl.DataFrame:
    """Descriptive statistics for numeric columns (on at most `max_rows` rows)."""
    sample = _sample(df, max_rows)
    rows: list[dict[str, Any]] = []
    for name in sample.columns:
        s = sample[name]
        if not s.dtype.is_numeric():
            continue
        missing = int(s.null_count())
        if s.dtype.is_float():
            missing += int(s.is_nan().sum() or 0)
        x = s.cast(pl.Float64, strict=False).drop_nulls()
        x = x.filter(x.is_finite())
        row: dict[str, Any] = {"column": name, "count": int(x.len()), "missing": missing}
        if x.len():
            try:
                row.update(
                    mean=_num(x.mean()), std=_num(x.std()), min=_num(x.min()),
                    p25=_num(x.quantile(0.25, interpolation="linear")),
                    median=_num(x.median()),
                    p75=_num(x.quantile(0.75, interpolation="linear")),
                    max=_num(x.max()),
                    skew=_num(x.skew()) if x.len() >= 3 else None,
                    kurtosis=_num(x.kurtosis()) if x.len() >= 4 else None,
                )
            except Exception as exc:
                logger.warning("Numeric summary failed for {}: {}", name, exc)
        rows.append(row)
    return pl.DataFrame(rows, schema=NUMERIC_SCHEMA)


def categorical_summary(df: pl.DataFrame, max_rows: int = 500_000) -> pl.DataFrame:
    """Unique values, most frequent value and missingness for categorical columns."""
    sample = _sample(df, max_rows)
    rows: list[dict[str, Any]] = []
    for name in sample.columns:
        s = sample[name]
        if not (s.dtype == pl.Boolean or s.dtype in (pl.String, pl.Categorical, pl.Enum)):
            continue
        non_null = s.drop_nulls()
        missing = int(s.null_count())
        row: dict[str, Any] = {
            "column": name, "unique": int(non_null.n_unique()) if non_null.len() else 0,
            "missing": missing, "missing_pct": round(missing / max(s.len(), 1) * 100, 2),
            "top": None, "top_count": 0, "top_pct": 0.0,
        }
        if non_null.len():
            counts = non_null.alias("value").value_counts(sort=True)
            top_value, top_count = counts.row(0)[0], int(counts.row(0)[1])
            row.update(top=str(top_value), top_count=top_count,
                       top_pct=round(top_count / non_null.len() * 100, 2))
        rows.append(row)
    return pl.DataFrame(rows, schema=CATEGORICAL_SCHEMA)


def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Missing values per column (uses all rows). Counts NaN in float columns too."""
    rows = []
    height = max(df.height, 1)
    for name in df.columns:
        s = df[name]
        missing = int(s.null_count())
        if s.dtype.is_float():
            missing += int(s.is_nan().sum() or 0)
        rows.append({"column": name, "missing": missing, "missing_pct": round(missing / height * 100, 2)})
    return pl.DataFrame(rows, schema=MISSING_SCHEMA).sort("missing", descending=True)


def correlation_matrix(
    df: pl.DataFrame, method: str = "pearson", max_cols: int = 40, max_rows: int = 100_000
) -> pd.DataFrame | None:
    """Correlation matrix of numeric columns, or None if fewer than two."""
    cols = [c for c, t in df.schema.items() if t.is_numeric()][:max_cols]
    if len(cols) < 2:
        return None
    frame = _sample(df.select(cols), max_rows)
    pdf = frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_pandas()
    pdf = pdf.replace([float("inf"), float("-inf")], float("nan"))
    return pdf.corr(method=method)


def run_eda(df: pl.DataFrame, max_rows: int = 200_000) -> EdaResult:
    """Run the full EDA summary."""
    if df.height == 0 or df.width == 0:
        raise ValueError("The dataset is empty, so there is nothing to analyse.")
    missing = missing_summary(df)
    try:
        duplicates = int(df.height - df.n_unique())
    except Exception as exc:  # e.g. list columns
        logger.warning("Duplicate check skipped: {}", exc)
        duplicates = 0
    overview = {
        "rows": df.height,
        "columns": df.width,
        "missing_cells_pct": round(int(missing["missing"].sum()) / (df.height * df.width) * 100, 2),
        "duplicate_rows": duplicates,
        "dtypes": dict(Counter(str(t) for t in df.dtypes)),
    }
    return EdaResult(
        overview=overview,
        numeric=numeric_summary(df, max_rows),
        categorical=categorical_summary(df),
        missing=missing,
        stats_rows=min(df.height, max_rows),
    )


def build_html_report(name: str, result: EdaResult) -> str:
    """A simple self-contained HTML report (no extra packages needed)."""

    def table(frame: pl.DataFrame) -> str:
        if frame.height == 0:
            return "<p><em>None.</em></p>"
        return frame.to_pandas().round(4).to_html(index=False, border=0, na_rep="")

    o = result.overview
    dtypes = ", ".join(f"{html.escape(k)}: {v}" for k, v in o["dtypes"].items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>EDA report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:2rem;color:#222}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}}
th{{background:#f3f4f6}}td:first-child,th:first-child{{text-align:left}}h2{{margin-top:2rem}}</style></head><body>
<h1>EDA report: {html.escape(name)}</h1>
<p>{o['rows']:,} rows &times; {o['columns']} columns &middot; {o['missing_cells_pct']}% missing cells &middot;
{o['duplicate_rows']:,} duplicate rows<br>Column types: {dtypes}<br>
Distribution statistics use {result.stats_rows:,} rows.</p>
<h2>Numeric summary</h2>{table(result.numeric)}
<h2>Categorical summary</h2>{table(result.categorical)}
<h2>Missing values</h2>{table(result.missing.filter(pl.col("missing") > 0))}
</body></html>"""