"""Feature engineering helpers (UI-free). All features here are stateless:
they learn nothing from the data, so they do not leak between train and test.

Target encoding is NOT here on purpose: it learns from the target, so it must
be done inside cross-validation (it arrives with the model pipeline).
"""

from __future__ import annotations

import polars as pl

from core.preprocessing import OpResult, PreprocessError, require_columns

DATE_PART_EXPRS = {
    "year": lambda e: e.dt.year(),
    "quarter": lambda e: e.dt.quarter(),
    "month": lambda e: e.dt.month(),
    "day": lambda e: e.dt.day(),
    "weekday": lambda e: e.dt.weekday(),          # 1 = Monday ... 7 = Sunday
    "day_of_year": lambda e: e.dt.ordinal_day(),
    "hour": lambda e: e.dt.hour(),
}
ROLLING_STATS = ["mean", "std", "min", "max"]
INTERACTION_OPS = ["product", "ratio", "sum", "difference"]


def date_parts_for(dtype: pl.DataType) -> list[str]:
    parts = ["year", "quarter", "month", "day", "weekday", "day_of_year"]
    if dtype == pl.Datetime:
        parts.append("hour")
    return parts


def _check_numeric(df: pl.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if not df[c].dtype.is_numeric():
            raise PreprocessError(f"'{c}' is not numeric.", "Choose numeric columns.")


def _check_order(df: pl.DataFrame, order_by: str | None) -> None:
    if order_by and not df[order_by].is_sorted():
        raise PreprocessError(
            f"Rows are not sorted by '{order_by}', so lags and rolling windows would be wrong.",
            "Sort the rows first (Clean tab, 'Sort rows').",
        )


def _result(df: pl.DataFrame, exprs: list[pl.Expr], names: list[str], note: str = "") -> OpResult:
    shown = ", ".join(names[:6]) + (f" (+{len(names) - 6} more)" if len(names) > 6 else "")
    return OpResult(df.with_columns(exprs), f"Added {len(names)} feature(s): {shown}.{note}", new_columns=names)


def add_date_parts(df: pl.DataFrame, col: str, parts: list[str]) -> OpResult:
    require_columns(df, [col])
    if df[col].dtype not in (pl.Date, pl.Datetime):
        raise PreprocessError(f"'{col}' is not a date column.", "Convert it first (Clean tab, 'Convert type').")
    if not parts:
        raise PreprocessError("No parts selected.", "Choose at least one part.")
    allowed = date_parts_for(df[col].dtype)
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise PreprocessError(f"Not available for this column: {', '.join(bad)}.", "Choose other parts.")
    names = [f"{col}_{p}" for p in parts]
    exprs = [DATE_PART_EXPRS[p](pl.col(col)).alias(n) for p, n in zip(parts, names)]
    return _result(df, exprs, names)


def add_lags(df: pl.DataFrame, cols: list[str], lags: list[int], order_by: str | None = None) -> OpResult:
    require_columns(df, cols)
    _check_numeric(df, cols)
    if not lags or any(k < 1 for k in lags):
        raise PreprocessError("Choose at least one lag of 1 or more.")
    _check_order(df, order_by)
    names = [f"{c}_lag{k}" for c in cols for k in lags]
    exprs = [pl.col(c).shift(k).alias(f"{c}_lag{k}") for c in cols for k in lags]
    return _result(df, exprs, names, " The first rows of each lag are missing by design.")


def add_rolling(
    df: pl.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: list[str],
    exclude_current: bool = True,
    order_by: str | None = None,
) -> OpResult:
    require_columns(df, cols)
    _check_numeric(df, cols)
    if not windows or not stats:
        raise PreprocessError("Choose at least one window and one statistic.")
    if any(w < 1 for w in windows) or any(s not in ROLLING_STATS for s in stats):
        raise PreprocessError("Invalid window or statistic.")
    if "std" in stats and any(w < 2 for w in windows):
        raise PreprocessError("A rolling standard deviation needs a window of at least 2.")
    _check_order(df, order_by)
    names: list[str] = []
    exprs: list[pl.Expr] = []
    for c in cols:
        base = pl.col(c).cast(pl.Float64)
        if exclude_current:
            base = base.shift(1)
        for w in windows:
            for stat in stats:
                name = f"{c}_roll{w}_{stat}"
                exprs.append(getattr(base, f"rolling_{stat}")(window_size=w).alias(name))
                names.append(name)
    note = " The current row is excluded from each window." if exclude_current else " Each window includes the current row."
    return _result(df, exprs, names, note)


def add_interaction(df: pl.DataFrame, a: str, b: str, op: str) -> OpResult:
    require_columns(df, [a, b])
    _check_numeric(df, [a, b])
    if a == b:
        raise PreprocessError("Choose two different columns.")
    left, right = pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)
    if op == "product":
        name, expr = f"{a}_x_{b}", left * right
    elif op == "ratio":
        name, expr = f"{a}_per_{b}", pl.when(right != 0).then(left / right).otherwise(None)
    elif op == "sum":
        name, expr = f"{a}_plus_{b}", left + right
    elif op == "difference":
        name, expr = f"{a}_minus_{b}", left - right
    else:
        raise PreprocessError(f"Unknown operation '{op}'.")
    note = " Division by zero gives a missing value." if op == "ratio" else ""
    return _result(df, [expr.alias(name)], [name], note)


def leakage_notes(sources: list[str], target: str | None, kind: str) -> list[str]:
    """Warnings to show before adding features. kind: date | lag | rolling | interaction."""
    notes: list[str] = []
    if target and target in sources:
        if kind in ("lag", "rolling"):
            notes.append(
                f"These features are built from the target '{target}'. Past values of the target are fine for "
                "forecasting, but only with a time-based split. In an ordinary model this is possible leakage."
            )
        else:
            notes.append(
                f"Possible leakage: this feature is built from the target '{target}'. "
                "Do not use it as a predictor of the same target."
            )
    if kind in ("lag", "rolling"):
        notes.append("Lag and rolling features depend on row order. Use a time-based train/test split, not a random one.")
    return notes