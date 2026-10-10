"""Dataset-level cleaning operations (UI-free).

Every function takes a Polars DataFrame and returns an OpResult. Nothing is
changed in place, and nothing is removed unless the caller asks for it.

Design rule: only operations that learn nothing from the data live here (or are
clearly flagged). Scaling and encoding belong inside the model pipeline, where
they are fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

TYPE_TARGETS: list[str] = [
    "Number (decimal)", "Integer", "Text", "Category", "Date", "Date and time", "True/False",
]
_TRUE_WORDS = ["true", "yes", "y", "t", "1"]
_FALSE_WORDS = ["false", "no", "n", "f", "0"]


class PreprocessError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


@dataclass
class OpResult:
    df: pl.DataFrame
    message: str
    changed: bool = True
    new_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def require_columns(df: pl.DataFrame, cols: list[str]) -> None:
    if not cols:
        raise PreprocessError("No columns selected.", "Choose at least one column.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise PreprocessError(f"Column(s) not found: {', '.join(missing)}.", "Pick the columns again.")


def missing_count(s: pl.Series) -> int:
    """Nulls plus NaN (for float columns)."""
    n = int(s.null_count())
    if s.dtype.is_float():
        n += int(s.is_nan().sum() or 0)
    return n


def _names(cols: list[str], limit: int = 5) -> str:
    shown = ", ".join(cols[:limit])
    return shown + (f" (+{len(cols) - limit} more)" if len(cols) > limit else "")


def _unchanged(df: pl.DataFrame, message: str) -> OpResult:
    return OpResult(df, message, changed=False)


def _as_float(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False).fill_nan(None)


# --------------------------------------------------------------------------- #
# Columns and rows
# --------------------------------------------------------------------------- #

def drop_columns(df: pl.DataFrame, cols: list[str]) -> OpResult:
    require_columns(df, cols)
    if len(cols) >= df.width:
        raise PreprocessError("That would remove every column.", "Keep at least one column.")
    return OpResult(df.drop(cols), f"Dropped {len(cols)} column(s): {_names(cols)}.")


def count_duplicates(df: pl.DataFrame, subset: list[str] | None = None) -> int:
    try:
        return int(df.height - df.n_unique(subset=subset or None))
    except Exception:  # e.g. list columns
        return 0


def drop_duplicate_rows(df: pl.DataFrame, subset: list[str] | None = None) -> OpResult:
    if subset:
        require_columns(df, subset)
    out = df.unique(subset=subset or None, keep="first", maintain_order=True)
    removed = df.height - out.height
    if removed == 0:
        return _unchanged(df, "No duplicate rows found.")
    return OpResult(out, f"Removed {removed:,} duplicate row(s), keeping the first occurrence.")


def drop_rows_with_missing(df: pl.DataFrame, cols: list[str]) -> OpResult:
    require_columns(df, cols)
    nan_to_null = [pl.col(c).fill_nan(None) for c in cols if df[c].dtype.is_float()]
    tmp = df.with_columns(nan_to_null) if nan_to_null else df
    out = tmp.drop_nulls(subset=cols)
    removed = df.height - out.height
    if removed == 0:
        return _unchanged(df, "No rows have missing values in the selected columns.")
    if out.height == 0:
        raise PreprocessError(
            "That would remove every row.", "Choose fewer columns, or fill the missing values instead."
        )
    return OpResult(out, f"Removed {removed:,} row(s) with missing values in: {_names(cols)}.")


def drop_high_missing_columns(df: pl.DataFrame, threshold_pct: float) -> OpResult:
    if not 0 < threshold_pct <= 100:
        raise PreprocessError("The threshold must be between 0 and 100.")
    height = max(df.height, 1)
    drop = [c for c in df.columns if missing_count(df[c]) / height * 100 >= threshold_pct]
    if not drop:
        return _unchanged(df, f"No column is at least {threshold_pct:.0f}% empty.")
    result = drop_columns(df, drop)
    result.message = f"Dropped {len(drop)} column(s) that are at least {threshold_pct:.0f}% empty: {_names(drop)}."
    return result


def sort_rows(df: pl.DataFrame, col: str, descending: bool = False) -> OpResult:
    require_columns(df, [col])
    return OpResult(
        df.sort(col, descending=descending, nulls_last=True),
        f"Sorted rows by '{col}' ({'descending' if descending else 'ascending'}).",
    )


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #

def fill_missing(
    df: pl.DataFrame, cols: list[str], method: str, constant: str | None = None
) -> OpResult:
    """Fill missing values. method: mean | median | mode | constant.

    Mean, median and mode are computed from ALL rows, so prefer imputing inside
    the model pipeline when you plan to train a model.
    """
    require_columns(df, cols)
    exprs: list[pl.Expr] = []
    total = 0
    for c in cols:
        s = df[c]
        n_missing = missing_count(s)
        if n_missing == 0:
            continue
        is_float = s.dtype.is_float()
        base = pl.col(c).fill_nan(None) if is_float else pl.col(c)
        clean = s.fill_nan(None) if is_float else s

        if method in ("mean", "median"):
            if not s.dtype.is_numeric():
                raise PreprocessError(f"'{c}' is not numeric, so a {method} cannot be used.", "Use mode or a constant.")
            value = clean.mean() if method == "mean" else clean.median()
            if value is None:
                raise PreprocessError(f"'{c}' has no values to compute a {method} from.", "Use a constant instead.")
            expr = base.cast(pl.Float64).fill_null(float(value))
        elif method == "mode":
            non_null = clean.drop_nulls()
            if non_null.len() == 0:
                raise PreprocessError(f"'{c}' has no values to compute a mode from.", "Use a constant instead.")
            expr = base.fill_null(non_null.mode().sort()[0])
        elif method == "constant":
            if constant is None or str(constant).strip() == "":
                raise PreprocessError("Enter a constant value.")
            if s.dtype.is_numeric():
                try:
                    number = float(constant)
                except ValueError:
                    raise PreprocessError(f"'{constant}' is not a number, but '{c}' is numeric.", "Enter a number.") from None
                if s.dtype.is_integer() and number.is_integer():
                    expr = base.fill_null(int(number))
                else:
                    expr = base.cast(pl.Float64).fill_null(number)
            elif s.dtype in (pl.String, pl.Categorical, pl.Enum):
                expr = base.fill_null(str(constant))
            else:
                raise PreprocessError(f"A constant cannot be used for '{c}' ({s.dtype}).", "Use mode instead.")
        else:
            raise PreprocessError(f"Unknown method '{method}'.")
        exprs.append(expr.alias(c))
        total += n_missing

    if not exprs:
        return _unchanged(df, "No missing values in the selected columns.")
    return OpResult(
        df.with_columns(exprs), f"Filled {total:,} missing value(s) in {len(exprs)} column(s) using {method}."
    )


# --------------------------------------------------------------------------- #
# Type conversion
# --------------------------------------------------------------------------- #

def convert_type(df: pl.DataFrame, col: str, target: str) -> OpResult:
    """Convert one column. Values that cannot be converted become missing (and are reported)."""
    require_columns(df, [col])
    s = df[col]
    dtype = s.dtype
    before = missing_count(s)
    base = pl.col(col)
    text = base.str.strip_chars() if dtype == pl.String else base

    if target == "Number (decimal)":
        expr = text.cast(pl.Float64, strict=False)
    elif target == "Integer":
        if dtype.is_integer():
            expr = base.cast(pl.Int64)
        else:
            expr = text.cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    elif target == "Text":
        expr = base.cast(pl.String)
    elif target == "Category":
        expr = base.cast(pl.String).cast(pl.Categorical)
    elif target == "Date":
        if dtype == pl.Date:
            return _unchanged(df, f"'{col}' is already a date.")
        if dtype == pl.Datetime:
            expr = base.dt.date()
        elif dtype == pl.String:
            expr = text.str.to_date(strict=False)
        else:
            raise PreprocessError(f"'{col}' ({dtype}) cannot be read as a date.", "Convert it to Text first.")
    elif target == "Date and time":
        if dtype == pl.Datetime:
            return _unchanged(df, f"'{col}' is already a date and time.")
        if dtype == pl.Date:
            expr = base.cast(pl.Datetime)
        elif dtype == pl.String:
            expr = text.str.to_datetime(strict=False)
        else:
            raise PreprocessError(f"'{col}' ({dtype}) cannot be read as a date and time.", "Convert it to Text first.")
    elif target == "True/False":
        if dtype == pl.Boolean:
            return _unchanged(df, f"'{col}' is already True/False.")
        if dtype == pl.String:
            low = text.str.to_lowercase()
            expr = (
                pl.when(low.is_in(_TRUE_WORDS)).then(True)
                .when(low.is_in(_FALSE_WORDS)).then(False)
                .otherwise(None)
            )
        elif dtype.is_numeric():
            expr = base != 0
        else:
            raise PreprocessError(f"'{col}' ({dtype}) cannot be converted to True/False.")
    else:
        raise PreprocessError(f"Unknown type '{target}'.")

    try:
        out = df.with_columns(expr.alias(col))
    except Exception as exc:
        raise PreprocessError(f"'{col}' could not be converted to {target}.", f"Details: {exc}") from exc

    lost = missing_count(out[col]) - before
    message = f"Converted '{col}' from {dtype} to {out[col].dtype}."
    if lost > 0:
        message += f" {lost:,} value(s) could not be converted and are now missing."
    return OpResult(out, message)


# --------------------------------------------------------------------------- #
# Outliers (always user-approved)
# --------------------------------------------------------------------------- #

def iqr_bounds(s: pl.Series, factor: float) -> tuple[float, float] | None:
    x = s.cast(pl.Float64, strict=False).drop_nulls()
    x = x.filter(x.is_finite())
    if x.len() < 4:
        return None
    q1 = x.quantile(0.25, interpolation="linear")
    q3 = x.quantile(0.75, interpolation="linear")
    if q1 is None or q3 is None or q3 <= q1:
        return None
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def outlier_table(df: pl.DataFrame, cols: list[str], factor: float = 1.5) -> pl.DataFrame:
    """Preview: IQR limits and outlier counts per column. Changes nothing."""
    rows = []
    for c in cols:
        s = df[c]
        if not s.dtype.is_numeric():
            continue
        bounds = iqr_bounds(s, factor)
        if bounds is None:
            rows.append({"column": c, "lower": None, "upper": None, "outliers": 0})
            continue
        x = s.cast(pl.Float64, strict=False).fill_nan(None)
        n = int(((x < bounds[0]) | (x > bounds[1])).sum() or 0)
        rows.append({"column": c, "lower": bounds[0], "upper": bounds[1], "outliers": n})
    return pl.DataFrame(
        rows, schema={"column": pl.String, "lower": pl.Float64, "upper": pl.Float64, "outliers": pl.Int64}
    )


def handle_outliers(
    df: pl.DataFrame, cols: list[str], factor: float = 1.5, method: str = "clip"
) -> OpResult:
    """method: 'clip' (cap at the IQR limits; columns become decimal) or 'remove' (drop rows)."""
    require_columns(df, cols)
    bounds = {c: b for c in cols if df[c].dtype.is_numeric() and (b := iqr_bounds(df[c], factor))}
    table = outlier_table(df, list(bounds), factor)
    total = int(table["outliers"].sum() or 0) if table.height else 0
    if total == 0:
        return _unchanged(df, "No outliers found in the selected columns.")

    if method == "clip":
        out = df.with_columns([_as_float(c).clip(lo, hi).alias(c) for c, (lo, hi) in bounds.items()])
        return OpResult(out, f"Clipped {total:,} outlying value(s) in {len(bounds)} column(s) to the IQR limits (factor {factor}).")
    if method == "remove":
        conds = [((_as_float(c) < lo) | (_as_float(c) > hi)).fill_null(False) for c, (lo, hi) in bounds.items()]
        out = df.filter(~pl.any_horizontal(conds))
        if out.height == 0:
            raise PreprocessError("That would remove every row.", "Raise the IQR factor or use clipping.")
        return OpResult(out, f"Removed {df.height - out.height:,} row(s) containing outliers (factor {factor}).")
    raise PreprocessError(f"Unknown method '{method}'.")


# --------------------------------------------------------------------------- #
# Stateless numeric transforms (learn nothing from the data)
# --------------------------------------------------------------------------- #

def transform_numeric(df: pl.DataFrame, cols: list[str], how: str) -> OpResult:
    """Add log1p / sqrt versions of numeric columns as NEW columns."""
    require_columns(df, cols)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for c in cols:
        s = df[c]
        if not s.dtype.is_numeric():
            raise PreprocessError(f"'{c}' is not numeric.", "Choose numeric columns.")
        x = s.cast(pl.Float64, strict=False).fill_nan(None).drop_nulls()
        x = x.filter(x.is_finite())
        low = x.min() if x.len() else None
        if how == "log1p":
            if low is not None and low <= -1:
                raise PreprocessError(f"'{c}' has values of -1 or less, so log(1 + x) is undefined.", "Shift the column or skip it.")
            expr = _as_float(c).log1p()
        elif how == "sqrt":
            if low is not None and low < 0:
                raise PreprocessError(f"'{c}' has negative values, so the square root is undefined.", "Shift the column or skip it.")
            expr = _as_float(c).sqrt()
        else:
            raise PreprocessError(f"Unknown transform '{how}'.")
        name = f"{c}_{how}"
        exprs.append(expr.alias(name))
        names.append(name)
    return OpResult(df.with_columns(exprs), f"Added {len(names)} column(s): {_names(names)}.", new_columns=names)