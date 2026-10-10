"""Dataset comparison and drift detection (UI-free).

PSI is the main drift signal. The KS test is reported too, but with large
samples it flags tiny differences, so it does not decide the drift level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from config import settings

PSI_MODERATE = 0.10
PSI_HIGH = 0.25
MAX_ROWS = 200_000
EPS = 1e-4
MISSING_SHIFT_PP = 5.0
MAX_CATEGORIES = 50


class CompareError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


@dataclass
class CompareReport:
    ref_rows: int
    new_rows: int
    ref_cols: int
    new_cols: int
    sampled: bool
    schema: pd.DataFrame
    drift: pd.DataFrame
    missing: pd.DataFrame
    summary: pd.DataFrame
    categories: pd.DataFrame
    findings: list[tuple[str, str]] = field(default_factory=list)   # (ok | info | warn, text)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def kind_of(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean or dtype in (pl.String, pl.Categorical, pl.Enum):
        return "categorical"
    if dtype.is_numeric():
        return "numeric"
    if dtype in (pl.Date, pl.Datetime):
        return "datetime"
    return "other"


def _names(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _sample(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    return df if df.height <= MAX_ROWS else df.sample(n=MAX_ROWS, seed=seed)


def _num_array(s: pl.Series) -> np.ndarray:
    x = s.cast(pl.Float64, strict=False).drop_nulls()
    return x.filter(x.is_finite()).to_numpy()


def _cat_values(s: pl.Series) -> pd.Series:
    return s.drop_nulls().cast(pl.String).to_pandas()


def _missing_pct(s: pl.Series) -> float:
    n = int(s.null_count())
    if s.dtype.is_float():
        n += int(s.is_nan().sum() or 0)
    return n / max(s.len(), 1) * 100


def level_from_psi(psi: float | None) -> str:
    if psi is None:
        return "not tested"
    if psi < PSI_MODERATE:
        return "none"
    return "moderate" if psi < PSI_HIGH else "high"


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #

def psi_numeric(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float | None:
    """PSI with bins taken from the reference quantiles. None if the reference is nearly constant."""
    if len(ref) < 20 or len(new) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    inner = edges[1:-1]
    k = len(edges) - 1
    p = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=k) / len(ref)
    q = np.bincount(np.searchsorted(inner, new, side="right"), minlength=k) / len(new)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(ref: pd.Series, new: pd.Series) -> float | None:
    if len(ref) < 30 or len(new) < 30:
        return None
    both = pd.concat([ref.value_counts().rename("r"), new.value_counts().rename("n")], axis=1).fillna(0.0)
    both = both.sort_values("r", ascending=False)
    if len(both) > MAX_CATEGORIES:
        tail = both.iloc[MAX_CATEGORIES:].sum()
        both = pd.concat([both.iloc[:MAX_CATEGORIES],
                          pd.DataFrame([tail.to_numpy()], columns=both.columns, index=["(other)"])])
    p = np.clip(both["r"].to_numpy() / both["r"].sum(), EPS, None)
    q = np.clip(both["n"].to_numpy() / both["n"].sum(), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def schema_diff(ref: pl.DataFrame, new: pl.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for c in ref.columns:
        if c in new.columns:
            a, b = str(ref.schema[c]), str(new.schema[c])
            rows.append({"column": c, "reference type": a, "new type": b,
                         "status": "same" if a == b else "type changed"})
        else:
            rows.append({"column": c, "reference type": str(ref.schema[c]), "new type": "",
                         "status": "only in reference"})
    for c in new.columns:
        if c not in ref.columns:
            rows.append({"column": c, "reference type": "", "new type": str(new.schema[c]), "status": "only in new"})
    return pd.DataFrame(rows)


def _missing_table(ref: pl.DataFrame, new: pl.DataFrame, common: list[str]) -> pd.DataFrame:
    rows = []
    for c in common:
        a, b = _missing_pct(ref[c]), _missing_pct(new[c])
        rows.append({"column": c, "reference missing %": round(a, 2), "new missing %": round(b, 2),
                     "change (pp)": round(b - a, 2)})
    t = pd.DataFrame(rows, columns=["column", "reference missing %", "new missing %", "change (pp)"])
    return t.reindex(t["change (pp)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _summary_table(ref: pl.DataFrame, new: pl.DataFrame, common: list[str]) -> pd.DataFrame:
    rows = []
    for c in common:
        if kind_of(ref.schema[c]) != "numeric" or kind_of(new.schema[c]) != "numeric":
            continue
        a, b = _num_array(ref[c]), _num_array(new[c])
        if len(a) == 0 or len(b) == 0:
            continue
        change = (b.mean() - a.mean()) / abs(a.mean()) * 100 if a.mean() != 0 else None
        rows.append({"column": c, "mean (ref)": a.mean(), "mean (new)": b.mean(), "mean change %": change,
                     "std (ref)": a.std(), "std (new)": b.std(), "median (ref)": float(np.median(a)),
                     "median (new)": float(np.median(b)), "min (new)": b.min(), "max (new)": b.max()})
    return pd.DataFrame(rows)


def category_changes(ref: pl.DataFrame, new: pl.DataFrame, common: list[str]) -> pd.DataFrame:
    cols = ["column", "new categories", "removed categories"]
    rows = []
    for c in common:
        if kind_of(ref.schema[c]) != "categorical" or kind_of(new.schema[c]) != "categorical":
            continue
        a, b = _cat_values(ref[c]), _cat_values(new[c])
        if len(a) == 0 or len(b) == 0 or a.nunique() > 200 or b.nunique() > 200:
            continue
        ra, nb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        added = [k for k in nb.index if k not in ra.index]
        gone = [k for k in ra.index if k not in nb.index]
        if added or gone:
            fmt = lambda keys, share: ", ".join(f"{k} ({share[k]:.1%})" for k in keys[:5]) + (" ..." if len(keys) > 5 else "")
            rows.append({"column": c, "new categories": fmt(added, nb), "removed categories": fmt(gone, ra)})
    return pd.DataFrame(rows, columns=cols)


def _drift_row(name: str, r: pl.Series, n: pl.Series, bins: int) -> dict[str, Any]:
    kr, kn = kind_of(r.dtype), kind_of(n.dtype)
    row: dict[str, Any] = {"column": name, "type": kr if kr == kn else f"{kr} → {kn}", "PSI": None,
                           "KS statistic": None, "KS p-value": None, "level": "not tested", "detail": ""}
    if kr != kn:
        row.update(level="high", detail="The column type changed.")
        return row
    if kr == "numeric":
        a, b = _num_array(r), _num_array(n)
        if len(a) < 30 or len(b) < 30:
            row["detail"] = "Too few values to test."
            return row
        psi = psi_numeric(a, b, bins)
        ks = stats.ks_2samp(a, b)
        row.update({"PSI": psi, "KS statistic": float(ks.statistic), "KS p-value": float(ks.pvalue),
                    "level": level_from_psi(psi),
                    "detail": f"mean {a.mean():,.4g} → {b.mean():,.4g}" if psi is not None
                    else "The reference values are nearly constant, so PSI cannot be computed."})
    elif kr == "categorical":
        a, b = _cat_values(r), _cat_values(n)
        if len(a) < 30 or len(b) < 30:
            row["detail"] = "Too few values to test."
            return row
        psi = psi_categorical(a, b)
        top = a.value_counts().index[0]
        row.update({"PSI": psi, "level": level_from_psi(psi),
                    "detail": f"most common value '{top}': {(a == top).mean():.1%} → {(b == top).mean():.1%}"})
    else:
        row["detail"] = "Not tested for this column type."
    return row


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def compare_datasets(ref: pl.DataFrame, new: pl.DataFrame, bins: int = 10, seed: int | None = None) -> CompareReport:
    """Compare two datasets. Raises CompareError for unusable input."""
    seed = settings.sample_seed if seed is None else seed
    if min(ref.height, ref.width, new.height, new.width) == 0:
        raise CompareError("One of the datasets is empty.", "Load data with at least one row and one column.")
    common = [c for c in ref.columns if c in new.columns]
    if not common:
        raise CompareError("The datasets have no column names in common.",
                           "Check that both describe the same data, or rename the columns.")

    sampled = ref.height > MAX_ROWS or new.height > MAX_ROWS
    r, n = _sample(ref, seed), _sample(new, seed)
    schema = schema_diff(ref, new)
    drift = pd.DataFrame([_drift_row(c, r[c], n[c], bins) for c in common])
    drift = drift.reindex(drift["PSI"].fillna(-1).sort_values(ascending=False).index).reset_index(drop=True)
    missing = _missing_table(ref, new, common)
    summary = _summary_table(r, n, common)
    cats = category_changes(r, n, common)

    f: list[tuple[str, str]] = [("info", f"The reference has {ref.height:,} rows and {ref.width} columns; "
                                         f"the new data has {new.height:,} rows and {new.width} columns.")]
    if sampled:
        f.append(("info", f"Distribution comparisons use a random sample of at most {MAX_ROWS:,} rows per dataset."))
    status = schema.set_index("column")["status"]
    for state, text in (("only in reference", "exist only in the reference data"),
                        ("only in new", "exist only in the new data"),
                        ("type changed", "changed type")):
        cols = status[status == state].index.tolist()
        if cols:
            f.append(("warn", f"{len(cols)} column(s) {text}: {_names(cols)}."))
    high = drift.loc[drift["level"] == "high", "column"].tolist()
    mod = drift.loc[drift["level"] == "moderate", "column"].tolist()
    if high:
        f.append(("warn", f"{len(high)} column(s) show large drift (PSI above {PSI_HIGH}): {_names(high)}."))
    if mod:
        f.append(("info", f"{len(mod)} column(s) show moderate drift (PSI {PSI_MODERATE} to {PSI_HIGH}): {_names(mod)}."))
    shifted = missing.loc[missing["change (pp)"].abs() >= MISSING_SHIFT_PP, "column"].tolist()
    if shifted:
        f.append(("warn", f"The share of missing values changed by {MISSING_SHIFT_PP:.0f} points or more in: {_names(shifted)}."))
    if len(cats):
        added = cats.loc[cats["new categories"] != "", "column"].tolist()
        if added:
            f.append(("warn", f"New categories appear in: {_names(added)}. A model trained on the reference data has never seen them."))
    if not any(level == "warn" for level, _ in f):
        f.append(("ok", "No large differences were found in the shared columns."))
    return CompareReport(ref.height, new.height, ref.width, new.width, sampled, schema, drift, missing,
                         summary, cats, f)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def psi_fig(drift: pd.DataFrame, top: int = 25) -> go.Figure:
    t = drift.dropna(subset=["PSI"]).sort_values("PSI", ascending=False).head(top).iloc[::-1]
    colors = {"none": "#2CA02C", "moderate": "#FF7F0E", "high": "#D62728"}
    fig = go.Figure(go.Bar(x=t["PSI"], y=t["column"], orientation="h",
                           marker_color=[colors.get(lv, "#999999") for lv in t["level"]]))
    fig.add_vline(x=PSI_MODERATE, line_dash="dash", line_color="#FF7F0E")
    fig.add_vline(x=PSI_HIGH, line_dash="dash", line_color="#D62728")
    fig.update_layout(template="plotly_white", xaxis_title="Population Stability Index (PSI)", yaxis_title="",
                      height=max(300, 24 * len(t) + 120))
    return fig


def distribution_fig(ref: pl.DataFrame, new: pl.DataFrame, column: str, seed: int | None = None) -> go.Figure:
    """Overlay of the reference and new distributions of one column."""
    seed = settings.sample_seed if seed is None else seed
    if column not in ref.columns or column not in new.columns:
        raise CompareError(f"'{column}' is not in both datasets.", "Pick another column.")
    kr, kn = kind_of(ref.schema[column]), kind_of(new.schema[column])
    if kr != kn or kr not in ("numeric", "categorical"):
        raise CompareError("This column cannot be plotted.", "Choose a numeric or categorical column of the same type in both datasets.")
    r, n = _sample(ref, seed)[column], _sample(new, seed)[column]
    fig = go.Figure()
    if kr == "numeric":
        a, b = _num_array(r), _num_array(n)
        if len(a) == 0 or len(b) == 0:
            raise CompareError("This column has no usable numbers in one of the datasets.")
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        xbins = dict(start=lo, end=hi, size=(hi - lo) / 40) if hi > lo else None
        for name, vals, color in (("Reference", a, "#1F77B4"), ("New", b, "#D62728")):
            fig.add_histogram(x=vals, name=name, histnorm="probability density", opacity=0.6,
                              marker_color=color, xbins=xbins)
        fig.update_layout(barmode="overlay", yaxis_title="Density")
    else:
        a, b = _cat_values(r), _cat_values(n)
        if len(a) == 0 or len(b) == 0:
            raise CompareError("This column has no values in one of the datasets.")
        sa, sb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        order = (sa.add(sb, fill_value=0)).sort_values(ascending=False).head(15).index.tolist()
        fig.add_bar(x=order, y=[float(sa.get(k, 0.0)) for k in order], name="Reference", marker_color="#1F77B4")
        fig.add_bar(x=order, y=[float(sb.get(k, 0.0)) for k in order], name="New", marker_color="#D62728")
        fig.update_layout(barmode="group", yaxis_title="Share of rows", yaxis_tickformat=".0%")
    fig.update_layout(template="plotly_white", xaxis_title=column)
    return fig