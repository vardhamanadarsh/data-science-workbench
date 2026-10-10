"""Anomaly detection (UI-free): IQR, Z-score, Isolation Forest, Local Outlier Factor.

Nothing is ever removed. Results only describe which rows look unusual; the
user decides what to do (inspect, export, or add a flag column).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config import MODE_ROW_LIMITS, ProcessingMode, settings

METHODS = ["IQR", "Z-score", "Isolation Forest", "Local Outlier Factor"]
METHOD_SLUG = {"IQR": "iqr", "Z-score": "zscore", "Isolation Forest": "iforest", "Local Outlier Factor": "lof"}
METHOD_ROW_CAPS: dict[str, int | None] = {
    "IQR": None, "Z-score": None, "Isolation Forest": 200_000, "Local Outlier Factor": 20_000,
}
SCORE_LABEL = {
    "Z-score": "Largest |z-score| across the columns",
    "Isolation Forest": "Anomaly score (higher = more unusual)",
    "Local Outlier Factor": "Outlier factor (higher = more unusual)",
}
_MULTIVARIATE = {"Isolation Forest", "Local Outlier Factor"}
MAX_COLUMNS = 50
MIN_ROWS = 20
PLOT_MAX_POINTS = 20_000

_UNI_SCHEMA = {"column": pl.String, "checked": pl.Int64, "anomalies": pl.Int64, "pct": pl.Float64}
_MULTI_SCHEMA = {"column": pl.String, "flagged_deviation": pl.Float64,
                 "other_deviation": pl.Float64, "ratio": pl.Float64}


class AnomalyError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


@dataclass
class AnomalyResult:
    method: str
    columns: list[str]
    row_ids: np.ndarray              # positions (0-based) of the analysed rows in the dataset
    values: np.ndarray               # analysed values, shape (rows, columns)
    mask: np.ndarray                 # True = flagged as anomaly
    scores: np.ndarray | None
    per_column: pl.DataFrame
    notes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    n_total_rows: int = 0

    @property
    def n_rows(self) -> int:
        return int(len(self.row_ids))

    @property
    def n_anomalies(self) -> int:
        return int(self.mask.sum())

    @property
    def pct(self) -> float:
        return self.n_anomalies / max(self.n_rows, 1) * 100


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _univariate(X: np.ndarray, cols: list[str], method: str, factor: float, z_threshold: float):
    n, k = X.shape
    flags = np.zeros((n, k), dtype=bool)
    checked = np.zeros(k, dtype=int)
    scores = np.zeros(n) if method == "Z-score" else None
    skipped: list[str] = []
    with np.errstate(invalid="ignore"):
        for j, col in enumerate(cols):
            x = X[:, j]
            finite = x[np.isfinite(x)]
            checked[j] = finite.size
            if finite.size < 8:
                skipped.append(col)
                continue
            if method == "IQR":
                q1, q3 = np.percentile(finite, [25, 75])
                spread = q3 - q1
                if spread <= 0:
                    skipped.append(col)
                    continue
                flags[:, j] = (x < q1 - factor * spread) | (x > q3 + factor * spread)
            else:
                sd = finite.std()
                if sd <= 0:
                    skipped.append(col)
                    continue
                z = np.abs((x - finite.mean()) / sd)
                flags[:, j] = z > z_threshold
                scores = np.maximum(scores, np.nan_to_num(z, nan=0.0))
    if len(skipped) == k:
        raise AnomalyError("None of the selected columns could be checked.",
                           "They are constant, nearly constant, or have too few values.")
    notes = [f"Skipped (constant or too few values): {', '.join(skipped)}."] if skipped else []
    counts = flags.sum(axis=0)
    pct = [round(float(c) / max(int(m), 1) * 100, 2) for c, m in zip(counts, checked)]
    table = pl.DataFrame(
        {"column": cols, "checked": checked.tolist(), "anomalies": counts.tolist(), "pct": pct},
        schema=_UNI_SCHEMA,
    ).sort("anomalies", descending=True)
    return flags.any(axis=1), scores, table, notes


def _deviation_table(X: np.ndarray, mask: np.ndarray, cols: list[str]) -> pl.DataFrame:
    """Which variables make the flagged rows unusual (robust z-score from the median)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, col in enumerate(cols):
            x = X[:, j]
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)) * 1.4826)
            scale = mad if mad > 0 else float(x.std())
            if scale > 0:
                d = np.abs(x - med) / scale
                flagged = float(d[mask].mean()) if mask.any() else 0.0
                other = float(d[~mask].mean()) if (~mask).any() else 0.0
            else:
                flagged = other = 0.0
            rows.append({"column": col, "flagged_deviation": flagged, "other_deviation": other,
                         "ratio": flagged / other if other > 0 else None})
    return pl.DataFrame(rows, schema=_MULTI_SCHEMA).sort("flagged_deviation", descending=True)


def detect_anomalies(
    df: pl.DataFrame, columns: list[str], method: str, mode: ProcessingMode = ProcessingMode.STANDARD,
    factor: float = 1.5, z_threshold: float = 3.0, contamination: float = 0.05,
    n_neighbors: int = 20, seed: int | None = None,
) -> AnomalyResult:
    """Flag unusual rows. Raises AnomalyError with a suggestion for unusable input."""
    seed = settings.sample_seed if seed is None else seed
    if method not in METHODS:
        raise AnomalyError(f"Unknown method '{method}'.")
    cols = list(dict.fromkeys(columns))
    if not cols:
        raise AnomalyError("No columns selected.", "Choose at least one numeric column.")
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise AnomalyError(f"Column(s) not found: {', '.join(absent)}.", "Pick the columns again.")
    non_numeric = [c for c in cols if not df[c].dtype.is_numeric()]
    if non_numeric:
        raise AnomalyError(f"Not numeric: {', '.join(non_numeric)}.", "Choose numeric columns only.")
    if len(cols) > MAX_COLUMNS:
        raise AnomalyError(f"Too many columns ({len(cols)}).", f"Choose at most {MAX_COLUMNS}.")
    if method in _MULTIVARIATE and not 0 < contamination <= 0.5:
        raise AnomalyError("The expected share of anomalies must be between 0 and 0.5.")

    notes: list[str] = []
    frame = df.select([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in cols])
    frame = frame.with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in cols]
    ).with_row_index("__row__")

    if method in _MULTIVARIATE:
        before = frame.height
        frame = frame.drop_nulls(subset=cols)
        if before - frame.height:
            notes.append(f"{before - frame.height:,} row(s) with missing values were not analysed "
                         "(this method needs complete rows).")
    caps = [c for c in (MODE_ROW_LIMITS[mode], METHOD_ROW_CAPS[method]) if c is not None]
    cap = min(caps) if caps else None
    if cap and frame.height > cap:
        notes.append(f"A random sample of {cap:,} of {frame.height:,} rows was analysed "
                     f"(limit of this method and the '{mode.value}' mode).")
        frame = frame.sample(n=cap, seed=seed).sort("__row__")
    if frame.height < MIN_ROWS:
        raise AnomalyError(f"Only {frame.height} usable rows.", f"At least {MIN_ROWS} are needed.")

    row_ids = frame["__row__"].to_numpy().astype(np.int64)
    X = frame.select(cols).to_numpy().astype(float)

    if method in _MULTIVARIATE:
        if method == "Isolation Forest":
            model = IsolationForest(n_estimators=200, contamination=contamination,
                                    random_state=seed, n_jobs=-1)
            pred = model.fit_predict(X)
            scores = -model.score_samples(X)
        else:
            k = max(2, min(n_neighbors, len(X) - 1))
            model = LocalOutlierFactor(n_neighbors=k, contamination=contamination, n_jobs=-1)
            pred = model.fit_predict(StandardScaler().fit_transform(X))
            scores = -model.negative_outlier_factor_
        mask = pred == -1
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
        per_column = _deviation_table(X, mask, cols)
    else:
        mask, scores, per_column, extra = _univariate(X, cols, method, factor, z_threshold)
        notes += extra

    return AnomalyResult(
        method=method, columns=cols, row_ids=row_ids, values=X, mask=mask, scores=scores,
        per_column=per_column, notes=notes, n_total_rows=df.height,
        params={"factor": factor, "z_threshold": z_threshold, "contamination": contamination,
                "n_neighbors": n_neighbors},
    )


# --------------------------------------------------------------------------- #
# Using the result
# --------------------------------------------------------------------------- #

def anomaly_rows(df: pl.DataFrame, result: AnomalyResult, limit: int = 1000) -> pl.DataFrame:
    """Original rows flagged as anomalies (most unusual first when a score exists).

    The 'row' column is the 0-based position of the row in the dataset.
    """
    ids = result.row_ids[result.mask]
    if ids.size == 0:
        return df.head(0)
    data: dict = {"row": ids.astype(np.uint32)}
    if result.scores is not None:
        data["anomaly_score"] = result.scores[result.mask]
    flagged = pl.DataFrame(data)
    if result.scores is not None:
        flagged = flagged.sort("anomaly_score", descending=True)
    flagged = flagged.head(limit)
    out = df.with_row_index("row").with_columns(pl.col("row").cast(pl.UInt32)).join(flagged, on="row", how="inner")
    return out.sort("anomaly_score", descending=True) if result.scores is not None else out.sort("row")


def flag_column(df: pl.DataFrame, result: AnomalyResult, name: str) -> pl.DataFrame:
    """Add a True/False column. Rows that were not analysed are left empty (missing)."""
    if not name.strip():
        raise AnomalyError("Enter a column name.")
    if name in df.columns:
        raise AnomalyError(f"A column named '{name}' already exists.", "Use another name.")
    values = np.zeros(df.height, dtype=bool)
    known = np.zeros(df.height, dtype=bool)
    values[result.row_ids] = result.mask
    known[result.row_ids] = True
    tmp = pl.DataFrame({"known": known, "val": values})
    column = tmp.select(pl.when(pl.col("known")).then(pl.col("val")).alias(name)).to_series()
    return df.with_columns(column)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _plot_idx(result: AnomalyResult) -> np.ndarray:
    n = result.n_rows
    if n <= PLOT_MAX_POINTS:
        return np.arange(n)
    rng = np.random.default_rng(42)
    anomalies = np.flatnonzero(result.mask)
    normal = np.flatnonzero(~result.mask)
    if anomalies.size > PLOT_MAX_POINTS // 2:
        anomalies = rng.choice(anomalies, PLOT_MAX_POINTS // 2, replace=False)
    normal = rng.choice(normal, min(PLOT_MAX_POINTS - anomalies.size, normal.size), replace=False)
    return np.sort(np.concatenate([anomalies, normal]))


def _two_trace_scatter(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for flag, name, color, size, opacity in ((False, "Normal", "#1F77B4", 5, 0.5), (True, "Anomaly", "#D62728", 7, 0.9)):
        sel = m == flag
        fig.add_trace(go.Scattergl(x=x[sel], y=y[sel], mode="markers", name=name,
                                   marker=dict(size=size, color=color, opacity=opacity)))
    fig.update_layout(template="plotly_white")
    return fig


def scatter_fig(result: AnomalyResult, x_col: str, y_col: str) -> go.Figure:
    idx = _plot_idx(result)
    xi, yi = result.columns.index(x_col), result.columns.index(y_col)
    fig = _two_trace_scatter(result.values[idx, xi], result.values[idx, yi], result.mask[idx])
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig


def projection_fig(result: AnomalyResult) -> go.Figure:
    """2-D PCA projection of all selected columns."""
    if len(result.columns) < 2:
        raise AnomalyError("A projection needs at least two columns.")
    idx = _plot_idx(result)
    X = result.values[idx].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(X, axis=0)
    missing = np.where(np.isnan(X))
    X[missing] = np.take(np.nan_to_num(med), missing[1])
    pca = PCA(n_components=2, random_state=settings.sample_seed)
    comp = pca.fit_transform(StandardScaler().fit_transform(X))
    fig = _two_trace_scatter(comp[:, 0], comp[:, 1], result.mask[idx])
    ratio = pca.explained_variance_ratio_
    fig.update_layout(xaxis_title=f"PC1 ({ratio[0]:.0%} of variance)", yaxis_title=f"PC2 ({ratio[1]:.0%})")
    return fig


def score_fig(result: AnomalyResult) -> go.Figure | None:
    if result.scores is None:
        return None
    fig = go.Figure()
    fig.add_histogram(x=result.scores[~result.mask], name="Normal", opacity=0.7, marker_color="#1F77B4")
    fig.add_histogram(x=result.scores[result.mask], name="Anomaly", opacity=0.7, marker_color="#D62728")
    fig.update_layout(barmode="overlay", template="plotly_white",
                      xaxis_title=SCORE_LABEL.get(result.method, "Score"), yaxis_title="Rows")
    return fig


def distribution_fig(result: AnomalyResult, col: str) -> go.Figure:
    j = result.columns.index(col)
    x = result.values[:, j]
    ok = np.isfinite(x)
    fig = go.Figure()
    fig.add_histogram(x=x[ok & ~result.mask], nbinsx=60, name="Normal", opacity=0.7, marker_color="#1F77B4")
    fig.add_histogram(x=x[ok & result.mask], nbinsx=60, name="Anomaly", opacity=0.7, marker_color="#D62728")
    fig.update_layout(barmode="overlay", template="plotly_white", xaxis_title=col, yaxis_title="Rows")
    return fig


def affected_fig(result: AnomalyResult) -> go.Figure:
    t = result.per_column
    fig = go.Figure()
    if "anomalies" in t.columns:
        fig.add_bar(x=t["column"].to_list(), y=t["pct"].to_list(), name="Flagged")
        fig.update_layout(yaxis_title="Share of values flagged (%)")
    else:
        fig.add_bar(x=t["column"].to_list(), y=t["flagged_deviation"].to_list(), name="Flagged rows")
        fig.add_bar(x=t["column"].to_list(), y=t["other_deviation"].to_list(), name="Other rows")
        fig.update_layout(barmode="group", yaxis_title="Typical distance from the median (robust z)")
    fig.update_layout(template="plotly_white")
    return fig