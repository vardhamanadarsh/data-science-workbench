"""Data Health assessment: quality checks and an explainable 0-100 score.

UI-free. Pass the dataset the user loaded (for huge files the loader has
already sampled it). Skewness, kurtosis and outliers are computed on at most
`stats_max_rows` rows; missing values, duplicates and uniqueness use all rows.

Nothing is ever removed or modified here. Findings only describe problems and
suggest options; the user decides what to do in Preprocessing.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from loguru import logger

from config import settings

Severity = Literal["ok", "info", "warn", "bad"]
ICON: dict[str, str] = {"ok": "✓", "info": "ℹ", "warn": "⚠", "bad": "✗"}

# Thresholds (design choices, easy to tune)
SKEW_THRESHOLD = 2.0
OUTLIER_PCT_THRESHOLD = 5.0
CORR_THRESHOLD = 0.95
LEAKAGE_CORR_THRESHOLD = 0.95
NEAR_CONSTANT_SHARE = 0.99
CLASS_MAX_UNIQUE_INT = 10      # integer target with <= this many values = classes
CLASS_MAX_UNIQUE_TEXT = 50     # text target with <= this many values = classes

_ID_NAME = re.compile(r"(^|[_\s.\-])(id|uuid|guid)$", re.IGNORECASE)
_CAMEL_ID = re.compile(r"[a-z](Id|ID)$")
_NONNEG_NAME = re.compile(
    r"(^|[_\s.\-])(age|price|cost|count|qty|quantity|amount|salary|income|height|weight|distance|duration)($|[_\s.\-])",
    re.IGNORECASE,
)
_PLACEHOLDERS = {"", "?", "-", "--", "n/a", "#n/a", "nan", "null", "none", "unknown", "missing", "undefined", "tbd"}
_SENTINELS = (-9999.0, -999.0, 9999.0, 99999.0)

_PROFILE_SCHEMA: dict[str, Any] = {
    "column": pl.String, "dtype": pl.String, "kind": pl.String,
    "missing": pl.Int64, "missing_pct": pl.Float64,
    "unique": pl.Int64, "unique_pct": pl.Float64, "top_share": pl.Float64,
    "constant": pl.Boolean, "near_constant": pl.Boolean,
    "skew": pl.Float64, "kurtosis": pl.Float64,
    "outliers": pl.Int64, "outlier_pct": pl.Float64,
    "suspicious": pl.String,
}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Finding:
    """One line of the health report."""

    category: str
    severity: Severity
    title: str
    detail: str = ""
    penalty: float = 0.0          # points taken off the score
    suggestion: str = ""

    @property
    def icon(self) -> str:
        return ICON[self.severity]


@dataclass(frozen=True)
class LeakageCandidate:
    """A variable that MAY leak information about the target (never certain)."""

    column: str
    reason: str
    strength: Literal["strong", "weak"]
    score: float | None = None


@dataclass
class HealthReport:
    n_rows: int
    n_cols: int
    target: str | None
    score: int
    label: str
    findings: list[Finding]
    columns: pl.DataFrame                    # one profile row per column
    correlated_pairs: pl.DataFrame
    class_distribution: pl.DataFrame | None
    leakage: list[LeakageCandidate]
    duplicate_rows: int
    stats_rows: int                          # rows used for distribution statistics

    @property
    def headline(self) -> str:
        return f"Dataset Health: {self.score}/100 ({self.label})"

    def breakdown(self) -> pl.DataFrame:
        """Why the score is not 100: every check that cost points."""
        rows = [{"check": f.title, "points_lost": f.penalty} for f in self.findings if f.penalty > 0]
        return pl.DataFrame(rows, schema={"check": pl.String, "points_lost": pl.Float64})

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target": self.target,
            "duplicate_rows": self.duplicate_rows,
            "findings": [asdict(f) for f in self.findings],
            "columns": self.columns.to_dicts(),
            "correlated_pairs": self.correlated_pairs.to_dicts(),
            "class_distribution": (
                self.class_distribution.to_dicts() if self.class_distribution is not None else None
            ),
            "leakage": [asdict(c) for c in self.leakage],
        }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _num(value: object) -> float | None:
    """Convert to a finite float, else None."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _names(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _is_id_name(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))


def _finite_floats(s: pl.Series) -> pl.Series:
    x = s.cast(pl.Float64, strict=False).drop_nulls()
    return x.filter(x.is_finite())


def _missing_count(s: pl.Series) -> int:
    count = s.null_count()
    if s.dtype.is_float():
        count += int(s.is_nan().sum())
    return int(count)


def _top_share(non_null: pl.Series) -> float:
    """Share of the most frequent value (1.0 = constant)."""
    if non_null.len() == 0:
        return 1.0
    counts = non_null.alias("value").value_counts(sort=True)
    return float(counts.row(0)[1]) / non_null.len()


def infer_kind(s: pl.Series, n_unique: int) -> str:
    """Rough variable type: numeric, categorical, datetime, text, id, boolean, other.

    `n_unique` is the number of distinct non-null values.
    """
    dtype = s.dtype
    non_null = s.drop_nulls()
    ratio = n_unique / max(non_null.len(), 1)
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_temporal():
        return "datetime"
    if dtype in (pl.Categorical, pl.Enum):
        return "categorical"
    if dtype.is_integer():
        return "id" if _is_id_name(s.name) and ratio >= 0.9 else "numeric"
    if dtype.is_numeric():
        return "numeric"
    if dtype == pl.String:
        if _is_id_name(s.name) and ratio >= 0.5:
            return "id"
        avg_len = non_null.head(5000).str.len_chars().mean() if non_null.len() else 0
        if (avg_len or 0) > 30 and ratio > 0.5:
            return "text"
        if ratio >= 0.95 and s.len() >= 50:
            return "id"
        return "categorical"
    return "other"


def _numeric_stats(s: pl.Series) -> dict[str, Any]:
    """Skewness, excess kurtosis and IQR outliers (finite values only)."""
    out: dict[str, Any] = {"skew": None, "kurtosis": None, "outliers": 0, "outlier_pct": 0.0}
    x = _finite_floats(s)
    if x.len() < 8 or x.n_unique() < 3:
        return out
    out["skew"] = _num(x.skew())
    out["kurtosis"] = _num(x.kurtosis())
    q1 = _num(x.quantile(0.25, interpolation="linear"))
    q3 = _num(x.quantile(0.75, interpolation="linear"))
    if q1 is not None and q3 is not None and q3 > q1:
        iqr = q3 - q1
        n_out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        out["outliers"] = n_out
        out["outlier_pct"] = round(n_out / x.len() * 100, 2)
    return out


def _suspicious_values(s: pl.Series, kind: str) -> list[str]:
    """Things that look wrong. These are hints, not proof."""
    messages: list[str] = []
    if s.dtype == pl.String:
        values = s.drop_nulls().head(200_000)
        hits = int(values.str.strip_chars().str.to_lowercase().is_in(list(_PLACEHOLDERS)).sum())
        if hits:
            messages.append(f"{hits:,} placeholder-like text values (e.g. '?', 'N/A', 'unknown')")
    elif kind == "numeric" and s.dtype.is_numeric():
        if s.dtype.is_float():
            infinite = int(s.is_infinite().sum())
            if infinite:
                messages.append(f"{infinite:,} infinite values")
        x = _finite_floats(s)
        for sentinel in _SENTINELS:
            hits = int((x == sentinel).sum())
            if hits >= 3:
                messages.append(f"{hits:,} values equal to {sentinel:g} (possible missing-value code)")
        if _NONNEG_NAME.search(s.name):
            negatives = int((x < 0).sum())
            if negatives:
                messages.append(f"{negatives:,} negative values in a column that is normally non-negative")
    return messages


def _empty_row(name: str, dtype: str) -> dict[str, Any]:
    return {
        "column": name, "dtype": dtype, "kind": "other", "missing": 0, "missing_pct": 0.0,
        "unique": 0, "unique_pct": 0.0, "top_share": 0.0, "constant": False, "near_constant": False,
        "skew": None, "kurtosis": None, "outliers": 0, "outlier_pct": 0.0, "suspicious": "",
    }


def _profile_column(full: pl.Series, sampled: pl.Series) -> dict[str, Any]:
    row = _empty_row(full.name, str(full.dtype))
    height = max(full.len(), 1)
    non_null = full.drop_nulls()
    unique = int(non_null.n_unique()) if non_null.len() else 0
    kind = infer_kind(full, unique)
    top_share = _top_share(non_null)
    missing = _missing_count(full)

    row.update(
        kind=kind,
        missing=missing,
        missing_pct=round(missing / height * 100, 2),
        unique=unique,
        unique_pct=round(unique / height * 100, 2),
        top_share=round(top_share, 4),
        constant=unique <= 1,
        near_constant=unique > 1 and top_share >= NEAR_CONSTANT_SHARE,
        suspicious="; ".join(_suspicious_values(full, kind)),
    )
    if kind == "numeric":
        row.update(_numeric_stats(sampled))
    return row


# --------------------------------------------------------------------------- #
# Target-related checks
# --------------------------------------------------------------------------- #

def is_classification_target(s: pl.Series) -> bool:
    """True if the column looks like a set of classes rather than a measurement."""
    non_null = s.drop_nulls()
    if non_null.len() == 0:
        return False
    if s.dtype == pl.Boolean:
        return True
    if s.dtype in (pl.String, pl.Categorical, pl.Enum):
        return non_null.n_unique() <= CLASS_MAX_UNIQUE_TEXT
    if s.dtype.is_integer():
        return non_null.n_unique() <= CLASS_MAX_UNIQUE_INT
    return False


def class_distribution(s: pl.Series) -> pl.DataFrame:
    """Columns: class, count, pct (largest class first)."""
    non_null = s.drop_nulls()
    total = max(non_null.len(), 1)
    counts = non_null.alias("value").value_counts(sort=True)
    return counts.select(
        pl.col("value").cast(pl.String).alias("class"),
        pl.col("count"),
    ).with_columns((pl.col("count") / total * 100).round(2).alias("pct"))


def imbalance_level(dist: pl.DataFrame) -> tuple[str, float]:
    """Return ("single_class" | "severe" | "moderate" | "balanced", majority/minority ratio)."""
    if dist.height < 2:
        return "single_class", 1.0
    largest, smallest = int(dist["count"].max()), int(dist["count"].min())
    ratio = largest / max(smallest, 1)
    minority_pct = smallest / int(dist["count"].sum()) * 100
    if ratio >= 10 or minority_pct < 5:
        return "severe", ratio
    if ratio >= 3:
        return "moderate", ratio
    return "balanced", ratio


def _target_as_numeric(t: pl.Series) -> pl.Series | None:
    """Numeric version of the target for correlation (None for multi-class text)."""
    if t.dtype == pl.Boolean:
        return t.cast(pl.Int8).cast(pl.Float64)
    if t.dtype.is_numeric():
        return t.cast(pl.Float64, strict=False)
    classes = t.unique().sort()
    if classes.len() == 2:
        return (t == classes[1]).cast(pl.Float64)
    return None


def find_leakage_candidates(
    df: pl.DataFrame, target: str, max_rows: int = 100_000
) -> list[LeakageCandidate]:
    """Look for variables that may leak the target. Always 'possible', never certain.

    Checks: identical copy of the target, near-perfect correlation, categorical
    values that almost fully determine the target, and similar column names.
    """
    if target not in df.columns:
        return []
    frame = df.filter(pl.col(target).is_not_null())
    if frame.height < 20:
        return []
    if frame.height > max_rows:
        frame = frame.sample(n=max_rows, seed=settings.sample_seed)

    t = frame[target]
    features = [c for c in frame.columns if c != target]
    found: dict[str, LeakageCandidate] = {}

    # 1. Identical copy of the target
    for col in features:
        try:
            if frame[col].equals(t):
                found[col] = LeakageCandidate(col, "Values are identical to the target.", "strong", 1.0)
        except Exception:  # incomparable dtypes
            continue

    # 2. Near-perfect numeric correlation
    t_num = _target_as_numeric(t)
    if t_num is not None:
        num_cols = [c for c in features if c not in found and frame[c].dtype.is_numeric()]
        if num_cols:
            data = frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols])
            data = data.with_columns(t_num.alias("__target__"))
            pdf = data.to_pandas().replace([np.inf, -np.inf], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corrs = pdf[num_cols].corrwith(pdf["__target__"]).abs()
            for col, value in corrs.items():
                score = _num(value)
                if score is not None and score >= LEAKAGE_CORR_THRESHOLD:
                    found[str(col)] = LeakageCandidate(
                        str(col), f"Correlation with the target is {score:.3f}.", "strong", round(score, 4)
                    )

    # 3. Categories that (almost) fully determine a class target
    if is_classification_target(t):
        baseline = _top_share(t.drop_nulls())
        max_unique = max(2, min(50, frame.height // 20))
        for col in features:
            s = frame[col]
            if col in found or s.dtype not in (pl.String, pl.Categorical, pl.Enum, pl.Boolean):
                continue
            n_unique = s.n_unique()
            if not 2 <= n_unique <= max_unique:
                continue
            grouped = frame.group_by([col, target]).agg(pl.len().alias("n"))
            best = grouped.group_by(col).agg(pl.col("n").max().alias("m"))["m"].sum()
            purity = float(best) / frame.height
            if purity >= 0.99 and purity - baseline >= 0.05:
                found[col] = LeakageCandidate(
                    col, f"Its values almost fully determine the target ({purity:.1%}).", "strong", round(purity, 4)
                )

    # 4. Similar names (weak signal)
    tl = target.lower()
    if len(tl) >= 4:
        for col in features:
            if col not in found and tl in col.lower():
                found[col] = LeakageCandidate(col, "Its name contains the target name.", "weak")

    return sorted(found.values(), key=lambda c: (c.strength != "strong", -(c.score or 0)))


# --------------------------------------------------------------------------- #
# Correlated pairs
# --------------------------------------------------------------------------- #

def _correlated_pairs(
    sample: pl.DataFrame, numeric_cols: list[str], threshold: float, max_cols: int = 100, max_rows: int = 50_000
) -> pl.DataFrame:
    schema: dict[str, Any] = {"column_a": pl.String, "column_b": pl.String, "abs_corr": pl.Float64}
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return pl.DataFrame(schema=schema)
    frame = sample.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols])
    if frame.height > max_rows:
        frame = frame.sample(n=max_rows, seed=settings.sample_seed)
    pdf = frame.to_pandas().replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = pdf.corr().to_numpy()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = matrix[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append({"column_a": cols[i], "column_b": cols[j], "abs_corr": round(float(abs(value)), 4)})
    return pl.DataFrame(pairs, schema=schema).sort("abs_corr", descending=True)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def _score_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Fair"
    return "Poor"


def assess_data_health(
    df: pl.DataFrame, target: str | None = None, stats_max_rows: int = 200_000
) -> HealthReport:
    """Run all health checks and build an explainable 0-100 score.

    Raises:
        ValueError: if the frame is empty or the target column does not exist.
    """
    if df.height == 0 or df.width == 0:
        raise ValueError("The dataset is empty, so there is nothing to assess.")
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    sample = df if df.height <= stats_max_rows else df.sample(n=stats_max_rows, seed=settings.sample_seed)

    rows: list[dict[str, Any]] = []
    for name in df.columns:
        try:
            rows.append(_profile_column(df[name], sample[name]))
        except Exception as exc:  # unusual dtype: keep going
            logger.warning("Could not profile column {}: {}", name, exc)
            rows.append(_empty_row(name, str(df[name].dtype)))
    columns = pl.DataFrame(rows, schema=_PROFILE_SCHEMA)

    try:
        duplicate_rows = int(df.height - df.n_unique())
    except Exception as exc:  # e.g. list columns
        logger.warning("Duplicate-row check skipped: {}", exc)
        duplicate_rows = 0

    findings: list[Finding] = []

    def add(category: str, severity: Severity, title: str, detail: str = "",
            penalty: float = 0.0, suggestion: str = "") -> None:
        findings.append(Finding(category, severity, title, detail, round(penalty, 1), suggestion))

    numeric = [r for r in rows if r["kind"] == "numeric"]

    # --- Sample size
    if df.height < 30:
        add("size", "bad", f"Very small dataset ({df.height} rows)", "", 15, "Statistics and models will be unreliable.")
    elif df.height < 100:
        add("size", "warn", f"Small dataset ({df.height} rows)", "", 8, "Prefer simple models and cross-validation.")
    elif df.height < 300:
        add("size", "info", f"Modest dataset size ({df.height} rows)", "", 3)

    # --- Missing values
    total_missing = sum(r["missing"] for r in rows)
    miss_pct = total_missing / (df.height * df.width) * 100
    if total_missing == 0:
        add("missing", "ok", "No missing values")
    else:
        worst = sorted((r for r in rows if r["missing"] > 0), key=lambda r: r["missing_pct"], reverse=True)
        severity: Severity = "ok" if miss_pct < 5 else ("warn" if miss_pct < 20 else "bad")
        label = "Low missingness" if severity == "ok" else "Missing values"
        add(
            "missing", severity, f"{label}: {miss_pct:.1f}% of cells are empty",
            "Most affected: " + ", ".join(f"{r['column']} ({r['missing_pct']:.0f}%)" for r in worst[:3]),
            min(20.0, miss_pct),
            "Impute, or drop columns/rows, in Preprocessing. Nothing is removed automatically.",
        )
    heavy = [r["column"] for r in rows if r["missing_pct"] >= 50]
    if heavy:
        add("missing", "warn", f"{len(heavy)} column(s) are more than half empty", _names(heavy),
            min(10.0, 3.0 * len(heavy)), "Consider leaving these columns out of models.")

    # --- Duplicates
    if duplicate_rows == 0:
        add("duplicates", "ok", "No duplicate rows")
    else:
        dup_pct = duplicate_rows / df.height * 100
        add("duplicates", "warn", f"{duplicate_rows:,} duplicate rows ({dup_pct:.1f}%)", "",
            min(10.0, max(1.0, dup_pct * 2)),
            "Review them first; remove only if they are true repeats.")

    dup_ids: dict[str, int] = {}
    for r in rows:
        if r["kind"] == "id":
            values = df[r["column"]].drop_nulls()
            extra = int(values.len() - values.n_unique())
            if extra > 0:
                dup_ids[r["column"]] = extra
    if dup_ids:
        add("duplicates", "bad", f"Duplicate values in ID column(s): {_names(list(dup_ids))}",
            ", ".join(f"{k}: {v:,} repeats" for k, v in list(dup_ids.items())[:3]), 5.0,
            "IDs should be unique. Check for repeated records or a wrong ID column.")

    # --- Constant / near-constant
    constant = [r["column"] for r in rows if r["constant"]]
    near = [r["column"] for r in rows if r["near_constant"]]
    if constant:
        add("variation", "warn", f"{len(constant)} constant column(s)", _names(constant),
            min(8.0, 2.0 * len(constant)), "Constant columns carry no information; consider excluding them.")
    if near:
        add("variation", "info", f"{len(near)} near-constant column(s) (one value in ≥99% of rows)",
            _names(near), min(4.0, 1.0 * len(near)), "These rarely help models.")
    if not constant and not near:
        add("variation", "ok", "No constant or near-constant columns")

    # --- Distribution shape and outliers
    if numeric:
        skewed = [r for r in numeric if r["skew"] is not None and abs(r["skew"]) > SKEW_THRESHOLD]
        if skewed:
            add("skew", "warn", f"{len(skewed)} highly skewed variable(s)",
                ", ".join(f"{r['column']} ({r['skew']:.1f})" for r in skewed[:4]),
                min(6.0, 1.0 * len(skewed)), "A log or power transform may help linear models.")
        else:
            add("skew", "ok", "No highly skewed numeric variables")

        outlying = [r for r in numeric if r["outlier_pct"] > OUTLIER_PCT_THRESHOLD]
        if outlying:
            add("outliers", "warn", f"{len(outlying)} variable(s) with more than {OUTLIER_PCT_THRESHOLD:.0f}% outliers (IQR rule)",
                ", ".join(f"{r['column']} ({r['outlier_pct']:.0f}%)" for r in outlying[:4]),
                min(6.0, 1.0 * len(outlying)), "Inspect them in Anomaly Detection. They are never removed automatically.")
        else:
            add("outliers", "ok", "Outliers are within a normal range (IQR rule)")

    # --- Correlated variables
    usable = [r["column"] for r in numeric if not r["constant"]]
    pairs = _correlated_pairs(sample, usable, CORR_THRESHOLD)
    if pairs.height:
        top = pairs.head(3).rows()
        add("correlation", "warn", f"{pairs.height} highly correlated pair(s) (|r| ≥ {CORR_THRESHOLD})",
            "; ".join(f"{a} ~ {b} ({c:.2f})" for a, b, c in top),
            min(6.0, 2.0 * pairs.height), "Redundant variables can destabilise linear models and inflate VIF.")
    elif len(usable) >= 2:
        add("correlation", "ok", "No highly correlated numeric pairs")

    # --- Suspicious values
    odd = [(r["column"], r["suspicious"]) for r in rows if r["suspicious"]]
    if odd:
        add("suspicious", "warn", f"Suspicious values in {len(odd)} column(s)",
            "; ".join(f"{c}: {m}" for c, m in odd[:3]), min(6.0, 2.0 * len(odd)),
            "Check whether these are real values or codes for 'missing'.")
    else:
        add("suspicious", "ok", "No suspicious placeholder or sentinel values found")

    # --- Target-dependent checks
    leakage: list[LeakageCandidate] = []
    distribution: pl.DataFrame | None = None
    if target is None:
        add("target", "info", "No target selected: leakage and class-imbalance checks were skipped")
    else:
        leakage = find_leakage_candidates(df, target)
        strong = [c for c in leakage if c.strength == "strong"]
        weak = [c for c in leakage if c.strength == "weak"]
        if strong:
            add("leakage", "bad", f"Possible leakage in {len(strong)} variable(s)",
                "; ".join(f"{c.column}: {c.reason}" for c in strong[:3]),
                min(15.0, 8.0 * len(strong)),
                "Investigate these before training. They may be legitimate, but they can inflate scores.")
        if weak:
            add("leakage", "warn", f"{len(weak)} variable name(s) resemble the target",
                _names([c.column for c in weak]), 0.0, "Check that they are not derived from the target.")
        if not leakage:
            add("leakage", "ok", "No obvious leakage signals found (this is not a proof)")

        if is_classification_target(df[target]):
            distribution = class_distribution(df[target])
            level, ratio = imbalance_level(distribution)
            shares = ", ".join(f"{r[0]} {r[2]:.1f}%" for r in distribution.head(4).rows())
            if level == "single_class":
                add("imbalance", "bad", "The target has only one class", "", 15.0, "Choose another target column.")
            elif level == "severe":
                add("imbalance", "bad", f"Severe class imbalance (ratio {ratio:.0f}:1)", shares, 8.0,
                    "Consider class weights, resampling or SMOTE, and use F1 / PR-AUC instead of accuracy.")
            elif level == "moderate":
                add("imbalance", "warn", f"Class imbalance detected (ratio {ratio:.1f}:1)", shares, 4.0,
                    "Consider class weights and stratified splits.")
            else:
                add("imbalance", "ok", "Classes are reasonably balanced", shares)

    score = int(max(0, min(100, round(100 - sum(f.penalty for f in findings)))))
    return HealthReport(
        n_rows=df.height,
        n_cols=df.width,
        target=target,
        score=score,
        label=_score_label(score),
        findings=findings,
        columns=columns,
        correlated_pairs=pairs,
        class_distribution=distribution,
        leakage=leakage,
        duplicate_rows=duplicate_rows,
        stats_rows=sample.height,
    )