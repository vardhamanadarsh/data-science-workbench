import numpy as np
import polars as pl
import pytest

from core.anomaly_detection import (
    AnomalyError, anomaly_rows, detect_anomalies, flag_column, projection_fig, scatter_fig,
)
from core.automl import leaderboard_notes, native_automl, run_flaml
from core.leakage import assess_leakage, check_contamination
from core.model_training import TrainConfig, train_models


def _anom_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    a, b = rng.normal(size=n), rng.normal(size=n)
    a[:5] = 25.0 + rng.normal(size=5)
    b[:5] = -25.0 + rng.normal(size=5)
    return pl.DataFrame({"a": a, "b": b})


def _clf_df(n: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def _leak_df() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    n = 600
    x1 = rng.normal(size=n)
    y = (x1 + rng.normal(scale=1.5, size=n) > 0).astype(int)
    return pl.DataFrame({
        "x1": x1,
        "noise": rng.normal(size=n),
        "leak": y * 2.0 + rng.normal(scale=0.05, size=n),
        "nonmono": np.where(y == 1, rng.choice([-1, 1], n) * rng.uniform(3, 4, n), rng.uniform(-1, 1, n)),
        "y": y,
    })


def test_iqr_flags_planted_outliers() -> None:
    res = detect_anomalies(_anom_df(), ["a", "b"], "IQR")
    assert res.mask[:5].all() and res.n_anomalies >= 5
    assert set(res.per_column["column"]) == {"a", "b"}


def test_zscore_flags_planted_outliers() -> None:
    res = detect_anomalies(_anom_df(), ["a", "b"], "Z-score", z_threshold=3.0)
    assert res.mask[:5].all() and res.scores is not None


def test_isolation_forest_flags_planted_outliers() -> None:
    res = detect_anomalies(_anom_df(), ["a", "b"], "Isolation Forest", contamination=0.02)
    assert res.mask[:5].sum() >= 4 and res.scores is not None


def test_lof_flags_planted_outliers() -> None:
    res = detect_anomalies(_anom_df(), ["a", "b"], "Local Outlier Factor", contamination=0.02)
    assert res.mask[:5].sum() >= 3


def test_non_numeric_column_is_rejected() -> None:
    df = pl.DataFrame({"s": ["x"] * 30, "a": [float(i) for i in range(30)]})
    with pytest.raises(AnomalyError):
        detect_anomalies(df, ["s"], "IQR")


def test_flag_column_is_null_for_unanalysed_rows() -> None:
    df = _anom_df()
    a = df["a"].to_list()
    for i in range(10, 20):
        a[i] = None
    df = df.with_columns(pl.Series("a", a, dtype=pl.Float64))
    res = detect_anomalies(df, ["a", "b"], "Isolation Forest")
    assert res.n_rows == 490 and res.n_total_rows == 500
    flagged = flag_column(df, res, "flag")
    assert flagged.height == 500 and flagged["flag"].null_count() == 10
    with pytest.raises(AnomalyError):
        flag_column(flagged, res, "flag")


def test_anomaly_rows_and_figures() -> None:
    df = _anom_df()
    res = detect_anomalies(df, ["a", "b"], "Isolation Forest", contamination=0.02)
    rows = anomaly_rows(df, res)
    assert rows.height == res.n_anomalies and "anomaly_score" in rows.columns
    assert len(scatter_fig(res, "a", "b").data) == 2
    assert projection_fig(res).data


def test_leakage_finds_planted_leaks() -> None:
    report = assess_leakage(_leak_df(), "y")
    strong = {f.column for f in report.findings if f.strength == "strong"}
    assert {"leak", "nonmono"} <= strong
    flagged = {f.column for f in report.findings if f.strength in ("strong", "moderate")}
    assert "x1" not in flagged and "noise" not in flagged
    assert report.counts()["strong"] >= 2


def test_leakage_name_hint() -> None:
    rng = np.random.default_rng(5)
    df = _leak_df().with_columns(pl.Series("outcome_code", rng.integers(0, 5, 600)))
    report = assess_leakage(df, "y")
    assert any(f.column == "outcome_code" and f.strength == "weak" for f in report.findings)


def test_contamination_detected_with_duplicates() -> None:
    base = _clf_df(200)
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0)
    run = train_models(pl.concat([base, base]), cfg, ["Logistic Regression"])
    info = check_contamination(run)
    assert info["share"] > 0.5 and info["level"] == "strong"


def test_no_contamination_for_unique_rows() -> None:
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0)
    run = train_models(_clf_df(300), cfg, ["Logistic Regression"])
    info = check_contamination(run)
    assert info["share"] == 0.0 and info["level"] == "ok"


def test_native_automl_and_notes() -> None:
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0)
    run = native_automl(_clf_df(), cfg)
    names = [r.name for r in run.results]
    assert "Baseline" in names and "Random Forest" in names and len(names) >= 6
    assert "best" in leaderboard_notes(run)[0].lower()


def test_flaml_adds_result() -> None:
    pytest.importorskip("flaml")
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0)
    run = train_models(_clf_df(), cfg, ["Logistic Regression"])
    res = run_flaml(run, time_budget=5)
    assert res.ok and res.name.startswith("FLAML (") and res.test["F1"] > 0.6
    assert res.name in [r.name for r in run.results]