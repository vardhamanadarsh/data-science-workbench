import numpy as np
import polars as pl
import pytest

from core.data_loader import load_dataset
from core.data_quality import assess_data_health
from core.drift import CompareError, compare_datasets, distribution_fig, psi_numeric
from core.model_training import TrainConfig, train_models
from core.recommendations import recommend


def _pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    n = 2000
    ref = pl.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n),
                        "c": rng.choice(["a", "b"], n).tolist(), "gone": rng.normal(size=n)})
    new = pl.DataFrame({"x": rng.normal(1.5, 1, n), "y": rng.normal(0, 1, n),
                        "c": rng.choice(["a", "b", "c"], n, p=[0.4, 0.4, 0.2]).tolist(),
                        "extra": rng.normal(size=n)})
    return ref, new


def test_compare_flags_drift_and_schema() -> None:
    ref, new = _pair()
    rep = compare_datasets(ref, new)
    d = rep.drift.set_index("column")
    assert d.loc["x", "level"] == "high" and d.loc["y", "level"] == "none" and d.loc["c", "level"] == "high"
    status = rep.schema.set_index("column")["status"]
    assert status["gone"] == "only in reference" and status["extra"] == "only in new" and status["x"] == "same"
    assert "c" in rep.categories["column"].tolist()
    assert any(level == "warn" for level, _ in rep.findings)


def test_missing_change_is_measured() -> None:
    ref, new = _pair()
    vals = new["y"].to_list()
    for i in range(400):
        vals[i] = None
    new = new.with_columns(pl.Series("y", vals, dtype=pl.Float64))
    rep = compare_datasets(ref, new)
    assert rep.missing.set_index("column").loc["y", "change (pp)"] == pytest.approx(20.0)


def test_psi_numeric() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=3000)
    assert psi_numeric(a, rng.normal(size=3000)) < 0.05
    assert psi_numeric(a, rng.normal(2, 1, 3000)) > 0.25
    assert psi_numeric(np.ones(100), np.ones(100)) is None


def test_distribution_figures() -> None:
    ref, new = _pair()
    assert len(distribution_fig(ref, new, "x").data) == 2
    assert len(distribution_fig(ref, new, "c").data) == 2
    with pytest.raises(CompareError):
        distribution_fig(ref, new, "gone")


def test_unusable_input_raises() -> None:
    ref, _ = _pair()
    with pytest.raises(CompareError):
        compare_datasets(pl.DataFrame(), ref)
    with pytest.raises(CompareError):
        compare_datasets(pl.DataFrame({"a": [1.0, 2.0]}), pl.DataFrame({"b": [1.0, 2.0]}))


def _state(tmp_path, df: pl.DataFrame, target: str | None = None, health: bool = True) -> dict:
    path = tmp_path / "d.csv"
    df.write_csv(path)
    ds = load_dataset(path)
    state: dict = {"dataset": ds}
    if health:
        state["health"] = {"report": assess_data_health(ds.df, target=target)}
    return state


def test_recommend_without_dataset_or_health(tmp_path) -> None:
    assert recommend({})[0].page == "views/upload.py"
    df = pl.DataFrame({"a": [float(i) for i in range(50)]})
    assert recommend(_state(tmp_path, df, health=False))[0].page == "views/data_health.py"


def test_recommend_duplicates_and_leakage(tmp_path) -> None:
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pl.DataFrame({"x": x, "leak": y * 2.0 + rng.normal(scale=0.01, size=n), "y": y})
    df = pl.concat([df, df.head(20)])
    recs = recommend(_state(tmp_path, df, "y"))
    assert any("duplicate" in r.title for r in recs)
    assert any(r.priority == "high" and "leakage" in r.title.lower() for r in recs)
    assert [r.priority for r in recs] == sorted((r.priority for r in recs), key=lambda p: {"high": 0, "medium": 1, "low": 2}[p])


def test_recommend_after_training(tmp_path) -> None:
    rng = np.random.default_rng(3)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    state = _state(tmp_path, pl.DataFrame({"x1": x1, "x2": x2, "y": y}), "y")
    ds = state["dataset"]
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0)
    run = train_models(ds.df, cfg, ["Logistic Regression"])
    run.signature = (ds.name, ds.df.height, ds.df.width)
    state["ml_run"] = run
    pages = [r.page for r in recommend(state)]
    assert "views/explainability.py" in pages and "views/reports.py" in pages