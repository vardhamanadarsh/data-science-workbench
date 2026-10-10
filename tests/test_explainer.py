import numpy as np
import polars as pl
import pytest

from core.data_loader import load_dataset
from core.data_quality import assess_data_health
from core.explainer import TERMS, ExplainError, explain, explain_all
from core.model_training import TrainConfig, train_models


def test_every_term_has_text() -> None:
    for term in TERMS:
        e = explain(term)
        assert e.meaning and e.caution and e.reading == ""


def test_reading_uses_value() -> None:
    assert "guessing" in explain("ROC-AUC", 0.5).reading
    assert "very strong" in explain("ROC-AUC", 0.95).reading
    assert "worse than predicting the average" in explain("R2", -0.2).reading


def test_regression_baseline_comparison() -> None:
    assert "50%" in explain("RMSE", 5.0, baseline=10.0).reading


def test_unknown_term() -> None:
    with pytest.raises(ExplainError):
        explain("nonsense")


def test_case_insensitive() -> None:
    assert explain("roc-auc").term == "ROC-AUC"


def test_explain_all_from_state(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    path = tmp_path / "d.csv"
    pl.DataFrame({"x1": x1, "x2": x2, "y": y}).write_csv(path)
    ds = load_dataset(path)
    run = train_models(ds.df, TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=0),
                       ["Logistic Regression"])
    state = {"dataset": ds, "health": {"report": assess_data_health(ds.df, target="y")}, "ml_run": run}
    terms = [e.term for _, e in explain_all(state)]
    assert "Health score" in terms and "F1" in terms and "Baseline" in terms
    assert explain_all({}) == []