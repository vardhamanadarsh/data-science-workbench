import numpy as np
import polars as pl
import pytest

from config import settings
from core.experiments import (
    delete_experiment, experiments_table, list_experiments, load_experiment, save_experiment,
)
from core.model_diagnostics import calibration_fig, learning_curve_fig
from core.model_training import MLError, TrainConfig, best_result, train_models
from core.tuning import tune_model


def _clf_df(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def _run():
    cfg = TrainConfig(target="y", features=["x1", "x2"], task="classification", cv_folds=3)
    return train_models(_clf_df(), cfg, ["Logistic Regression"])


def test_tuning_adds_a_tuned_model() -> None:
    pytest.importorskip("optuna")
    run = _run()
    outcome = tune_model(run, "Logistic Regression", n_trials=4, timeout=120, folds=3)
    assert "C" in outcome.best_params
    assert outcome.n_trials_done == 4 and len(outcome.trials) == 4
    assert "Logistic Regression (tuned)" in [r.name for r in run.results]
    assert outcome.result.ok and outcome.result.test["F1"] > 0.7


def test_untunable_model_is_rejected() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pl.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.1, size=200)})
    cfg = TrainConfig(target="y", features=["x"], task="regression", cv_folds=0)
    run = train_models(df, cfg, ["Linear Regression"])
    with pytest.raises(MLError):
        tune_model(run, "Linear Regression", 3, 30, 3)


def test_experiment_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "experiments_dir", tmp_path)
    run = _run()
    res = best_result(run)
    exp_id = save_experiment(run, res, {"name": "demo.csv", "rows": 400, "columns": 3},
                             ["Fill missing (median)"], note="first")
    records = list_experiments()
    assert [r["id"] for r in records] == [exp_id]
    rec = load_experiment(exp_id)
    assert rec["model"] == "Logistic Regression" and rec["metrics"]["test"]["F1"] > 0.7
    assert "C" in rec["hyperparameters"] and rec["preprocessing_steps"] == ["Fill missing (median)"]
    assert "Test F1" in experiments_table(records).columns
    delete_experiment(exp_id)
    assert list_experiments() == []


def test_ids_are_unique_and_validated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "experiments_dir", tmp_path)
    run = _run()
    res = best_result(run)
    a = save_experiment(run, res, {"name": "d"}, [])
    b = save_experiment(run, res, {"name": "d"}, [])
    assert a != b
    with pytest.raises(ValueError):
        load_experiment("../secret")


def test_calibration_figure() -> None:
    run = _run()
    res = best_result(run)
    out = calibration_fig(run.y_test, res.scores)
    assert out is not None and 0 <= out[1] <= 1


def test_learning_curve_figure() -> None:
    run = _run()
    res = best_result(run)
    assert learning_curve_fig(run, res, folds=3).data