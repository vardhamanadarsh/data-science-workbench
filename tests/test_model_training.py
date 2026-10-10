import numpy as np
import polars as pl
import pytest

from core.model_evaluation import confusion_matrix_fig, permutation_table, roc_fig
from core.model_training import (
    MLError, TrainConfig, best_result, eligible_features, results_table, train_models,
)


def _clf_df(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def _cfg(**kw) -> TrainConfig:
    base = dict(target="y", features=["x1", "x2"], task="classification", cv_folds=3)
    base.update(kw)
    return TrainConfig(**base)


def test_classification_run() -> None:
    run = train_models(_clf_df(), _cfg(), ["Logistic Regression", "Decision Tree"])
    assert [r.name for r in run.results][0] == "Baseline"
    assert all(r.ok for r in run.results)
    best = best_result(run)
    assert best is not None and best.name != "Baseline" and best.test["F1"] > 0.7
    assert "F1" in best.cv
    assert "Test F1" in results_table(run).columns


def test_scaler_is_fitted_on_training_rows_only() -> None:
    run = train_models(_clf_df(), _cfg(), ["Logistic Regression"])
    res = next(r for r in run.results if r.name == "Logistic Regression")
    scaler = res.pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_, run.X_train[run.num].mean().to_numpy())
    assert not np.allclose(scaler.mean_, run.X_test[run.num].mean().to_numpy())


def test_regression_run() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pl.DataFrame({"x": x, "y": 3 * x + rng.normal(scale=0.1, size=300)})
    cfg = TrainConfig(target="y", features=["x"], task="regression", cv_folds=3)
    run = train_models(df, cfg, ["Linear Regression"])
    best = best_result(run)
    assert best is not None and best.test["R2"] > 0.9


def test_missing_values_and_categories_are_handled() -> None:
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    x_missing = [None if i % 10 == 0 else float(v) for i, v in enumerate(x)]
    cat = [None if i % 7 == 0 else ("a" if i % 2 else "b") for i in range(n)]
    df = pl.DataFrame({"x": x_missing, "c": cat, "y": y})
    cfg = TrainConfig(target="y", features=["x", "c"], task="classification", cv_folds=3)
    run = train_models(df, cfg, ["Random Forest"])
    assert all(r.ok for r in run.results)


def test_class_weight_option() -> None:
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 1.3).astype(int)
    df = pl.DataFrame({"x": x, "y": y})
    cfg = TrainConfig(target="y", features=["x"], task="classification", cv_folds=3,
                      imbalance="class_weight")
    run = train_models(df, cfg, ["Logistic Regression", "KNN"])
    assert all(r.ok for r in run.results)
    knn = next(r for r in run.results if r.name == "KNN")
    assert any("not supported" in n for n in knn.notes)


def test_oversampling_runs_inside_pipeline() -> None:
    pytest.importorskip("imblearn")
    cfg = _cfg(imbalance="oversample")
    run = train_models(_clf_df(), cfg, ["Logistic Regression"])
    assert all(r.ok for r in run.results)


def test_class_with_one_row_raises() -> None:
    df = pl.DataFrame({"x": [float(i) for i in range(60)], "y": [0] * 59 + [1]})
    cfg = TrainConfig(target="y", features=["x"], task="classification")
    with pytest.raises(MLError):
        train_models(df, cfg, ["Logistic Regression"])


def test_eligible_features_skips_ids_and_constants() -> None:
    n = 200
    df = pl.DataFrame({
        "customer_id": list(range(n)), "const": [1] * n,
        "x": [float(i % 17) for i in range(n)], "y": [i % 2 for i in range(n)],
    })
    ok, skipped = eligible_features(df, "y")
    assert ok == ["x"] and set(skipped) == {"customer_id", "const"}


def test_time_split_keeps_order() -> None:
    n = 200
    df = pl.DataFrame({"x": [float(i) for i in range(n)], "y": [float(i) for i in range(n)]})
    cfg = TrainConfig(target="y", features=["x"], task="regression", cv_folds=0, time_split=True)
    run = train_models(df, cfg, ["Linear Regression"])
    assert run.y_test.min() > run.y_train.max()


def test_evaluation_figures_and_importance() -> None:
    run = train_models(_clf_df(), _cfg(), ["Logistic Regression"])
    best = best_result(run)
    assert best is not None
    assert confusion_matrix_fig(run.y_test, best.y_pred, run.classes).data
    assert roc_fig(run.y_test, best.scores, run.classes) is not None
    table = permutation_table(run, best, repeats=3)
    assert set(table["feature"]) == {"x1", "x2"}