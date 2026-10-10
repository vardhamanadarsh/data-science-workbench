import numpy as np
import polars as pl
import pytest

from core.explain import (
    compute_shap, importance_fig, importance_table, local_contributions, summary_fig, waterfall_fig,
)
from core.fairness import fairness_report, group_labels
from core.model_training import MLError, TrainConfig, train_models
from core.whatif import (
    apply_changes, feature_domains, find_counterfactuals, predict_frame, prediction_summary, typical_row,
)


def _df(n: int = 500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    c = rng.choice(["a", "b"], n)
    y = (2 * x1 + x2 + (c == "a") + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "c": c.tolist(), "y": y})


def _run(models=("Logistic Regression", "Random Forest")):
    cfg = TrainConfig(target="y", features=["x1", "x2", "c"], task="classification", cv_folds=0)
    return train_models(_df(), cfg, list(models))


def _res(run, name):
    return next(r for r in run.results if r.name == name)


def test_shap_tree_values_add_up() -> None:
    pytest.importorskip("shap")
    run = _run()
    res = _res(run, "Random Forest")
    sr = compute_shap(run, res, max_rows=80)
    assert sr.values.shape == (80, len(sr.names))
    proba = res.pipeline.predict_proba(sr.X_raw)[:, 1]
    assert np.allclose(sr.base + sr.values.sum(axis=1), proba, atol=0.02)
    assert importance_table(sr, run)["feature"].iloc[0] == "x1"
    assert importance_fig(sr, run).data and summary_fig(sr).data
    assert waterfall_fig(local_contributions(sr, run, 0), sr.base, sr.unit).data


def test_shap_linear_values_add_up() -> None:
    pytest.importorskip("shap")
    run = _run()
    res = _res(run, "Logistic Regression")
    sr = compute_shap(run, res, max_rows=80)
    assert sr.unit == "log-odds"
    assert np.allclose(sr.base + sr.values.sum(axis=1), res.pipeline.decision_function(sr.X_raw), atol=1e-3)


def test_feature_domains_and_typical_row() -> None:
    run = _run(("Logistic Regression",))
    domains = feature_domains(run)
    assert domains["x1"].kind == "num" and set(domains["c"].categories) == {"a", "b"}
    row = typical_row(run, domains)
    assert list(row.columns) == list(run.X_train.columns) and len(row) == 1


def test_prediction_helpers() -> None:
    run = _run(("Logistic Regression",))
    res = _res(run, "Logistic Regression")
    proba, has = predict_frame(run, res, run.X_test)
    assert has and proba.shape == (len(run.X_test), 2) and np.allclose(proba.sum(axis=1), 1)
    assert "probability" in prediction_summary(run, res, run.X_test.iloc[[0]])


def test_counterfactuals_flip_the_prediction() -> None:
    run = _run(("Logistic Regression",))
    res = _res(run, "Logistic Regression")
    proba, _ = predict_frame(run, res, run.X_test)
    candidates = np.flatnonzero(proba.argmax(axis=1) == 0)
    i = candidates[np.argmax(proba[candidates, 1])]
    row = run.X_test.iloc[[i]].reset_index(drop=True)
    out = find_counterfactuals(run, res, row, desired=1, mutable=["x1", "x2", "c"])
    assert out.items
    for cf in out.items:
        new_proba, _ = predict_frame(run, res, apply_changes(row, cf.changes))
        assert int(new_proba[0].argmax()) == 1


def test_counterfactual_for_current_class_is_rejected() -> None:
    run = _run(("Logistic Regression",))
    res = _res(run, "Logistic Regression")
    row = run.X_test.iloc[[0]].reset_index(drop=True)
    current = int(predict_frame(run, res, row)[0][0].argmax())
    with pytest.raises(MLError):
        find_counterfactuals(run, res, row, desired=current, mutable=["x1"])


def test_fairness_selection_rates() -> None:
    run = _run(("Logistic Regression",))
    res = _res(run, "Logistic Regression")
    out = fairness_report(run, res, "c", min_group=10)
    assert sorted(out.table["group"]) == ["a", "b"]
    labels = run.X_test["c"].fillna("(missing)").astype(str).to_numpy()
    for g in ("a", "b"):
        expected = float((np.asarray(res.y_pred)[labels == g] == 1).mean())
        assert out.table.set_index("group").loc[g, "selection_rate"] == pytest.approx(expected)
    assert "Selection rate gap" in out.gaps and out.findings


def test_fairness_numeric_column_is_binned() -> None:
    run = _run(("Logistic Regression",))
    res = _res(run, "Logistic Regression")
    out = fairness_report(run, res, "x1", min_group=10, bins=4)
    assert out.binned and 2 <= len(out.table) <= 4
    labels, binned = group_labels(run.X_test["x1"], 3)
    assert binned and labels.nunique() <= 3
    with pytest.raises(MLError):
        fairness_report(run, res, "not_a_column")


def test_fairness_regression() -> None:
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    g = rng.choice(["u", "v"], n)
    df = pl.DataFrame({"x": x, "g": g.tolist(), "y": 3 * x + rng.normal(scale=0.2, size=n)})
    cfg = TrainConfig(target="y", features=["x", "g"], task="regression", cv_folds=0)
    run = train_models(df, cfg, ["Linear Regression"])
    out = fairness_report(run, _res(run, "Linear Regression"), "g", min_group=10)
    assert {"mae", "rmse", "bias"} <= set(out.table.columns) and "MAE gap" in out.gaps