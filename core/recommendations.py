"""Rule-based "What should I do?" advice (UI-free).

Every suggestion is triggered by something that was actually computed
(a count, a share, a score). Nothing here is generic filler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl

from core.data_quality import imbalance_level
from core.model_training import best_result, detect_task

PRIORITY = {"high": 0, "medium": 1, "low": 2}
PRE = "views/preprocessing.py"
ML = "views/machine_learning.py"


@dataclass(frozen=True)
class Recommendation:
    priority: str            # high | medium | low
    title: str
    why: str
    page: str
    label: str


def _names(items: list[str], limit: int = 4) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _from_health(h: Any, state: Mapping[str, Any], recs: list[Recommendation]) -> None:
    cols = h.columns
    if h.duplicate_rows > 0:
        pct = h.duplicate_rows / max(h.n_rows, 1) * 100
        recs.append(Recommendation(
            "medium", f"Review {h.duplicate_rows:,} duplicate rows ({pct:.1f}%)",
            "Exact repeats can make model scores look better than they are, because the same row may land in "
            "both the training and the test data. Remove them only if they are true repeats.", PRE, "Go to Preprocessing"))

    mostly = cols.filter(pl.col("missing_pct") >= 50)["column"].to_list()
    if mostly:
        recs.append(Recommendation(
            "medium", f"{len(mostly)} column(s) are more than half empty: {_names(mostly)}",
            "Columns that are mostly empty rarely help a model. Consider dropping them.", PRE, "Go to Preprocessing"))
    partly = cols.filter((pl.col("missing_pct") > 0) & (pl.col("missing_pct") < 50))["column"].to_list()
    if partly:
        recs.append(Recommendation(
            "low", f"Missing values in {len(partly)} column(s): {_names(partly)}",
            "Machine Learning fills them using training rows only, so you can leave them. "
            "Fill them in Preprocessing only if you need a complete dataset.", PRE, "Go to Preprocessing"))
    odd = cols.filter(pl.col("suspicious") != "")["column"].to_list()
    if odd:
        recs.append(Recommendation(
            "medium", f"Suspicious values in {len(odd)} column(s): {_names(odd)}",
            "Placeholder text such as '?' or codes such as -999 may be missing values in disguise. "
            "Check them, then convert or fill.", PRE, "Go to Preprocessing"))
    const = cols.filter(pl.col("constant"))["column"].to_list()
    if const:
        recs.append(Recommendation("low", f"{len(const)} constant column(s): {_names(const)}",
                                   "A column with one value carries no information. Models leave these out automatically.",
                                   PRE, "Go to Preprocessing"))
    skewed = cols.filter(pl.col("skew").abs() > 2)["column"].to_list()
    if skewed:
        recs.append(Recommendation(
            "low", f"{len(skewed)} highly skewed variable(s): {_names(skewed)}",
            "A log transform (Preprocessing, Transform tab) often helps linear models. Tree-based models do not need it.",
            PRE, "Go to Preprocessing"))
    outl = cols.filter(pl.col("outlier_pct") > 5)["column"].to_list()
    if outl:
        recs.append(Recommendation(
            "low", f"More than 5% outliers in {len(outl)} column(s): {_names(outl)}",
            "Inspect them before deciding anything. Nothing is removed automatically.",
            "views/anomaly.py", "Go to Anomaly Detection"))
    if h.correlated_pairs.height:
        a, b, c = h.correlated_pairs.row(0)
        recs.append(Recommendation(
            "low", f"{h.correlated_pairs.height} highly correlated pair(s), for example {a} and {b} ({c:.2f})",
            "Near-duplicate columns can destabilise linear models. Keeping one of each pair is usually enough.",
            PRE, "Go to Preprocessing"))
    dates = cols.filter(pl.col("kind") == "datetime")["column"].to_list()
    if dates:
        recs.append(Recommendation(
            "low", f"Date column(s) found: {_names(dates)}",
            "Models skip date columns. Turn them into features (year, month, weekday) in Preprocessing, "
            "Feature engineering tab.", PRE, "Go to Preprocessing"))

    if h.target is None:
        recs.append(Recommendation(
            "medium", "Choose a target column",
            "Leakage and class-imbalance checks, and the models, need to know what you want to predict.",
            "views/data_health.py", "Go to Data Health"))
        return
    strong = [c.column for c in h.leakage if c.strength == "strong"]
    if strong:
        done = "leakage_report" in state
        recs.append(Recommendation(
            "medium" if done else "high", f"Investigate possible leakage in {_names(strong)}",
            "These columns may reveal the answer to the model. They can inflate scores, so check whether they "
            "are really known at prediction time before training.", "views/leakage.py", "Go to Leakage Check"))
    if h.class_distribution is not None:
        level, ratio = imbalance_level(h.class_distribution)
        if level == "severe":
            recs.append(Recommendation(
                "high", f"Severe class imbalance (ratio {ratio:.0f}:1)",
                "Plain accuracy is misleading here. Use class weights or resampling and judge models by F1 and ROC-AUC.",
                ML, "Go to Machine Learning"))
        elif level == "moderate":
            recs.append(Recommendation(
                "medium", f"Class imbalance detected (ratio {ratio:.1f}:1)",
                "Class weights may help. Compare F1 and ROC-AUC, not only accuracy.", ML, "Go to Machine Learning"))


def _from_models(state: Mapping[str, Any], ds: Any, health: Any, recs: list[Recommendation]) -> None:
    runs = [(label, state[key]) for label, key in (("Machine Learning", "ml_run"), ("AutoML", "automl_run"))
            if state.get(key) is not None]
    if not runs:
        target = health.target if health is not None else None
        if target and target in ds.df.columns:
            try:
                task = detect_task(ds.df[target])
            except Exception:
                task = "prediction"
            recs.append(Recommendation(
                "medium", f"Train models to predict '{target}'",
                f"The target looks like a {task} problem. A Baseline is added automatically, so you can see "
                "whether the models learn anything real.", ML, "Go to Machine Learning"))
        return

    leak_report = (state.get("leakage_report") or {}).get("report")
    strong_leak = {f.column for f in leak_report.findings if f.strength == "strong"} if leak_report is not None else set()
    sig = (ds.name, ds.df.height, ds.df.width)
    for label, run in runs:
        clf = run.task == "classification"
        key = "F1" if clf else "RMSE"
        best = best_result(run)
        if best is None or best.name == "Baseline" or best.test.get(key) is None:
            recs.append(Recommendation("high", f"No {label} model was trained successfully",
                                       "Open the leaderboard and read the Status column for the reasons.", ML, "Go to Machine Learning"))
            continue
        value = float(best.test[key])
        base = next((r for r in run.results if r.name == "Baseline" and r.ok and r.test.get(key) is not None), None)
        if base is not None:
            b = float(base.test[key])
            weak = (value - b <= 0.01) if clf else (b > 0 and (b - value) / b < 0.01)
            if weak:
                recs.append(Recommendation(
                    "high", f"{label}: the best model barely beats the Baseline",
                    f"Best test {key} is {value:.3f} against {b:.3f} for the Baseline. The features may carry little "
                    "signal, or the problem may need more data or better features.", ML, "Go to Machine Learning"))
        if run.signature and run.signature != sig:
            recs.append(Recommendation("medium", f"Retrain the {label} models",
                                       "The data changed after they were trained, so the results are out of date.",
                                       ML, "Go to Machine Learning"))
        used = sorted(strong_leak & set(run.num + run.cat))
        if used:
            recs.append(Recommendation(
                "high", f"{label}: models use columns flagged for possible leakage ({_names(used)})",
                "Their scores may be too optimistic. Drop the columns and train again to see the difference.",
                "views/leakage.py", "Go to Leakage Check"))
        if len(run.X_test) < 100:
            recs.append(Recommendation("low", f"{label}: the test set has only {len(run.X_test)} rows",
                                       "Scores from so few rows are uncertain. Treat small differences as noise.",
                                       ML, "Go to Machine Learning"))
    if not state.get("shap_cache"):
        recs.append(Recommendation("low", "Explain the best model",
                                   "SHAP and What-if show which columns drive the predictions.",
                                   "views/explainability.py", "Go to Explainability"))
    if "rp_files" not in state:
        recs.append(Recommendation("low", "Export a report",
                                   "Everything you have run can be written to an HTML report, Excel or JSON.",
                                   "views/reports.py", "Go to Reports & Export"))


def recommend(state: Mapping[str, Any]) -> list[Recommendation]:
    """Advice based on the results that exist in the session, most important first."""
    ds = state.get("dataset")
    if ds is None:
        return [Recommendation("high", "Load a dataset",
                               "Nothing can be analysed yet. Upload a file or open a demo dataset.",
                               "views/upload.py", "Go to Data Upload")]
    recs: list[Recommendation] = []
    health = (state.get("health") or {}).get("report")
    if health is None:
        recs.append(Recommendation(
            "high", "Check the data health",
            "A quick scan finds missing values, duplicates, constant columns and possible leakage before you "
            "spend time on models.", "views/data_health.py", "Go to Data Health"))
    else:
        _from_health(health, state, recs)
    _from_models(state, ds, health, recs)
    recs.sort(key=lambda r: PRIORITY[r.priority])
    return recs