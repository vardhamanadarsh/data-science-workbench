"""Plain-English explanations of metrics and findings (UI-free).

Each explanation has a fixed meaning plus a reading that is computed from the
value you actually got. Nothing is generated or guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from core.drift import PSI_HIGH, PSI_MODERATE, level_from_psi
from core.model_training import best_result


class ExplainError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


@dataclass(frozen=True)
class Explanation:
    term: str
    meaning: str
    reading: str = ""     # what the actual value says ("" when no value was given)
    caution: str = ""


# term -> (meaning, caution)
TERMS: dict[str, tuple[str, str]] = {
    "Accuracy": ("The share of predictions that were correct.",
                 "Misleading when one class is much more common than the others. Compare with the Baseline."),
    "Precision": ("Of the rows the model labelled positive, the share that really were positive.",
                  "High precision with low recall means the model is careful but misses many cases."),
    "Recall": ("Of the rows that really were positive, the share the model found.",
               "High recall with low precision means the model catches most cases but raises many false alarms."),
    "F1": ("A single score that balances precision and recall. 1 is perfect, 0 is useless.",
           "It ignores true negatives. Always compare it with the Baseline."),
    "ROC-AUC": ("How well the model ranks positive rows above negative ones. 0.5 is random guessing, 1 is perfect.",
                "It says nothing about whether the predicted probabilities are well calibrated."),
    "RMSE": ("The typical size of the prediction error, in the same units as the target. Large errors count extra.",
             "Only meaningful against something: compare it with the Baseline and with the spread of the target."),
    "MAE": ("The average size of the prediction error, in the same units as the target.",
            "Less sensitive to a few large errors than RMSE."),
    "R2": ("The share of the target's variation that the model explains. 1 is perfect, 0 is no better than "
           "predicting the average.", "It can be negative on test data, which means worse than predicting the average."),
    "Baseline": ("A deliberately simple model (most common class, or the mean). It shows what you get with no learning.",
                 "If a real model barely beats it, the features may carry little signal."),
    "Cross-validation": ("Training and testing several times on different parts of the training rows, to check that a "
                         "score is stable.", "A large spread between folds means the score is unreliable."),
    "Health score": ("A 0 to 100 summary of data quality: missing values, duplicates, constant columns, outliers and so on.",
                     "It is a rough guide. Read the individual findings, not just the number."),
    "Leakage": ("A column that reveals the answer, or information you would not have when making a real prediction.",
                "Findings are only 'possible' leakage. Decide using what you know about how the data was collected."),
    "Class imbalance ratio": ("How many times more common the biggest class is than the smallest.",
                              "At high ratios, accuracy hides failures on the rare class."),
    "PSI": ("Population Stability Index: how much a column's distribution moved between two datasets.",
            f"Below {PSI_MODERATE} small, up to {PSI_HIGH} moderate, above that large. These are rules of thumb."),
    "KS p-value": ("From the Kolmogorov-Smirnov test: how surprising the difference between two distributions would be "
                   "if they were really the same.", "With many rows even tiny shifts give small p-values. Use PSI for size."),
    "SHAP": ("A way to split each prediction into the contribution of every column.",
             "It describes how the model behaves, not what causes things in the real world."),
    "Permutation importance": ("How much the test score drops when one column is shuffled. A bigger drop means the model "
                               "relied on that column more.", "Correlated columns can share or hide importance."),
    "Anomaly rate": ("The share of rows the method marked as unusual.",
                     "Unusual is not the same as wrong. Nothing is removed automatically."),
    "Disparate impact": ("The ratio of the lowest group's rate of favourable outcomes to the highest group's.",
                         "The four-fifths rule treats a ratio below 0.8 as a warning sign, not as proof of unfairness."),
    "Fairness check": ("A comparison of model results across groups defined by one column.",
                       "Differences are possible disparities. They need human judgement."),
}


def _band(v: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if v < cut:
            return label
    return labels[-1]


def _classification_vs_baseline(v: float, b: float | None) -> str:
    if b is None:
        return ""
    gap = v - b
    if gap > 0.01:
        return f" The Baseline scored {b:.3f}, so the model adds {gap:.3f}."
    return f" The Baseline scored {b:.3f}, so the model does not clearly beat simple guessing."


def _r_f1(v: float, c: Mapping[str, Any]) -> str:
    band = _band(v, [0.5, 0.7, 0.9], ["weak", "modest", "good", "very high"])
    text = f"Your value is {v:.3f}, which is {band}." + _classification_vs_baseline(v, c.get("baseline"))
    return text + (" Scores this high deserve a leakage check." if v >= 0.9 else "")


def _r_auc(v: float, c: Mapping[str, Any]) -> str:
    band = _band(v, [0.55, 0.7, 0.8, 0.9], ["close to random guessing (0.5)", "weak", "fair", "good", "very strong"])
    text = f"Your value is {v:.3f}, which is {band}."
    return text + (" Scores this high deserve a leakage check." if v >= 0.95 else "")


def _r_accuracy(v: float, c: Mapping[str, Any]) -> str:
    text = f"Your model was right {v:.1%} of the time."
    m = c.get("majority_share")
    if m is not None:
        text += (f" Always guessing the most common class would already give {m:.1%}, "
                 + ("so the model adds little." if v - m < 0.01 else f"so the model adds {v - m:.1%}."))
    return text


def _r_precision(v: float, c: Mapping[str, Any]) -> str:
    return f"When the model says positive, it is right {v:.1%} of the time."


def _r_recall(v: float, c: Mapping[str, Any]) -> str:
    return f"The model finds {v:.1%} of the rows that are really positive."


def _r_rmse(v: float, c: Mapping[str, Any]) -> str:
    text = f"Your predictions are typically off by about {v:,.4g} (in the units of the target)."
    b = c.get("baseline")
    if b is not None and b > 0:
        red = (b - v) / b * 100
        text += (f" The Baseline's RMSE is {b:,.4g}, so the model reduces the error by {red:.0f}%." if red > 1
                 else f" The Baseline's RMSE is {b:,.4g}, so the model does not clearly beat predicting the mean.")
    return text


def _r_mae(v: float, c: Mapping[str, Any]) -> str:
    return f"On average the prediction is off by {v:,.4g} (in the units of the target)."


def _r_r2(v: float, c: Mapping[str, Any]) -> str:
    band = _band(v, [0, 0.3, 0.6, 0.8], ["worse than predicting the average", "weak", "moderate", "good", "strong"])
    text = f"Your value is {v:.3f}, which is {band}."
    return text + (f" The model explains about {v:.0%} of the variation." if v >= 0 else "")


def _r_health(v: float, c: Mapping[str, Any]) -> str:
    label = c.get("label")
    return f"Your data scored {v:.0f}/100" + (f" ({label})." if label else ".") + \
        " Open Data Health to see which checks lost points."


def _r_leakage(v: float, c: Mapping[str, Any]) -> str:
    n = int(v)
    return ("No column showed strong signs of leakage. This is not a proof." if n == 0 else
            f"{n} column(s) showed strong signs of possible leakage. Check whether each one is really known "
            "at prediction time.")


def _r_imbalance(v: float, c: Mapping[str, Any]) -> str:
    band = _band(v, [3, 10], ["mild", "moderate", "severe"])
    return f"The biggest class is {v:.1f} times the smallest, which is {band} imbalance."


def _r_psi(v: float, c: Mapping[str, Any]) -> str:
    level = level_from_psi(v)
    text = {"none": "a small shift", "moderate": "a moderate shift", "high": "a large shift"}.get(level, "not tested")
    return f"Your value is {v:.3f}, which is {text}."


def _r_ks(v: float, c: Mapping[str, Any]) -> str:
    if v < 0.05:
        return (f"p = {v:.3g}. The difference is unlikely to be chance, but with many rows even a tiny shift "
                "gives a small p-value. Check PSI for the size of the shift.")
    return f"p = {v:.3g}. There is no clear evidence that the distributions differ."


def _r_anomaly(v: float, c: Mapping[str, Any]) -> str:
    return f"{v:.2f}% of the analysed rows were marked unusual. Nothing was removed."


def _r_disparate(v: float, c: Mapping[str, Any]) -> str:
    return (f"Your ratio is {v:.2f}, below the 0.8 warning line." if v < 0.8
            else f"Your ratio is {v:.2f}, above the 0.8 warning line.")


def _r_top(v: float, c: Mapping[str, Any]) -> str:
    top = c.get("top")
    return f"For your model, the most influential columns are {', '.join(map(str, top))}." if top else ""


_READERS: dict[str, Callable[[float, Mapping[str, Any]], str]] = {
    "F1": _r_f1, "ROC-AUC": _r_auc, "Accuracy": _r_accuracy, "Precision": _r_precision, "Recall": _r_recall,
    "RMSE": _r_rmse, "MAE": _r_mae, "R2": _r_r2, "Health score": _r_health, "Leakage": _r_leakage,
    "Class imbalance ratio": _r_imbalance, "PSI": _r_psi, "KS p-value": _r_ks, "Anomaly rate": _r_anomaly,
    "Disparate impact": _r_disparate,
}


def _lookup(term: str) -> str:
    key = term.strip().lower()
    for name in TERMS:
        if name.lower() == key:
            return name
    raise ExplainError(f"There is no explanation for '{term}'.", "Pick one of: " + ", ".join(TERMS))


def explain(term: str, value: float | None = None, **context: Any) -> Explanation:
    """Explain a term. If a value is given, also say what that value means."""
    name = _lookup(term)
    meaning, caution = TERMS[name]
    reading = ""
    if value is not None:
        reader = _READERS.get(name)
        reading = reader(float(value), context) if reader else ""
    elif context.get("top"):
        reading = _r_top(0.0, context)
    return Explanation(name, meaning, reading, caution)


def _model_items(label: str, run: Any) -> list[tuple[str, Explanation]]:
    best = best_result(run)
    if best is None or best.name == "Baseline":
        return []
    section = f"{label}: {best.name} (test set)"
    base = next((r for r in run.results if r.name == "Baseline" and r.ok), None)
    bt = base.test if base is not None else {}
    out: list[tuple[str, Explanation]] = []
    if run.task == "classification":
        majority = None
        if base is not None and bt.get("Accuracy") is not None:
            majority = float(bt["Accuracy"])
        for term, extra in (("F1", {"baseline": bt.get("F1")}), ("ROC-AUC", {}),
                            ("Accuracy", {"majority_share": majority}), ("Precision", {}), ("Recall", {})):
            v = best.test.get(term)
            if v is not None:
                out.append((section, explain(term, float(v), **{k: x for k, x in extra.items() if x is not None})))
    else:
        for term, extra in (("RMSE", {"baseline": bt.get("RMSE")}), ("R2", {}), ("MAE", {})):
            v = best.test.get(term)
            if v is not None:
                out.append((section, explain(term, float(v), **{k: x for k, x in extra.items() if x is not None})))
    return out


def explain_all(state: Mapping[str, Any]) -> list[tuple[str, Explanation]]:
    """Explanations for every result that exists in the session, as (section, Explanation)."""
    out: list[tuple[str, Explanation]] = []
    h = (state.get("health") or {}).get("report")
    if h is not None:
        out.append(("Data", explain("Health score", h.score, label=h.label)))
        if h.class_distribution is not None:
            from core.data_quality import imbalance_level
            _, ratio = imbalance_level(h.class_distribution)
            out.append(("Data", explain("Class imbalance ratio", ratio)))
    leak = (state.get("leakage_report") or {}).get("report")
    if leak is not None:
        strong = len({f.column for f in leak.findings if f.strength == "strong"})
        out.append(("Leakage check", explain("Leakage", strong)))
    a = state.get("anomaly_result")
    if a is not None:
        out.append(("Anomaly detection", explain("Anomaly rate", a.pct)))
    for label, key in (("Machine Learning", "ml_run"), ("AutoML", "automl_run")):
        run = state.get(key)
        if run is not None:
            out.extend(_model_items(label, run))
            out.append((f"{label}: baseline", explain("Baseline")))
    cmp_ = (state.get("compare_result") or {}).get("report")
    if cmp_ is not None and cmp_.drift["PSI"].notna().any():
        top = cmp_.drift.dropna(subset=["PSI"]).iloc[0]
        out.append((f"Drift: largest shift is '{top['column']}'", explain("PSI", float(top["PSI"]))))
    fair = state.get("fairness_result")
    if fair is not None:
        warns = sum(1 for lvl, _ in fair[1].findings if lvl == "warn")
        out.append(("Fairness", Explanation(
            "Fairness check", TERMS["Fairness check"][0],
            f"The check on '{fair[1].column}' found {warns} possible disparity(ies).", TERMS["Fairness check"][1])))
    return out