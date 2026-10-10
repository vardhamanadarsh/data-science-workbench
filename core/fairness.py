"""Group comparison of model performance (UI-free, no extra packages).

Findings are only POSSIBLE disparities. Differences can reflect real differences
in the data, and whether they are acceptable is a decision for people.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score

from core.model_training import MLError, ModelResult, TrainingRun

RATIO_RULE = 0.8          # four-fifths rule of thumb
RATE_GAP = 0.10
ACCURACY_GAP = 0.05
MAE_RATIO = 1.25


@dataclass
class FairnessResult:
    column: str
    task: str
    binned: bool
    min_group: int
    table: pd.DataFrame
    eligible: list[str]
    gaps: dict[str, float] = field(default_factory=dict)
    findings: list[tuple[str, str]] = field(default_factory=list)   # (level, text)


def group_labels(s: pd.Series, bins: int = 4) -> tuple[pd.Series, bool]:
    """Group labels for a column. Numeric columns with many values are cut into quantile bins."""
    s = s.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
        try:
            labels = pd.qcut(s, q=bins, duplicates="drop").astype(str)
            return labels.where(s.notna(), "(missing)"), True
        except ValueError:
            pass
    return s.astype(str).where(s.notna(), "(missing)"), False


def _rate(mask: np.ndarray, hit: np.ndarray) -> float:
    return float(hit[mask].mean()) if mask.any() else float("nan")


def _spread(table: pd.DataFrame, col: str) -> tuple[float, float, str, str] | None:
    s = table.set_index("group")[col].dropna()
    if len(s) < 2:
        return None
    return float(s.min()), float(s.max()), str(s.idxmin()), str(s.idxmax())


def fairness_report(run: TrainingRun, res: ModelResult, column: str,
                    min_group: int = 30, bins: int = 4) -> FairnessResult:
    if res.y_pred is None:
        raise MLError("This model has no test predictions.")
    if column not in run.X_test.columns:
        raise MLError(f"'{column}' is not one of the model's features.",
                      "Only columns used as features can be audited here.")
    raw = run.X_test[column].reset_index(drop=True)
    labels, binned = group_labels(raw, bins)
    y, pred = np.asarray(run.y_test), np.asarray(res.y_pred)
    clf = run.task == "classification"
    k = len(run.classes)

    groups = list(labels.unique())
    if binned:
        groups.sort(key=lambda g: float(raw[labels == g].min()) if (labels == g).any() and g != "(missing)" else 1e30)
    else:
        groups.sort()

    rows = []
    for g in groups:
        m = (labels == g).to_numpy()
        yt, yp = y[m], pred[m]
        row: dict = {"group": g, "n": int(m.sum()), "share": float(m.mean())}
        if clf and k == 2:
            row.update(
                actual_positive_rate=float(yt.mean()),
                selection_rate=float((yp == 1).mean()),
                accuracy=float((yt == yp).mean()),
                tpr=_rate(yt == 1, yp == 1),
                fpr=_rate(yt == 0, yp == 1),
                precision=_rate(yp == 1, yt == 1),
            )
        elif clf:
            row.update(accuracy=float((yt == yp).mean()),
                       f1_weighted=float(f1_score(yt, yp, average="weighted", zero_division=0)))
        else:
            err = yp - yt
            row.update(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                       bias=float(err.mean()))
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table.loc[table["n"] >= min_group, "group"].tolist()
    out = FairnessResult(column, run.task, binned, min_group, table, eligible)

    if len(eligible) < 2:
        out.findings.append(("info", f"Fewer than two groups have at least {min_group} test rows, so group "
                                     "differences cannot be judged reliably."))
        return out
    sub = table[table["group"].isin(eligible)]

    def add_gap(name: str, col: str) -> tuple[float, float, str, str] | None:
        sp = _spread(sub, col)
        if sp is not None:
            out.gaps[name] = sp[1] - sp[0]
        return sp

    if clf and k == 2:
        sp = add_gap("Selection rate gap", "selection_rate")
        if sp is not None:
            lo, hi, g_lo, g_hi = sp
            ratio = lo / hi if hi > 0 else 1.0
            out.gaps["Selection rate ratio"] = ratio
            if ratio < RATIO_RULE:
                out.findings.append(("warn", f"Possible disparity: '{g_lo}' is predicted positive {lo:.1%} of the "
                                             f"time and '{g_hi}' {hi:.1%} (ratio {ratio:.2f}, below the {RATIO_RULE} rule "
                                             "of thumb). The actual positive rates may also differ between these groups."))
            else:
                out.findings.append(("ok", f"Selection rates are similar across groups (ratio {ratio:.2f})."))
        for name, col, label in (("True positive rate gap", "tpr", "recall"),
                                 ("False positive rate gap", "fpr", "false positive rate")):
            sp = add_gap(name, col)
            if sp is not None and sp[1] - sp[0] > RATE_GAP:
                out.findings.append(("warn", f"Possible disparity in {label}: {sp[0]:.1%} for '{sp[2]}' vs "
                                             f"{sp[1]:.1%} for '{sp[3]}'."))
        sp = add_gap("Accuracy gap", "accuracy")
        if sp is not None and sp[1] - sp[0] > ACCURACY_GAP:
            out.findings.append(("warn", f"Accuracy differs by {sp[1] - sp[0]:.1%} between '{sp[2]}' ({sp[0]:.1%}) "
                                         f"and '{sp[3]}' ({sp[1]:.1%})."))
    elif clf:
        sp = add_gap("Accuracy gap", "accuracy")
        if sp is not None:
            level = "warn" if sp[1] - sp[0] > ACCURACY_GAP else "ok"
            out.findings.append((level, f"Accuracy ranges from {sp[0]:.1%} ('{sp[2]}') to {sp[1]:.1%} ('{sp[3]}')."))
    else:
        sp = add_gap("MAE gap", "mae")
        if sp is not None:
            ratio = sp[1] / sp[0] if sp[0] > 0 else 1.0
            out.gaps["MAE ratio"] = ratio
            level = "warn" if ratio > MAE_RATIO else "ok"
            out.findings.append((level, f"Mean absolute error is {sp[0]:,.3f} for '{sp[2]}' and {sp[1]:,.3f} for "
                                        f"'{sp[3]}' (ratio {ratio:.2f})."))
        sp = add_gap("Bias gap", "bias")
        spread_y = float(np.std(y)) or 1.0
        if sp is not None and sp[1] - sp[0] > 0.1 * spread_y:
            out.findings.append(("warn", f"The model over- or under-predicts differently by group: average error "
                                         f"{sp[0]:+,.3f} for '{sp[2]}' vs {sp[1]:+,.3f} for '{sp[3]}'."))
    return out


def fairness_fig(result: FairnessResult, metric: str) -> go.Figure:
    t = result.table
    fig = px.bar(t, x="group", y=metric, text=t[metric].round(3), hover_data=["n"])
    fig.update_traces(marker_color=["#1F77B4" if g in result.eligible else "#BBBBBB" for g in t["group"]])
    fig.update_layout(template="plotly_white", xaxis_title=result.column, yaxis_title=metric.replace("_", " "))
    return fig