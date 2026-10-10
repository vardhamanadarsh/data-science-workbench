"""Report context and automatic Key Findings (UI-free).

Every sentence is computed from results that exist. Nothing is invented: when an
analysis was not run, it is simply absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from loguru import logger

from core.explain import importance_table
from core.model_training import TrainingRun, best_result


@dataclass
class ReportContext:
    title: str = "Data analysis report"
    dataset: Any = None
    health: Any = None
    eda: Any = None
    leakage: Any = None
    anomaly: Any = None
    runs: list[tuple[str, TrainingRun]] = field(default_factory=list)
    shap: list[dict[str, Any]] = field(default_factory=list)
    perm: list[dict[str, Any]] = field(default_factory=list)
    fairness: Any = None
    steps: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    @property
    def is_empty(self) -> bool:
        singles = (self.dataset, self.health, self.eda, self.leakage, self.anomaly, self.fairness)
        return all(x is None for x in singles) and not self.runs


def build_context(state: Mapping[str, Any], title: str = "Data analysis report",
                  notes: dict[str, str] | None = None) -> ReportContext:
    """Collect everything that exists in the session (works with st.session_state or a dict)."""
    ds = state.get("dataset")
    ctx = ReportContext(
        title=title or "Data analysis report",
        dataset=ds,
        notes={k: v.strip() for k, v in (notes or {}).items() if v and v.strip()},
    )
    ctx.health = (state.get("health") or {}).get("report")
    ctx.eda = (state.get("eda") or {}).get("result")
    ctx.leakage = (state.get("leakage_report") or {}).get("report")
    ctx.anomaly = state.get("anomaly_result")
    fair = state.get("fairness_result")
    ctx.fairness = fair[1] if fair else None

    for label, key in (("Machine Learning", "ml_run"), ("AutoML", "automl_run")):
        run = state.get(key)
        if run is None:
            continue
        ctx.runs.append((label, run))
        if ds is not None and run.signature and run.signature != (ds.name, ds.df.height, ds.df.width):
            ctx.stale.append(label)
        for name, table in run.importance.items():
            ctx.perm.append({"source": label, "model": name, "table": table})

    for (created, name, _ci), sr in (state.get("shap_cache") or {}).items():
        for label, run in ctx.runs:
            if run.created == created:
                try:
                    ctx.shap.append({"source": label, "model": name, "unit": sr.unit, "rows": len(sr.X_raw),
                                     "table": importance_table(sr, run)})
                except Exception as exc:
                    logger.warning("SHAP table skipped for the report: {}", exc)
                break

    history = state.get("history")
    if history is not None:
        ctx.steps = [s.label for s in history.snapshots[1:history.index + 1]]
    return ctx


def _join(items: list[str]) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _model_sentence(label: str, run: TrainingRun) -> str | None:
    clf = run.task == "classification"
    key = "F1" if clf else "RMSE"
    best = best_result(run)
    if best is None or best.name == "Baseline" or best.test.get(key) is None:
        return None
    value = float(best.test[key])
    n_models = len([r for r in run.results if r.ok and r.name != "Baseline"])
    text = (f"{label}: {best.name} performed best on the {len(run.X_test):,} test rows by {key} "
            f"({value:.3f}), out of {n_models} model(s) compared.")
    base = next((r for r in run.results if r.name == "Baseline" and r.ok and r.test.get(key) is not None), None)
    if base is not None:
        b = float(base.test[key])
        if clf:
            gap = value - b
            text += (f" The Baseline (always guessing the most common class) scored {b:.3f}, so the model "
                     f"improves F1 by {gap:+.3f}." if gap > 0.01 else
                     f" The Baseline scored {b:.3f}, so the model does not clearly beat simple guessing.")
        elif b > 0:
            red = (b - value) / b * 100
            text += (f" The Baseline (always predicting the mean) has RMSE {b:.3f}; the model reduces it by "
                     f"{red:.0f}%." if red > 1 else
                     f" The Baseline has RMSE {b:.3f}, so the model does not clearly beat predicting the mean.")
    if clf and best.test.get("ROC-AUC") is not None:
        text += f" ROC-AUC is {best.test['ROC-AUC']:.3f}."
    elif not clf and best.test.get("R2") is not None:
        text += f" R² is {best.test['R2']:.3f}."
    return text


def key_findings(ctx: ReportContext) -> list[str]:
    """Plain-language findings computed from the results that exist."""
    out: list[str] = []
    ds = ctx.dataset
    if ds is not None:
        text = f"The dataset '{ds.name}' has {ds.n_rows_loaded:,} rows and {ds.n_cols} columns"
        out.append(text + (f" (a sample of the {ds.total_rows:,} rows in the file)." if ds.sampled else "."))

    h = ctx.health
    if h is not None:
        out.append(f"The data health score is {h.score}/100 ({h.label}).")
        for f in [f for f in h.findings if f.severity in ("warn", "bad")][:5]:
            out.append(f.title + ("" if f.title.endswith(".") else "."))

    if ctx.eda is not None:
        miss = ctx.eda.missing
        if miss.height and int(miss["missing"].max() or 0) > 0:
            top = miss.row(0, named=True)
            out.append(f"Missing values are most concentrated in '{top['column']}' "
                       f"({top['missing_pct']:.1f}% empty); {ctx.eda.overview['missing_cells_pct']}% of all cells are empty.")

    if ctx.leakage is not None:
        strong = list(dict.fromkeys(f.column for f in ctx.leakage.findings if f.strength == "strong"))
        out.append(f"Possible leakage was flagged in {len(strong)} column(s): {_join(strong[:5])}. "
                   "Please investigate before trusting model scores." if strong else
                   "The leakage check found no strong signals (this is not a proof).")

    a = ctx.anomaly
    if a is not None:
        out.append(f"{a.method} flagged {a.n_anomalies:,} of {a.n_rows:,} analysed rows ({a.pct:.2f}%) as unusual "
                   f"across {len(a.columns)} column(s). None were removed.")

    for label, run in ctx.runs:
        sentence = _model_sentence(label, run)
        if sentence:
            out.append(sentence)

    if ctx.shap:
        item = ctx.shap[0]
        top = item["table"]["feature"].head(3).tolist()
        if top:
            out.append(f"According to SHAP, the columns that influence {item['model']} most are {_join(top)}.")
    elif ctx.perm:
        item = ctx.perm[0]
        top = item["table"]["feature"].head(3).tolist()
        if top:
            out.append(f"By permutation importance, the columns that matter most for {item['model']} are {_join(top)}.")

    if ctx.fairness is not None:
        warns = [t for lvl, t in ctx.fairness.findings if lvl == "warn"]
        out.append(f"The fairness check on '{ctx.fairness.column}' found {len(warns)} possible disparity(ies)."
                   if warns else f"The fairness check on '{ctx.fairness.column}' found no clear disparity.")
    return out


def limitations(ctx: ReportContext) -> list[str]:
    """Caveats that follow from what was actually done."""
    out: list[str] = []
    ds = ctx.dataset
    if ds is not None and ds.sampled:
        out.append(f"Only a sample of {ds.n_rows_loaded:,} of the {ds.total_rows:,} rows in the file was analysed.")
    for label in ctx.stale:
        out.append(f"The {label} results were trained on an earlier version of the data and may be out of date.")
    if ctx.leakage is not None:
        strong = {f.column for f in ctx.leakage.findings if f.strength == "strong"}
        for label, run in ctx.runs:
            used = sorted(strong & set(run.num + run.cat))
            if used:
                out.append(f"The {label} models used column(s) flagged for possible leakage: {_join(used)}. "
                           "Their scores may be too optimistic.")
    for label, run in ctx.runs:
        if len(run.X_test) < 100:
            out.append(f"The {label} test set has only {len(run.X_test)} rows, so the scores are uncertain.")
    if ctx.shap:
        out.append("SHAP explanations describe how the model behaves. They do not show causes in the real world.")
    if ctx.fairness is not None:
        out.append("Fairness findings are possible disparities, not proof of unfair treatment.")
    return out