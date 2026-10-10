"""Report rendering and export (UI-free): HTML, JSON, Excel and LaTeX."""

from __future__ import annotations

import html
import io
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from config import settings
from core.insights import ReportContext, key_findings, limitations
from core.model_training import IMBALANCE_OPTIONS, best_result, results_table
from utils.optional_imports import MissingDependencyError, lazy_import

TEMPLATES = ["Executive summary", "Technical appendix", "Research write-up"]
NOTE_SECTIONS = {
    "findings": "Key findings", "data": "Data and health", "models": "Models",
    "explain": "Explainability and fairness", "general": "General remarks",
}


class ReportError(Exception):
    """A user-facing error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2933;line-height:1.5}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #d9dee3;padding-bottom:.2rem}
.muted{color:#6b7280;font-size:.85rem}
table.tbl{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}
table.tbl th,table.tbl td{padding:3px 10px;border-bottom:1px solid #e5e7eb;text-align:right}
table.tbl th{background:#f3f4f6}table.tbl td:first-child,table.tbl th:first-child{text-align:left}
ul.find{list-style:none;padding-left:0}ul.find li{padding:6px 10px;margin:4px 0;border-left:4px solid #9ca3af;background:#f9fafb}
ul.find li.ok{border-left-color:#2ca02c}ul.find li.info{border-left-color:#1f77b4}
ul.find li.warn{border-left-color:#ff7f0e}ul.find li.bad{border-left-color:#d62728}
.score{font-size:2.4rem;font-weight:700}.score small{font-size:1rem;color:#6b7280}
.note{background:#fffbe6;border:1px solid #f2e3a2;padding:8px 12px;margin:.6rem 0;border-radius:4px}
.bar{display:flex;align-items:center;gap:8px;font-size:.85rem;margin:2px 0}.lab{width:220px;text-align:right}
.track{flex:1;background:#eef0f2;height:12px;border-radius:3px}
.fill{display:block;height:12px;background:#1f77b4;border-radius:3px}.val{width:70px}
@media print{.noprint{display:none}body{margin:0}h2{page-break-after:avoid}table,.bar{page-break-inside:avoid}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _df_html(df: pd.DataFrame | pl.DataFrame, max_rows: int = 40) -> str:
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    out = df.head(max_rows).to_html(index=False, border=0, na_rep="", classes="tbl",
                                    float_format=lambda v: f"{v:,.3f}")
    if len(df) > max_rows:
        out += f'<p class="muted">Showing {max_rows} of {len(df)} rows.</p>'
    return out


def _kv(pairs: list[tuple[str, Any]]) -> str:
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in pairs)
    return f'<table class="tbl">{rows}</table>'


def _bars(table: pd.DataFrame, label: str, value: str, n: int) -> str:
    t = table.head(n)
    top = float(t[value].abs().max())
    if not top or top != top:
        top = 1.0
    return "".join(
        f'<div class="bar"><span class="lab">{_e(r[label])}</span><span class="track"><span class="fill" '
        f'style="width:{abs(float(r[value])) / top * 100:.1f}%"></span></span>'
        f'<span class="val">{float(r[value]):.4f}</span></div>' for _, r in t.iterrows())


def _note(ctx: ReportContext, key: str) -> str:
    text = ctx.notes.get(key)
    if not text:
        return ""
    return f'<div class="note"><b>Note:</b> {_e(text).replace(chr(10), "<br>")}</div>'


def _section(title: str, body: str, ctx: ReportContext | None = None, note: str | None = None) -> str:
    if not body:
        return ""
    return f"<h2>{_e(title)}</h2>{body}{_note(ctx, note) if ctx is not None and note else ''}"


def _overview(ctx: ReportContext) -> str:
    ds = ctx.dataset
    if ds is None:
        return ""
    pairs: list[tuple[str, Any]] = [(k, f"{v:,}" if isinstance(v, int) else v) for k, v in ds.summary().items()]
    if ctx.health is not None and ctx.health.target:
        pairs.append(("Target", ctx.health.target))
    return _kv(pairs)


def _findings(ctx: ReportContext) -> str:
    items = key_findings(ctx)
    return "<ol>" + "".join(f"<li>{_e(t)}</li>" for t in items) + "</ol>" if items else ""


def _steps(ctx: ReportContext) -> str:
    if not ctx.steps:
        return ""
    return "<ol>" + "".join(f"<li>{_e(s)}</li>" for s in ctx.steps) + "</ol>"


def _health(ctx: ReportContext, detailed: bool) -> str:
    h = ctx.health
    if h is None:
        return ""
    shown = h.findings if detailed else [f for f in h.findings if f.severity != "ok"]
    items = "".join(
        f'<li class="{f.severity}"><b>{_e(f.icon)} {_e(f.title)}</b>'
        + (f"<br>{_e(f.detail)}" if f.detail else "")
        + (f"<br><i>{_e(f.suggestion)}</i>" if f.suggestion and f.severity != "ok" else "") + "</li>"
        for f in shown)
    body = f'<div class="score">{h.score}<small>/100</small> &nbsp;{_e(h.label)}</div><ul class="find">{items}</ul>'
    if not detailed:
        ok = sum(f.severity == "ok" for f in h.findings)
        body += f'<p class="muted">{ok} check(s) passed without problems.</p>'
        return body
    breakdown = h.breakdown()
    if breakdown.height:
        body += "<p><b>Points lost by each check</b></p>" + _df_html(breakdown)
    cols = ["column", "dtype", "kind", "missing_pct", "unique", "skew", "outlier_pct"]
    body += "<p><b>Columns</b></p>" + _df_html(h.columns.select(cols), 60)
    if h.correlated_pairs.height:
        body += "<p><b>Highly correlated pairs</b></p>" + _df_html(h.correlated_pairs, 15)
    return body


def _eda(ctx: ReportContext) -> str:
    r = ctx.eda
    if r is None:
        return ""
    o = r.overview
    body = (f"<p>{o['rows']:,} rows × {o['columns']} columns, {o['missing_cells_pct']}% of cells empty, "
            f"{o['duplicate_rows']:,} duplicate rows. Distribution statistics use {r.stats_rows:,} rows.</p>")
    if r.numeric.height:
        body += "<p><b>Numeric columns</b></p>" + _df_html(
            r.numeric.select(["column", "count", "missing", "mean", "std", "min", "median", "max", "skew"]), 40)
    if r.categorical.height:
        body += "<p><b>Categorical columns</b></p>" + _df_html(
            r.categorical.select(["column", "unique", "missing_pct", "top", "top_pct"]), 40)
    return body


def _leakage(ctx: ReportContext) -> str:
    r = ctx.leakage
    if r is None:
        return ""
    c = r.counts()
    body = (f"<p>Target: <b>{_e(r.target)}</b>. Strong signals: {c['strong']}, moderate: {c['moderate']}, "
            f"weak: {c['weak']} ({r.rows_checked:,} rows and {r.columns_checked} columns checked). "
            "Every finding is only a <i>possible</i> leakage.</p>")
    if r.findings:
        body += _df_html(r.table().select(["column", "check", "strength", "score", "detail"]), 15)
    return body


def _anomaly(ctx: ReportContext) -> str:
    a = ctx.anomaly
    if a is None:
        return ""
    return _kv([("Method", a.method), ("Columns", ", ".join(a.columns)), ("Rows analysed", f"{a.n_rows:,}"),
                ("Anomalies flagged", f"{a.n_anomalies:,} ({a.pct:.2f}%)")]) + \
        '<p class="muted">No rows were removed.</p>'


def _models(ctx: ReportContext, detailed: bool) -> str:
    parts: list[str] = []
    for label, run in ctx.runs:
        cfg, clf = run.config, run.task == "classification"
        split = "time-based (last rows)" if cfg.time_split else ("random, stratified" if clf else "random")
        setup: list[tuple[str, Any]] = [
            ("Source", label), ("Target", cfg.target), ("Task", run.task),
            ("Features", f"{len(run.num) + len(run.cat)} ({len(run.num)} numeric, {len(run.cat)} categorical)"),
            ("Training / test rows", f"{len(run.X_train):,} / {len(run.X_test):,}"), ("Split", split),
            ("Cross-validation", f"{cfg.cv_folds} folds" if cfg.cv_folds >= 2 else "none"),
        ]
        if clf:
            setup.append(("Class imbalance handling", IMBALANCE_OPTIONS.get(cfg.imbalance, cfg.imbalance)))
        table = results_table(run)
        body = _kv(setup) + _df_html(table, 50 if detailed else 12)
        if run.notes:
            body += "<ul>" + "".join(f"<li>{_e(n)}</li>" for n in run.notes) + "</ul>"
        parts.append(body)
    return "".join(parts)


def _importance(ctx: ReportContext, top: int) -> str:
    body = ""
    for item in ctx.shap:
        body += (f"<p><b>SHAP importance: {_e(item['model'])}</b> "
                 f'<span class="muted">(mean |SHAP| in {_e(item["unit"])}, {item["rows"]:,} rows)</span></p>'
                 + _bars(item["table"], "feature", "mean_abs_shap", top))
    for item in ctx.perm:
        body += (f"<p><b>Permutation importance: {_e(item['model'])}</b> "
                 '<span class="muted">(drop in test score when the column is shuffled)</span></p>'
                 + _bars(item["table"], "feature", "importance", top))
    return body


def _fairness(ctx: ReportContext, detailed: bool) -> str:
    f = ctx.fairness
    if f is None:
        return ""
    items = "".join(f'<li class="{lvl}">{_e(t)}</li>' for lvl, t in f.findings)
    body = f"<p>Groups defined by <b>{_e(f.column)}</b>.</p><ul class=\"find\">{items}</ul>"
    return body + (_df_html(f.table, 15) if detailed else "")


def _methods(ctx: ReportContext) -> str:
    items: list[str] = []
    for label, run in ctx.runs:
        cfg, clf = run.config, run.task == "classification"
        text = (f"{label}: the target '{cfg.target}' was modelled as a {run.task} problem using "
                f"{len(run.num) + len(run.cat)} features. The data were split into {len(run.X_train):,} training "
                f"and {len(run.X_test):,} test rows ({'time-based' if cfg.time_split else 'random'} split, seed {cfg.seed}). "
                f"Missing numeric values were filled with the {cfg.num_impute} of the training rows"
                f"{' and numeric features were standardised' if cfg.scale else ''}; categorical features were "
                "one-hot encoded. All of these steps were fitted on training rows only, inside the model pipeline.")
        if cfg.select_percentile < 100:
            text += f" Feature selection kept the top {cfg.select_percentile}% of encoded features."
        if clf and cfg.imbalance != "none":
            text += f" Class imbalance was handled with: {IMBALANCE_OPTIONS.get(cfg.imbalance, cfg.imbalance)}."
        if cfg.cv_folds >= 2:
            text += f" Models were also compared with {cfg.cv_folds}-fold cross-validation on the training rows."
        names = [r.name for r in run.results if r.ok]
        text += f" Models compared: {', '.join(names)}."
        items.append(text)
    return "".join(f"<p>{_e(t)}</p>" for t in items)


def _caveats(ctx: ReportContext) -> str:
    items = limitations(ctx)
    return "<ul>" + "".join(f"<li>{_e(t)}</li>" for t in items) + "</ul>" if items else ""


def build_html(ctx: ReportContext, template: str = "Executive summary") -> str:
    """A self-contained HTML report. Print it to PDF from the browser."""
    if template not in TEMPLATES:
        raise ReportError(f"Unknown template '{template}'.", f"Choose one of: {', '.join(TEMPLATES)}.")
    S = _section
    if template == "Executive summary":
        sections = [
            S("Dataset", _overview(ctx)), S("Key findings", _findings(ctx), ctx, "findings"),
            S("Data health", _health(ctx, False), ctx, "data"), S("Models", _models(ctx, False), ctx, "models"),
            S("Most important columns", _importance(ctx, 8), ctx, "explain"),
            S("Fairness", _fairness(ctx, False)), S("Caveats", _caveats(ctx)),
        ]
    elif template == "Technical appendix":
        sections = [
            S("Dataset", _overview(ctx)), S("Key findings", _findings(ctx), ctx, "findings"),
            S("Preprocessing steps", _steps(ctx)), S("Data health", _health(ctx, True), ctx, "data"),
            S("Exploratory analysis", _eda(ctx)), S("Leakage check", _leakage(ctx)),
            S("Anomaly detection", _anomaly(ctx)), S("Models", _models(ctx, True), ctx, "models"),
            S("Feature importance", _importance(ctx, 15), ctx, "explain"),
            S("Fairness", _fairness(ctx, True)), S("Caveats", _caveats(ctx)),
        ]
    else:
        sections = [
            S("1. Data", _overview(ctx) + (f"<p><b>Preprocessing</b></p>{_steps(ctx)}" if ctx.steps else "")
              + _health(ctx, False), ctx, "data"),
            S("2. Methods", _methods(ctx), ctx, "models"),
            S("3. Results", _findings(ctx) + _models(ctx, False) + _importance(ctx, 10) + _fairness(ctx, False),
              ctx, "findings"),
            S("4. Limitations", _caveats(ctx)),
        ]
    body = "".join(sections) or "<p>Nothing to report yet. Run some analyses in the app first.</p>"
    body += _note(ctx, "general")
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{_e(ctx.title)}</title>'
        f"<style>{_CSS}</style></head><body>"
        '<p class="noprint muted">To make a PDF, print this page (Ctrl+P) and choose "Save as PDF".</p>'
        f"<h1>{_e(ctx.title)}</h1>"
        f'<p class="muted">{_e(template)} · generated {_e(ctx.created)} by {_e(settings.app_name)} v{_e(settings.version)}</p>'
        f"{body}"
        '<p class="muted">Every number in this report was computed from the analyses run in the app. '
        'Only text marked "Note" was written by a person.</p></body></html>'
    )


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #

def _clean(o: Any) -> Any:
    """Make any value JSON-safe (no NaN, no numpy types)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    return str(o)


def results_dict(ctx: ReportContext) -> dict[str, Any]:
    from dataclasses import asdict

    d: dict[str, Any] = {"generated": ctx.created, "app_version": settings.version, "title": ctx.title,
                         "key_findings": key_findings(ctx), "caveats": limitations(ctx),
                         "preprocessing_steps": ctx.steps, "notes": ctx.notes}
    if ctx.dataset is not None:
        d["dataset"] = ctx.dataset.summary()
    if ctx.health is not None:
        d["health"] = ctx.health.to_dict()
    if ctx.eda is not None:
        d["eda"] = {"overview": ctx.eda.overview, "numeric": ctx.eda.numeric.to_dicts(),
                    "categorical": ctx.eda.categorical.to_dicts(), "missing": ctx.eda.missing.to_dicts()}
    if ctx.leakage is not None:
        d["leakage"] = {"target": ctx.leakage.target, "counts": ctx.leakage.counts(),
                        "findings": [asdict(f) for f in ctx.leakage.findings]}
    if ctx.anomaly is not None:
        a = ctx.anomaly
        d["anomalies"] = {"method": a.method, "columns": a.columns, "rows_analysed": a.n_rows,
                          "flagged": a.n_anomalies, "percent": a.pct}
    d["models"] = [{"source": label, "task": run.task, "target": run.config.target, "classes": run.classes,
                    "config": asdict(run.config), "notes": run.notes,
                    "results": results_table(run).to_dict("records")} for label, run in ctx.runs]
    d["importance"] = (
        [{"kind": "shap", "model": i["model"], "unit": i["unit"], "table": i["table"].to_dict("records")} for i in ctx.shap]
        + [{"kind": "permutation", "model": i["model"], "table": i["table"].to_dict("records")} for i in ctx.perm])
    if ctx.fairness is not None:
        f = ctx.fairness
        d["fairness"] = {"column": f.column, "findings": [{"level": a, "text": b} for a, b in f.findings],
                         "gaps": f.gaps, "table": f.table.to_dict("records")}
    return d


def build_json(ctx: ReportContext) -> str:
    return json.dumps(_clean(results_dict(ctx)), indent=2, allow_nan=False)


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #

def _plain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: v if v is None or isinstance(v, (str, int, float, bool)) else str(v))
    return out


def build_excel(ctx: ReportContext, include_data: bool = True, max_rows: int = 50_000) -> bytes:
    """An Excel workbook with the summary, health, models, importance and (optionally) the data."""
    try:
        lazy_import("xlsxwriter")
    except MissingDependencyError as exc:
        raise ReportError("Excel export needs the 'xlsxwriter' package.", str(exc)) from exc

    findings = key_findings(ctx)
    sheets: dict[str, pd.DataFrame] = {
        "Summary": pd.DataFrame({"#": range(1, len(findings) + 1), "Key finding": findings})
        if findings else pd.DataFrame({"Info": ["Nothing to report yet."]})
    }
    if ctx.health is not None:
        sheets["Health findings"] = pd.DataFrame(
            [{"severity": f.severity, "title": f.title, "detail": f.detail, "suggestion": f.suggestion,
              "points_lost": f.penalty} for f in ctx.health.findings])
        sheets["Health columns"] = ctx.health.columns.to_pandas()
    if ctx.eda is not None and ctx.eda.numeric.height:
        sheets["Numeric summary"] = ctx.eda.numeric.to_pandas()
    if ctx.runs:
        sheets["Models"] = pd.concat(
            [results_table(run).assign(Source=label) for label, run in ctx.runs], ignore_index=True)
    imp = [i["table"].assign(model=i["model"], kind="shap") for i in ctx.shap]
    imp += [i["table"].assign(model=i["model"], kind="permutation") for i in ctx.perm]
    if imp:
        sheets["Importance"] = pd.concat(imp, ignore_index=True)
    if include_data and ctx.dataset is not None:
        sheets["Dataset"] = ctx.dataset.df.head(max_rows).to_pandas()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
        for name, frame in sheets.items():
            frame = _plain(frame)
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(frame.shape[1] - 1, 0), 18)
            if name == "Summary" and frame.shape[1] > 1:
                ws.set_column(1, 1, 120)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

_LATEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
          "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: Any) -> str:
    return "".join(_LATEX.get(ch, ch) for ch in str(text)).replace("±", r"$\pm$")


def _cell(v: Any, decimals: int) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(float(v))):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{decimals}f}"
    return latex_escape(v)


def to_latex(df: pd.DataFrame, caption: str, label: str, decimals: int = 3) -> str:
    """A booktabs table (add \\usepackage{booktabs} to your document)."""
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_cell(v, decimals) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_tables(ctx: ReportContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, run in ctx.runs:
        slug = label.lower().replace(" ", "-")
        t = results_table(run).drop(columns=["Status"], errors="ignore").head(15)
        out[f"Model comparison ({label})"] = to_latex(
            t, f"Model comparison ({label}). Metrics are on the held-out test set.", f"tab:models-{slug}")
    if ctx.eda is not None and ctx.eda.numeric.height:
        t = ctx.eda.numeric.select(["column", "count", "mean", "std", "min", "median", "max"]).head(20).to_pandas()
        out["Descriptive statistics"] = to_latex(t, "Descriptive statistics of the numeric variables.", "tab:descriptives")
    for i in ctx.shap:
        out[f"SHAP importance ({i['model']})"] = to_latex(
            i["table"].head(10), f"Mean absolute SHAP value of the ten most influential columns ({i['model']}).",
            "tab:shap-" + str(abs(hash(i["model"])) % 10_000))
    return out