import io

import polars as pl
import streamlit as st

from core.insights import build_context, key_findings
from core.model_io import (
    ModelIOError, can_save, inspect_untrusted, load_model_bytes, predict_from_frame, save_model_bytes,
)
from core.model_training import ranked
from core.reports import NOTE_SECTIONS, TEMPLATES, ReportError, build_excel, build_html, build_json, latex_tables

st.title("📑 Reports & Export")

ds = st.session_state.get("dataset")
title = st.text_input("Report title", value=f"Analysis of {ds.name}" if ds else "Data analysis report",
                      key=f"rp_title|{ds.name if ds else ''}")
with st.expander("Your notes (included in the report, marked as notes)"):
    notes = {k: st.text_area(label, key=f"rp_note_{k}", height=80) for k, label in NOTE_SECTIONS.items()}

ctx = build_context(st.session_state, title, notes)
if ctx.is_empty:
    st.info("Nothing to report yet. Load a dataset and run some analyses first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

included = [("Dataset", ctx.dataset is not None), ("Data health", ctx.health is not None),
            ("EDA", ctx.eda is not None), ("Leakage check", ctx.leakage is not None),
            ("Anomalies", ctx.anomaly is not None), ("Models", bool(ctx.runs)),
            ("Explainability", bool(ctx.shap or ctx.perm)), ("Fairness", ctx.fairness is not None)]
st.caption("Included: " + " · ".join(f"{'✅' if ok else '⬜'} {name}" for name, ok in included)
           + ". Sections you have not run are left out.")
for label in ctx.stale:
    st.warning(f"The {label} results were trained on an earlier version of the data.")

files: dict = st.session_state.setdefault("rp_files", {})
tab_report, tab_data, tab_latex, tab_model = st.tabs(["Report", "Data and results", "LaTeX tables", "Model files"])

with tab_report:
    template = st.selectbox("Template", TEMPLATES, key="rp_template", help=(
        "Executive summary: short, for decision makers. Technical appendix: every table. "
        "Research write-up: data, methods, results and limitations."))
    findings = key_findings(ctx)
    st.subheader("Key findings")
    if findings:
        for i, text in enumerate(findings, start=1):
            st.write(f"{i}. {text}")
    else:
        st.caption("No findings yet.")
    try:
        report_html = build_html(ctx, template)
    except ReportError as err:
        st.error(err.message)
        st.info(err.suggestion)
    else:
        st.download_button("Download report (HTML)", report_html, file_name="report.html", mime="text/html",
                           type="primary")
        st.caption("To make a PDF: open the downloaded file in your browser, press Ctrl+P and choose 'Save as PDF'.")

with tab_data:
    st.download_button("Download all results (JSON)", build_json(ctx), file_name="results.json",
                       mime="application/json")
    include_data = st.checkbox("Include the dataset in the Excel file (first 50,000 rows)", value=True, key="rp_incl")
    if st.button("Prepare Excel workbook"):
        try:
            with st.spinner("Building workbook..."):
                files["xlsx"] = build_excel(ctx, include_data)
        except ReportError as err:
            st.error(err.message)
            st.info(err.suggestion)
        except Exception as exc:
            st.error(f"The workbook could not be built: {exc}")
    if "xlsx" in files:
        st.download_button("Download Excel workbook", files["xlsx"], file_name="results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if ds is not None:
        st.write(f"Current dataset ({ds.n_rows_loaded:,} rows, including your preprocessing steps)")
        if st.button("Prepare CSV"):
            files["csv"] = ds.df.write_csv().encode("utf-8")
        if "csv" in files:
            st.download_button("Download dataset (CSV)", files["csv"], file_name="dataset.csv", mime="text/csv")

with tab_latex:
    tables = latex_tables(ctx)
    if not tables:
        st.info("Run EDA, train models, or compute SHAP values to create tables.")
    else:
        name = st.selectbox("Table", list(tables), key="rp_tex")
        st.code(tables[name], language="latex")
        st.download_button("Download all tables (.tex)", "\n\n".join(tables.values()), file_name="tables.tex",
                           mime="text/plain")
        st.caption("Add \\usepackage{booktabs} to your LaTeX document.")

with tab_model:
    st.subheader("Save a trained model")
    options = {f"{label}: {r.name}": (run, r) for label, run in ctx.runs for r in ranked(run) if can_save(r)}
    if not options:
        st.info("Train a model in Machine Learning or AutoML first (the Baseline and FLAML models cannot be saved).")
    else:
        choice = st.selectbox("Model", list(options), key="rp_model_pick")
        if st.button("Prepare model file"):
            try:
                run, res = options[choice]
                files["model"] = (choice, save_model_bytes(run, res))
            except ModelIOError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
        if "model" in files:
            st.download_button(f"Download {files['model'][0]} (.skops)", files["model"][1],
                               file_name="model.skops", mime="application/octet-stream")
            st.caption("The file holds the fitted pipeline (imputing, scaling, encoding and model) and the "
                       "list of columns it needs.")

    st.divider()
    st.subheader("Load a model file and predict")
    st.warning("Only load model files from sources you trust. A file can contain code-like objects, which is why "
               "you must approve the types listed below.")
    up = st.file_uploader("Model file (.skops)", type=["skops"], key="rp_model_up")
    if up is not None:
        data = up.getvalue()
        try:
            untrusted = inspect_untrusted(data)
        except ModelIOError as err:
            st.error(err.message)
            st.info(err.suggestion)
            untrusted = None
        if untrusted is not None:
            if untrusted:
                st.write(f"{len(untrusted)} type(s) are not trusted by default:")
                st.code("\n".join(untrusted))
            ok = st.checkbox("I trust the source of this file and these types", key="rp_trust") if untrusted else True
            if ok and st.button("Load model"):
                try:
                    st.session_state["rp_loaded"] = load_model_bytes(data, untrusted)
                except ModelIOError as err:
                    st.error(err.message)
                    st.info(err.suggestion)

    bundle = st.session_state.get("rp_loaded")
    if bundle is not None:
        st.success(f"Loaded: {bundle['model_name']} ({bundle['task']}, target '{bundle['target']}', "
                   f"saved {bundle.get('created', 'unknown')})")
        st.write("Columns the model needs: " + ", ".join(bundle["num"] + bundle["cat"]))
        if bundle.get("test_metrics"):
            st.json(bundle["test_metrics"], expanded=False)
        data_up = st.file_uploader("Data to predict (CSV or Parquet)", type=["csv", "parquet"], key="rp_pred_up")
        if data_up is not None and st.button("Predict"):
            try:
                raw = data_up.getvalue()
                frame = (pl.read_parquet(io.BytesIO(raw)) if data_up.name.lower().endswith(".parquet")
                         else pl.read_csv(io.BytesIO(raw), infer_schema_length=10_000, encoding="utf8-lossy"))
                st.session_state["rp_pred"] = predict_from_frame(bundle, frame)
            except ModelIOError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
            except Exception as exc:
                st.error(f"The file could not be read: {exc}")
        pred = st.session_state.get("rp_pred")
        if pred is not None:
            st.dataframe(pred.head(200))
            st.download_button("Download predictions (CSV)", pred.write_csv(), file_name="predictions.csv",
                               mime="text/csv")