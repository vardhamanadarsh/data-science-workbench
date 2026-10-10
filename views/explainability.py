import pandas as pd
import streamlit as st

from core.explain import (
    compute_shap, dependence_fig, explain_frame, explainable, explainer_kind, importance_fig,
    importance_table, local_contributions, summary_fig, waterfall_fig,
)
from core.model_training import MLError
from core.whatif import (
    feature_domains, find_counterfactuals, fmt_value, predict_frame, prediction_summary, typical_row,
)
from utils.optional_imports import install_hint, is_available
from utils.run_picker import pick_model

st.title("💡 Explainability")

run, res = pick_model("ex", explainable)
classes = run.classes
clf = run.task == "classification"
ckey = f"{run.created}|{res.name}"

class_index = 1
if clf and len(classes) > 2:
    class_index = classes.index(st.selectbox("Class to explain", classes, key=f"ex_class|{ckey}"))
elif clf:
    st.caption(f"Explaining the probability of the positive class: '{classes[1]}'.")
st.caption(f"Target: {run.config.target}. Explanations use rows from the test set "
           f"({len(run.X_test):,} rows). They describe how the MODEL behaves, not causes in the real world.")

# ---- SHAP computation -----------------------------------------------------
shap_cache: dict = st.session_state.setdefault("shap_cache", {})
skey = (run.created, res.name, class_index)
sr = shap_cache.get(skey)
shap_ok = is_available("shap")
with st.expander("SHAP settings", expanded=sr is None):
    if not shap_ok:
        st.warning(install_hint("shap"))
        st.code("uv sync --extra ml --extra automl --extra explain", language="bash")
        st.caption("The What-if and Counterfactuals tabs work without SHAP.")
    else:
        kind = explainer_kind(res.name)
        rows = st.slider("Rows to explain", 50, 1000, 300, 50, key="ex_rows")
        if kind == "generic":
            st.caption("This model type uses a slower generic explainer, so at most 100 rows are explained.")
        if st.button("Compute SHAP values", type="primary"):
            try:
                with st.spinner("Computing SHAP values..."):
                    shap_cache[skey] = compute_shap(run, res, rows, class_index)
            except MLError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
            except Exception as exc:
                st.error(f"SHAP failed: {exc}")
            else:
                st.rerun()

tab_global, tab_summary, tab_dep, tab_local, tab_what, tab_cf = st.tabs(
    ["Global importance", "SHAP summary", "Dependence", "Single prediction", "What-if", "Counterfactuals"])


def need_shap() -> bool:
    if sr is None:
        st.info("Compute SHAP values first (see 'SHAP settings' above).")
        return True
    return False


with tab_global:
    if not need_shap():
        st.plotly_chart(importance_fig(sr, run))
        st.dataframe(importance_table(sr, run))
        st.caption(f"Mean absolute SHAP value per column over {len(sr.X_raw):,} rows, in {sr.unit}. "
                   "One-hot columns of the same category column are added together.")

with tab_summary:
    if not need_shap():
        st.plotly_chart(summary_fig(sr))
        st.caption("Each dot is one row. Right of the line pushes the prediction up, left pushes it down. "
                   "Red dots have a high value of that feature, blue dots a low value.")

with tab_dep:
    if not need_shap():
        feats = importance_table(sr, run)["feature"].tolist()
        feature = st.selectbox("Feature", feats, key=f"ex_dep|{ckey}")
        others = [c for c in feats if c in run.num and c != feature]
        color_by = None
        if feature in run.num and others:
            choice = st.selectbox("Colour by", ["(none)"] + others, key=f"ex_depc|{ckey}")
            color_by = None if choice == "(none)" else choice
        st.plotly_chart(dependence_fig(sr, run, feature, color_by))
        st.caption("Shows how the value of one feature changes its contribution to the prediction.")

with tab_local:
    if not need_shap():
        i = int(st.number_input("Row (among the explained test rows)", 0, len(sr.X_raw) - 1, 0, key=f"ex_row|{ckey}"))
        row = sr.X_raw.iloc[[i]]
        st.subheader(prediction_summary(run, res, row))
        if sr.y_true is not None:
            actual = classes[int(sr.y_true[i])] if clf else f"{float(sr.y_true[i]):,.3f}"
            st.caption(f"Actual value in the data: {actual}")
        contrib = local_contributions(sr, run, i)
        what = f"the chance of '{sr.class_name}'" if clf else "the prediction"
        up = contrib[contrib["shap"] > 0].sort_values("shap", ascending=False).head(3)
        down = contrib[contrib["shap"] < 0].sort_values("shap").head(3)
        left, right = st.columns(2)
        left.write(f"**Factors increasing {what}**")
        for r in up.itertuples():
            left.write(f"↑ {r.feature} = {r.value} ({r.shap:+.3f})")
        if up.empty:
            left.caption("None")
        right.write(f"**Factors reducing {what}**")
        for r in down.itertuples():
            right.write(f"↓ {r.feature} = {r.value} ({r.shap:+.3f})")
        if down.empty:
            right.caption("None")
        st.plotly_chart(waterfall_fig(contrib, sr.base, sr.unit))
        st.caption(f"Contributions are in {sr.unit}. They add up from the average prediction to this prediction.")
        with st.expander("Row values"):
            st.dataframe(row)

with tab_what:
    domains = feature_domains(run)
    feats = [c for c in run.num + run.cat if c in domains]
    how = st.radio("Start from", ["A test row", "A typical row"], horizontal=True, key=f"wi_how|{ckey}")
    if how == "A test row":
        n = int(st.number_input("Test row number", 0, len(run.X_test) - 1, 0, key=f"wi_n|{ckey}"))
        base_row, rk = run.X_test.iloc[[n]].reset_index(drop=True), f"row{n}"
    else:
        base_row, rk = typical_row(run, domains), "typical"

    edited: dict = {}
    cols_ui = st.columns(3)
    for pos, c in enumerate(feats[:30]):
        d, cur = domains[c], base_row.iloc[0][c]
        box, key = cols_ui[pos % 3], f"wi|{ckey}|{rk}|{c}"
        if d.kind == "num":
            value = min(max(d.median if pd.isna(cur) else float(cur), d.low), d.high)
            if d.high <= d.low:
                edited[c] = value
            elif d.is_int:
                edited[c] = box.slider(c, int(d.low), int(d.high), int(round(value)), 1, key=key)
            else:
                edited[c] = box.slider(c, float(d.low), float(d.high), float(value), key=key)
        else:
            current = str(cur) if not pd.isna(cur) and str(cur) in d.categories else d.categories[0]
            edited[c] = box.selectbox(c, d.categories, index=d.categories.index(current), key=key)
    if len(feats) > 30:
        st.caption("Only the first 30 features can be edited here; the others keep their original values.")

    scenario = base_row.copy()
    for c, v in edited.items():
        scenario[c] = [v]
    try:
        p0, has_proba = predict_frame(run, res, base_row)
        p1, _ = predict_frame(run, res, scenario)
    except Exception as exc:
        st.error(f"The prediction failed: {exc}")
    else:
        if not clf:
            st.metric("Prediction", f"{p1[0]:,.3f}", f"{p1[0] - p0[0]:+,.3f} vs the starting row")
        elif len(classes) == 2:
            st.metric(f"Probability of '{classes[1]}'", f"{p1[0][1]:.1%}",
                      f"{(p1[0][1] - p0[0][1]) * 100:+.1f} points vs the starting row")
            st.caption(f"Predicted class: '{classes[int(p1[0].argmax())]}' "
                       f"(starting row: '{classes[int(p0[0].argmax())]}').")
        else:
            st.dataframe(pd.DataFrame({"class": classes, "starting row": p0[0], "scenario": p1[0]}))
        if clf and not has_proba:
            st.caption("This model gives class labels only, so probabilities show as 0% or 100%.")
        if shap_ok and explainer_kind(res.name) != "generic":
            if st.checkbox("Explain this scenario with SHAP", key=f"wi_shap|{ckey}"):
                try:
                    one = explain_frame(run, res, scenario, class_index)
                    st.plotly_chart(waterfall_fig(local_contributions(one, run, 0), one.base, one.unit))
                except MLError as err:
                    st.error(err.message)

with tab_cf:
    if not clf:
        st.info("Counterfactuals are available for classification problems. Use the What-if tab for regression.")
    else:
        domains = feature_domains(run)
        n = int(st.number_input("Test row number", 0, len(run.X_test) - 1, 0, key=f"cf_n|{ckey}"))
        row = run.X_test.iloc[[n]].reset_index(drop=True)
        st.write(prediction_summary(run, res, row))
        p_now, _ = predict_frame(run, res, row)
        current = int(p_now[0].argmax())
        options = [c for i, c in enumerate(classes) if i != current]
        desired_name = st.selectbox("I want the model to predict", options, key=f"cf_cls|{ckey}")
        mutable = st.multiselect(
            "Features that may change", [c for c in run.num + run.cat if c in domains],
            default=[c for c in run.num + run.cat if c in domains], key=f"cf_mut|{ckey}",
            help="Remove features that cannot realistically be changed (for example age).")
        max_changes = st.radio("Maximum number of features to change", [1, 2], index=1, horizontal=True,
                               key=f"cf_max|{ckey}")
        if st.button("Search for counterfactuals", type="primary"):
            try:
                with st.spinner("Searching..."):
                    out = find_counterfactuals(run, res, row, classes.index(desired_name), mutable, max_changes)
            except MLError as err:
                st.error(err.message)
                if err.suggestion:
                    st.info(f"What you can do: {err.suggestion}")
            except Exception as exc:
                st.error(f"The search failed: {exc}")
            else:
                if not out.items:
                    st.warning(f"No combination of up to {max_changes} change(s) within the observed range makes the "
                               f"model predict '{desired_name}' ({out.n_tested:,} combinations tested).")
                else:
                    st.success(f"Found {len(out.items)} option(s), smallest change first "
                               f"({out.n_tested:,} combinations tested).")
                    st.dataframe(pd.DataFrame({
                        "change": [cf.describe() for cf in out.items],
                        f"probability of '{desired_name}'": [cf.prob for cf in out.items],
                        "size of change": [round(cf.distance, 2) for cf in out.items],
                    }))
        st.caption("Values are taken from the training data. These show what would change the MODEL's prediction. "
                   "They do not mean that changing the real-world feature would change the real outcome.")