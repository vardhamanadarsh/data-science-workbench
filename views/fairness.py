import streamlit as st

from core.fairness import fairness_fig, fairness_report
from core.model_training import MLError
from utils.run_picker import pick_model

st.title("⚖️ Fairness Check")
st.caption("Compares how the model performs for different groups. Findings are possible disparities, "
           "not proof of unfair treatment. Differences can reflect real differences in the data.")

run, res = pick_model("fa", lambda r: r.y_pred is not None)
clf = run.task == "classification"
features = run.cat + run.num
ckey = f"{run.created}|{res.name}"

column = st.selectbox("Column that defines the groups", features, key=f"fa_col|{ckey}",
                      help="Only columns the model used as features can be checked.")
bins = 4
if column in run.num:
    bins = st.slider("Number of groups (numeric columns are cut into equal-sized bins)", 2, 6, 4, key=f"fa_bins|{ckey}")
min_group = st.slider("Smallest group to judge (test rows)", 10, 200, 30, 10, key=f"fa_min|{ckey}")

if st.button("Run fairness check", type="primary"):
    try:
        st.session_state["fairness_result"] = (ckey, fairness_report(run, res, column, min_group, bins))
    except MLError as err:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    except Exception as exc:
        st.error(f"The check failed: {exc}")

entry = st.session_state.get("fairness_result")
if entry is None or entry[0] != ckey:
    st.stop()
result = entry[1]

SHOW = {"ok": st.success, "info": st.info, "warn": st.warning}
for level, text in result.findings:
    SHOW[level](text)
if result.gaps:
    st.dataframe({"measure": list(result.gaps), "value": [round(v, 4) for v in result.gaps.values()]})

st.dataframe(result.table)
st.caption(f"Groups with fewer than {result.min_group} test rows are shown in grey and are left out of the gaps. "
           + ("Numeric values were cut into quantile bins. " if result.binned else ""))
metrics = [c for c in result.table.columns if c not in ("group", "n", "share")]
metric = st.selectbox("Chart", metrics, key=f"fa_metric|{ckey}")
st.plotly_chart(fairness_fig(result, metric))
if clf and len(run.classes) == 2:
    st.caption(f"Positive class: '{run.classes[1]}'. Selection rate = share predicted positive; "
               "TPR = recall; FPR = false positive rate.")