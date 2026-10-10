import streamlit as st

from config import settings
from core.recommendations import recommend

st.title(f"{settings.app_icon} {settings.app_name}")
st.write(
    "Upload a dataset, check its health, explore it, clean it, train and compare models, "
    "explain them, look for leakage, anomalies and unfair group differences, compare datasets, "
    "and export a report. Everything runs locally."
)

dataset = st.session_state.get("dataset")
health = st.session_state.get("health")

if dataset is not None:
    top = recommend(st.session_state)[:3]
    if top:
        st.subheader("Suggested next steps")
        SHOW = {"high": st.error, "medium": st.warning, "low": st.info}
        for i, r in enumerate(top, start=1):
            SHOW[r.priority](f"**{r.title}**  \n{r.why}")
            st.page_link(r.page, label=f"Suggestion {i}: {r.label}")
        st.page_link("views/recommendations.py", label="See all suggestions")

if st.session_state.get("guided_mode", True):
    st.subheader("Your roadmap")
    steps = [
        ("Upload a dataset or open a demo", dataset is not None, "views/upload.py", "Go to Data Upload"),
        ("Check Data Health", health is not None, "views/data_health.py", "Go to Data Health"),
        ("Explore with EDA", "eda" in st.session_state, "views/eda.py", "Go to EDA"),
        ("Make plots", "plot_used" in st.session_state, "views/plots.py", "Go to Plot Studio"),
        ("Clean and prepare the data", "history" in st.session_state, "views/preprocessing.py", "Go to Preprocessing"),
        ("Check for leakage", "leakage_report" in st.session_state, "views/leakage.py", "Go to Leakage Check"),
        ("Look for anomalies", "anomaly_result" in st.session_state, "views/anomaly.py", "Go to Anomaly Detection"),
        ("Train and compare models", "ml_run" in st.session_state, "views/machine_learning.py", "Go to Machine Learning"),
        ("Try AutoML", "automl_run" in st.session_state, "views/automl.py", "Go to AutoML"),
        ("Explain a model", bool(st.session_state.get("shap_cache")), "views/explainability.py", "Go to Explainability"),
        ("Check fairness across groups", "fairness_result" in st.session_state, "views/fairness.py", "Go to Fairness Check"),
        ("Compare datasets and check drift", "compare_result" in st.session_state, "views/compare.py", "Go to Compare & Drift"),
        ("Export a report", "rp_files" in st.session_state, "views/reports.py", "Go to Reports & Export"),
    ]
    for i, (text, done, page, label) in enumerate(steps, start=1):
        left, right = st.columns([4, 1])
        left.markdown(f"{'✅' if done else '⬜'} **Step {i}:** {text}")
        right.page_link(page, label=label)
    st.caption("Turn Guided mode off in the sidebar to hide this roadmap.")
else:
    st.page_link("views/upload.py", label="Go to Data Upload")