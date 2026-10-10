"""Entry point: page setup, navigation and the persistent sidebar."""

from __future__ import annotations

import psutil
import streamlit as st

from config import ProcessingMode, settings
from core.data_loader import human_size

st.set_page_config(page_title=settings.app_name, page_icon=settings.app_icon, layout="wide")

st.session_state.setdefault("mode", settings.default_mode)
st.session_state.setdefault("guided_mode", True)

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/recommendations.py", title="What should I do?", icon="🧭"),
    st.Page("views/explain.py", title="Explain my results", icon="💬"),
    st.Page("views/upload.py", title="Data Upload", icon="📤"),
    st.Page("views/project.py", title="Project", icon="💾"),
    st.Page("views/data_health.py", title="Data Health", icon="🩺"),
    st.Page("views/eda.py", title="EDA", icon="🔎"),
    st.Page("views/plots.py", title="Plot Studio", icon="🎨"),
    st.Page("views/preprocessing.py", title="Preprocessing", icon="🧹"),
    st.Page("views/machine_learning.py", title="Machine Learning", icon="🤖"),
    st.Page("views/automl.py", title="AutoML", icon="⚡"),
    st.Page("views/explainability.py", title="Explainability", icon="💡"),
    st.Page("views/fairness.py", title="Fairness Check", icon="⚖️"),
    st.Page("views/leakage.py", title="Leakage Check", icon="🕵️"),
    st.Page("views/anomaly.py", title="Anomaly Detection", icon="🚨"),
    st.Page("views/compare.py", title="Compare & Drift", icon="🔀"),
    st.Page("views/experiments.py", title="Experiments", icon="🧪"),
    st.Page("views/reports.py", title="Reports & Export", icon="📑"),
]
page = st.navigation(pages)


def fingerprint_card() -> None:
    """One-glance summary of the loaded dataset, pinned in the sidebar."""
    ds = st.session_state.get("dataset")
    with st.container(border=True):
        st.markdown("**Dataset**")
        if ds is None:
            st.caption("No dataset loaded yet.")
            return
        st.write(f"`{ds.name}`")
        st.caption(f"{ds.n_rows_loaded:,} rows × {ds.n_cols} columns · {human_size(ds.memory_bytes)} in memory")
        if ds.sampled:
            st.caption(f"Sample of {ds.total_rows:,} rows in the file")
        entry = st.session_state.get("health")
        if entry is not None:
            report = entry["report"]
            st.metric("Health score", f"{report.score}/100", report.label, delta_color="off")
            top = next((f for f in report.findings if f.severity in ("bad", "warn")), None)
            if top is not None:
                st.caption(f"{top.icon} {top.title}")


with st.sidebar:
    st.title(f"{settings.app_icon} {settings.app_name}")
    st.caption(f"v{settings.version}")
    st.selectbox(
        "Processing mode",
        list(ProcessingMode),
        format_func=lambda m: m.value,
        key="mode",
        help="Fast Preview uses up to 50k rows, Standard up to 500k, Full uses everything. "
        "The mode applies the next time you load a file.",
    )
    st.toggle("Guided mode", key="guided_mode", help="Show a step-by-step roadmap on the Home page.")
    fingerprint_card()
    st.divider()
    memory = psutil.virtual_memory()
    app_memory = psutil.Process().memory_info().rss
    st.caption(f"🧠 App memory: {human_size(app_memory)} · System RAM used: {memory.percent:.0f}%")
    if settings.local_processing_mode:
        st.caption("🔒 Local Processing Mode: your data stays on this machine.")

page.run()