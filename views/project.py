import streamlit as st

from core.project import ProjectError, apply_project, load_project, read_manifest, save_project_bytes

st.title("💾 Project")
st.caption("Save your current data (after preprocessing) and notes to one file, and open it again later. "
           "Trained models and plots are not stored: save models in Reports & Export, and re-run analyses after loading.")

tab_save, tab_load = st.tabs(["Save", "Open"])

with tab_save:
    if st.session_state.get("dataset") is None:
        st.info("Load a dataset first.")
        st.page_link("views/upload.py", label="Go to Data Upload")
    else:
        if st.button("Prepare project file", type="primary"):
            try:
                st.session_state["_proj_bytes"] = save_project_bytes(st.session_state)
            except ProjectError as err:
                st.error(err.message)
                st.info(err.suggestion)
        data = st.session_state.get("_proj_bytes")
        if data:
            st.download_button("Download project (.wbproj)", data, file_name="project.wbproj",
                               mime="application/zip")

with tab_load:
    st.warning("Opening a project replaces the current dataset and clears results computed from the old data "
               "(health, models, SHAP and so on).")
    up = st.file_uploader("Project file", type=["wbproj"], key="proj_up")
    if up is not None:
        raw = up.getvalue()
        try:
            manifest = read_manifest(raw)
        except ProjectError as err:
            st.error(err.message)
            if err.suggestion:
                st.info(err.suggestion)
        else:
            st.write(f"**{manifest['dataset_name']}**: {manifest['rows']:,} rows × {manifest['columns']} columns, "
                     f"saved {manifest['created']} (app v{manifest['app_version']})")
            if manifest.get("preprocessing_steps"):
                st.caption("Preprocessing steps already applied: " + "; ".join(manifest["preprocessing_steps"]))
            if st.button("Open this project", type="primary"):
                try:
                    manifest, dataset = load_project(raw)
                except ProjectError as err:
                    st.error(err.message)
                    st.info(err.suggestion)
                else:
                    apply_project(st.session_state, manifest, dataset)
                    st.success("Project opened. Re-run Data Health and your other analyses.")
                    st.page_link("views/data_health.py", label="Go to Data Health")