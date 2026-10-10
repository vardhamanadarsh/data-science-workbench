import streamlit as st

from core.anomaly_detection import (
    METHOD_SLUG, METHODS, AnomalyError, affected_fig, anomaly_rows, detect_anomalies,
    distribution_fig, flag_column, projection_fig, scatter_fig, score_fig,
)
from utils.session import apply_step

st.title("🚨 Anomaly Detection")
st.caption("Finds unusual rows. Nothing is ever removed automatically.")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

df = dataset.df
mode = st.session_state["mode"]
num = [c for c, t in df.schema.items() if t.is_numeric()]
if not num:
    st.info("This dataset has no numeric columns.")
    st.stop()

flash = st.session_state.pop("an_flash", None)
if flash:
    st.success(flash)

method = st.selectbox("Method", METHODS, key="an_method", help=(
    "IQR and Z-score check each column separately. Isolation Forest and Local Outlier Factor "
    "look at all selected columns together and can find unusual combinations."))
cols = st.multiselect("Numeric columns to check", num, default=num[:8], max_selections=50,
                      key=f"an_cols|{dataset.name}")

factor, z_threshold, contamination, neighbors = 1.5, 3.0, 0.05, 20
if method == "IQR":
    factor = st.slider("IQR factor (1.5 is standard)", 1.0, 5.0, 1.5, 0.1, key="an_factor")
elif method == "Z-score":
    z_threshold = st.slider("Z-score threshold", 2.0, 6.0, 3.0, 0.5, key="an_z")
else:
    contamination = st.slider("Expected share of anomalies", 0.01, 0.20, 0.05, 0.01, key="an_cont",
                              help="The method flags about this share of rows.")
    if method == "Local Outlier Factor":
        neighbors = st.slider("Neighbours", 5, 100, 20, key="an_nn")
        st.caption("Local Outlier Factor is slow on big data, so it analyses at most 20,000 rows.")

if st.button("Detect anomalies", type="primary", disabled=not cols):
    try:
        with st.spinner("Analysing..."):
            st.session_state["anomaly_result"] = detect_anomalies(
                df, cols, method, mode, factor=factor, z_threshold=z_threshold,
                contamination=contamination, n_neighbors=neighbors)
    except AnomalyError as err:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        st.info("Check the selected columns and try again. Nothing was changed.")

res = st.session_state.get("anomaly_result")
if res is None:
    st.stop()

st.divider()
if res.n_total_rows != df.height:
    st.warning("These results belong to an earlier version of the data. Run detection again.")
a, b, c = st.columns(3)
a.metric("Rows analysed", f"{res.n_rows:,}")
b.metric("Anomalies", f"{res.n_anomalies:,}")
c.metric("Share", f"{res.pct:.2f}%")
st.caption(f"Method: {res.method}. A row counts as an anomaly if it is flagged in at least one selected "
           "column." if res.method in ("IQR", "Z-score") else f"Method: {res.method}.")
for note in res.notes:
    st.caption(f"ℹ {note}")

tab_plot, tab_cols, tab_rows, tab_act = st.tabs(["Visualization", "Affected variables", "Anomalous rows", "Actions"])
sig = "|".join(res.columns)

with tab_plot:
    if len(res.columns) >= 2:
        left, right = st.columns(2)
        x_col = left.selectbox("X axis", res.columns, index=0, key=f"an_x|{sig}")
        y_col = right.selectbox("Y axis", res.columns, index=1, key=f"an_y|{sig}")
        st.plotly_chart(scatter_fig(res, x_col, y_col))
        if len(res.columns) > 2:
            st.plotly_chart(projection_fig(res))
    d_col = st.selectbox("Distribution of", res.columns, key=f"an_d|{sig}")
    st.plotly_chart(distribution_fig(res, d_col))
    fig = score_fig(res)
    if fig is not None:
        st.plotly_chart(fig)

with tab_cols:
    st.plotly_chart(affected_fig(res))
    st.dataframe(res.per_column)
    if res.method in ("Isolation Forest", "Local Outlier Factor"):
        st.caption("Deviation = typical distance from the median in robust z-score units. "
                   "A high ratio means that variable is what makes the flagged rows unusual.")

with tab_rows:
    rows = anomaly_rows(df, res, limit=1000)
    st.caption("'row' is the 0-based position of the row in the dataset. At most 1,000 rows are shown.")
    st.dataframe(rows)
    st.download_button("Download flagged rows (CSV)", rows.write_csv(),
                       file_name="anomalies.csv", mime="text/csv")

with tab_act:
    st.write("Add a True/False column that marks the anomalies. No rows are removed, and the step can be undone.")
    name = st.text_input("Column name", value=f"is_anomaly_{METHOD_SLUG[res.method]}", key=f"an_name|{res.method}")
    if st.button("Add flag column"):
        try:
            new_df = flag_column(df, res, name.strip())
        except AnomalyError as err:
            st.error(err.message)
            if err.suggestion:
                st.info(f"What you can do: {err.suggestion}")
        else:
            apply_step(f"Flag anomalies ({res.method})", new_df)
            st.session_state["an_flash"] = (f"Added column '{name.strip()}'. It is empty for rows "
                                            "that were not analysed.")
            st.rerun()