from dataclasses import replace

import polars as pl
import streamlit as st

from core.feature_engineering import (
    INTERACTION_OPS, ROLLING_STATS, add_date_parts, add_interaction, add_lags, add_rolling,
    date_parts_for, leakage_notes,
)
from core.history import DataHistory
from core.preprocessing import (
    TYPE_TARGETS, PreprocessError, convert_type, count_duplicates, drop_columns,
    drop_duplicate_rows, drop_high_missing_columns, drop_rows_with_missing, fill_missing,
    handle_outliers, missing_count, outlier_table, sort_rows, transform_numeric,
)

st.title("🧹 Preprocessing")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

history: DataHistory | None = st.session_state.get("history")
if history is None:
    history = DataHistory(dataset.df)
    st.session_state["history"] = history

df: pl.DataFrame = dataset.df
all_cols = df.columns
num = [c for c, t in df.schema.items() if t.is_numeric()]
dates = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
target = st.session_state.get("target_col")

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)


def sync() -> None:
    """Make the rest of the app use the current history state."""
    st.session_state["dataset"] = replace(dataset, df=history.current.df)
    for key in ("health", "eda"):
        st.session_state.pop(key, None)


def run(label: str, operation) -> None:
    """Run an operation, record it in the history and refresh the page."""
    try:
        result = operation()
    except PreprocessError as err:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
        return
    except Exception as exc:
        st.error(f"The operation failed: {exc}")
        st.info("Check the selected columns and try again. Nothing was changed.")
        return
    if not result.changed:
        st.info(result.message)
        return
    history.push(label, result.df)
    sync()
    st.session_state["flash"] = result.message
    st.rerun()


# ---- Header ---------------------------------------------------------------
cells = sum(missing_count(df[c]) for c in all_cols)
m = st.columns(4)
m[0].metric("Rows", f"{df.height:,}")
m[1].metric("Columns", df.width)
m[2].metric("Missing cells", f"{cells:,}")
m[3].metric("Steps applied", history.index)

undo_col, redo_col, _ = st.columns([1, 1, 6])
if undo_col.button("Undo", disabled=not history.can_undo):
    history.undo()
    sync()
    st.rerun()
if redo_col.button("Redo", disabled=not history.can_redo):
    history.redo()
    sync()
    st.rerun()
st.caption("Nothing changes until you click an Apply button, and every step can be undone.")

tab_clean, tab_transform, tab_fe, tab_history = st.tabs(
    ["Clean", "Transform", "Feature engineering", "History"]
)

# ---- Clean ----------------------------------------------------------------
with tab_clean:
    with st.expander("Drop columns"):
        cols = st.multiselect("Columns to drop", all_cols, key="pp_drop_cols")
        if st.button("Drop columns", key="pp_drop_btn", disabled=not cols):
            run(f"Drop {len(cols)} column(s)", lambda: drop_columns(df, cols))

    with st.expander("Missing values"):
        has_missing = [c for c in all_cols if missing_count(df[c]) > 0]
        if not has_missing:
            st.success("No missing values in this dataset.")
        else:
            action = st.radio(
                "What do you want to do?",
                ["Fill missing values", "Drop rows with missing values", "Drop columns that are mostly empty"],
                key="pp_miss_action",
            )
            if action == "Fill missing values":
                cols = st.multiselect("Columns", has_missing, key="pp_fill_cols")
                method = st.selectbox("Fill with", ["median", "mean", "mode", "constant"], key="pp_fill_method")
                constant = st.text_input("Constant value", key="pp_fill_const") if method == "constant" else None
                if method != "constant":
                    st.warning(
                        "The mean, median and mode are computed from ALL rows. If you plan to train a model, "
                        "prefer imputing inside the model pipeline (next step), which learns from training rows only."
                    )
                if st.button("Fill missing values", key="pp_fill_btn", disabled=not cols):
                    run(f"Fill missing ({method})", lambda: fill_missing(df, cols, method, constant))
            elif action == "Drop rows with missing values":
                cols = st.multiselect("Drop rows that have a missing value in", has_missing, key="pp_droprows_cols")
                if cols:
                    st.caption(f"{df.drop_nulls(subset=cols).height:,} of {df.height:,} rows would remain.")
                if st.button("Drop rows", key="pp_droprows_btn", disabled=not cols):
                    run(f"Drop rows with missing values ({len(cols)} col.)", lambda: drop_rows_with_missing(df, cols))
            else:
                threshold = st.slider("Drop columns that are at least this % empty", 10, 100, 50, key="pp_thr")
                would = [c for c in all_cols if missing_count(df[c]) / max(df.height, 1) * 100 >= threshold]
                st.caption(f"Would drop: {', '.join(would) if would else 'nothing'}")
                if st.button("Drop columns", key="pp_dropmiss_btn", disabled=not would):
                    run(f"Drop columns >= {threshold}% empty", lambda: drop_high_missing_columns(df, float(threshold)))

    with st.expander("Duplicate rows"):
        subset = st.multiselect("Consider only these columns (empty = all columns)", all_cols, key="pp_dup_cols")
        n_dup = count_duplicates(df, subset or None)
        st.write(f"{n_dup:,} duplicate row(s) found.")
        if st.button("Remove duplicates", key="pp_dup_btn", disabled=n_dup == 0):
            run("Remove duplicate rows", lambda: drop_duplicate_rows(df, subset or None))

    with st.expander("Convert column type"):
        col = st.selectbox("Column", all_cols, key="pp_conv_col")
        st.caption(f"Current type: {df.schema[col]}")
        target_type = st.selectbox("Convert to", TYPE_TARGETS, key="pp_conv_type")
        if st.button("Convert", key="pp_conv_btn"):
            run(f"Convert '{col}' to {target_type}", lambda: convert_type(df, col, target_type))

    with st.expander("Outliers"):
        if not num:
            st.info("No numeric columns.")
        else:
            cols = st.multiselect("Numeric columns", num, key="pp_out_cols")
            factor = st.slider("IQR factor (1.5 is standard)", 1.0, 5.0, 1.5, 0.1, key="pp_out_factor")
            how = st.radio("Action", ["Clip to the limits", "Remove rows"], horizontal=True, key="pp_out_how")
            if cols:
                st.dataframe(outlier_table(df, cols, factor))
                st.caption("Preview only. Clipping turns the selected columns into decimal columns.")
            if st.button("Apply", key="pp_out_btn", disabled=not cols):
                method = "clip" if how.startswith("Clip") else "remove"
                run(f"Outliers: {how.lower()} ({len(cols)} col.)", lambda: handle_outliers(df, cols, factor, method))

    with st.expander("Sort rows"):
        col = st.selectbox("Sort by", dates + [c for c in all_cols if c not in dates], key="pp_sort_col")
        desc = st.checkbox("Descending", key="pp_sort_desc")
        if st.button("Sort", key="pp_sort_btn"):
            run(f"Sort by '{col}'", lambda: sort_rows(df, col, desc))

# ---- Transform ------------------------------------------------------------
with tab_transform:
    if not num:
        st.info("No numeric columns.")
    else:
        cols = st.multiselect("Numeric columns", num, key="pp_tf_cols")
        how = st.radio("Transform", ["log1p", "sqrt"], horizontal=True, key="pp_tf_how")
        st.caption("Creates new columns (for example income_log1p). The originals are kept.")
        if st.button("Add transformed columns", key="pp_tf_btn", disabled=not cols):
            run(f"Transform {how} ({len(cols)} col.)", lambda: transform_numeric(df, cols, how))
    st.info(
        "Scaling, categorical encoding and feature selection are done inside the model pipeline "
        "(next step) so they are fitted on training rows only."
    )

# ---- Feature engineering --------------------------------------------------
with tab_fe:
    kind_label = st.radio(
        "Feature type", ["Date parts", "Lags", "Rolling statistics", "Interactions"],
        horizontal=True, key="fe_kind",
    )
    order_options = ["(not checked)", *dates, *[c for c in all_cols if c not in dates]]
    order_index = 1 if dates else 0
    operation = None
    sources: list[str] = []
    label = ""
    kind = ""

    if kind_label == "Date parts":
        kind = "date"
        if not dates:
            st.info("No date column found. Convert one in Clean, 'Convert column type'.")
        else:
            col = st.selectbox("Date column", dates, key="fe_date_col")
            parts = st.multiselect("Parts", date_parts_for(df.schema[col]), default=["year", "month"], key="fe_date_parts")
            if parts:
                sources, label = [col], f"Date parts of '{col}'"
                operation = lambda: add_date_parts(df, col, parts)
    elif kind_label == "Lags":
        kind = "lag"
        cols = st.multiselect("Numeric columns", num, key="fe_lag_cols")
        lags = st.multiselect("Lags (rows back)", [1, 2, 3, 5, 7, 12, 24, 52], default=[1], key="fe_lags")
        order_by = st.selectbox("Rows are ordered by (checked for you)", order_options, index=order_index, key="fe_lag_order")
        if cols and lags:
            sources, label = cols, f"Lag features ({len(cols)} col.)"
            operation = lambda: add_lags(df, cols, lags, None if order_by == "(not checked)" else order_by)
    elif kind_label == "Rolling statistics":
        kind = "rolling"
        cols = st.multiselect("Numeric columns", num, key="fe_roll_cols")
        windows = st.multiselect("Window sizes (rows)", [2, 3, 5, 7, 14, 30], default=[3], key="fe_roll_win")
        stats = st.multiselect("Statistics", ROLLING_STATS, default=["mean"], key="fe_roll_stats")
        exclude = st.checkbox("Exclude the current row (recommended)", value=True, key="fe_roll_excl")
        order_by = st.selectbox("Rows are ordered by (checked for you)", order_options, index=order_index, key="fe_roll_order")
        if cols and windows and stats:
            sources, label = cols, f"Rolling statistics ({len(cols)} col.)"
            operation = lambda: add_rolling(df, cols, windows, stats, exclude, None if order_by == "(not checked)" else order_by)
    else:
        kind = "interaction"
        if len(num) < 2:
            st.info("At least two numeric columns are needed.")
        else:
            a = st.selectbox("First column", num, key="fe_int_a")
            b = st.selectbox("Second column", [c for c in num if c != a], key="fe_int_b")
            op = st.selectbox("Operation", INTERACTION_OPS, key="fe_int_op")
            sources, label = [a, b], f"Interaction: {a} {op} {b}"
            operation = lambda: add_interaction(df, a, b, op)

    for note in leakage_notes(sources, target, kind):
        st.warning(note)
    if operation is not None and st.button("Add features", type="primary", key="fe_apply"):
        run(label, operation)

# ---- History --------------------------------------------------------------
with tab_history:
    st.dataframe(pl.DataFrame(history.timeline()))
    st.caption(f"The last {history.max_snapshots} states are kept in memory (the original is always kept).")
    step = st.selectbox(
        "Go to step",
        range(len(history.snapshots)),
        index=history.index,
        format_func=lambda i: f"{i}: {history.snapshots[i].label}",
        key="hist_step",
    )
    if st.button("Go to this step", disabled=step == history.index):
        history.rollback_to(step)
        sync()
        st.rerun()