import streamlit as st

from visualization.plot_config import (
    AGG_KINDS, BAR_KINDS, MAX_HEATMAP_COLUMNS, MAX_PAIR_COLUMNS, PLOT_KINDS,
    SERIES_BAR_KINDS, PlotError, PlotSpec, PlotStyle, column_groups,
)
from visualization.plots import (
    apply_series_colors, build_figure, chart_config, export_figure, finalize,
    prepare_data, series_names,
)
from visualization.themes import PALETTES, THEMES

st.title("🎨 Plot Studio")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Load a dataset first.")
    st.page_link("views/upload.py", label="Go to Data Upload")
    st.stop()

df = dataset.df
mode = st.session_state["mode"]
groups = column_groups(df)
num, cat, tm = groups["numeric"], groups["categorical"], groups["datetime"]
ns = dataset.path.name

controls, canvas = st.columns([1, 3])

with controls:
    kind = st.selectbox("Plot type", PLOT_KINDS)
    display = st.radio("Display mode", ["Normal (static)", "Live (interactive)"],
                       help="Normal is a clean report-style image. Live adds hover, zoom, pan and legend toggling.")
    live = display.startswith("Live")

    def pick(label: str, options: list[str], optional: bool = False) -> str | None:
        if not optional and not options:
            st.warning(f"This plot needs a column for '{label}', but the dataset has none of the right type.")
            st.stop()
        opts = ["(none)", *options] if optional else options
        value = st.selectbox(label, opts, key=f"{ns}|{kind}|{label}")
        return None if value == "(none)" else value

    spec = PlotSpec(kind=kind)
    if kind == "Histogram":
        spec.x = pick("Variable", num)
        spec.color = pick("Color by", cat, optional=True)
        spec.bins = st.slider("Bins", 5, 200, 30)
    elif kind in ("KDE", "ECDF"):
        spec.x = pick("Variable", num)
        spec.color = pick("Color by", cat, optional=True)
    elif kind in ("Box plot", "Violin plot", "Strip plot"):
        spec.y = [pick("Value", num)]
        spec.x = pick("Group by", cat, optional=True)
    elif kind == "Q-Q plot":
        spec.x = pick("Variable", num)
    elif kind in ("Scatter", "Bubble", "Regression plot", "Joint plot"):
        spec.x = pick("X", num)
        spec.y = [pick("Y", num)]
        if kind == "Bubble":
            spec.size = pick("Size", num)
        spec.color = pick("Color by", cat, optional=True)
    elif kind == "Pair plot":
        spec.y = st.multiselect(f"Variables (2 to {MAX_PAIR_COLUMNS})", num, default=num[:4],
                                max_selections=MAX_PAIR_COLUMNS, key=f"{ns}|pair|vars")
        spec.color = pick("Color by", cat, optional=True)
        st.caption("Uses at most 2,000 rows to stay fast.")
    elif kind in AGG_KINDS:
        spec.x = pick("Category", cat)
        spec.agg = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"],
                                key=f"{ns}|{kind}|agg")
        if spec.agg != "count":
            spec.y = [pick("Value", num)]
        if kind in BAR_KINDS:
            others = [c for c in cat if c != spec.x]
            needs_series = kind in SERIES_BAR_KINDS
            spec.color = pick("Series" if needs_series else "Color by", others, optional=not needs_series)
    elif kind == "Radar":
        spec.y = st.multiselect("Variables (3 to 12)", num, default=num[:5], max_selections=12,
                                key=f"{ns}|radar|vars")
        spec.color = pick("Group by", cat, optional=True)
        st.caption("Each variable is scaled 0 to 1; shapes show group averages.")
    elif kind in ("Line", "Area"):
        spec.x = pick("X (time or number)", tm + num)
        options = [c for c in num if c != spec.x]
        spec.y = st.multiselect("Series", options, default=options[:1], key=f"{ns}|{kind}|series")
    elif kind == "Correlation heatmap":
        spec.y = st.multiselect("Variables", num, default=num[:8], max_selections=MAX_HEATMAP_COLUMNS,
                                key=f"{ns}|heat|vars")
        spec.corr_method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)

    with st.expander("Style", expanded=False):
        style = PlotStyle(
            theme=st.selectbox("Theme", list(THEMES)),
            palette=st.selectbox("Palette", list(PALETTES)),
            title=st.text_input("Title"),
            x_label=st.text_input("X axis label"),
            y_label=st.text_input("Y axis label"),
            font_size=st.slider("Font size", 8, 28, 14),
            line_width=st.slider("Line width", 1.0, 8.0, 2.0, 0.5),
            marker_size=st.slider("Marker size", 2, 20, 6),
            opacity=st.slider("Opacity", 0.1, 1.0, 0.85, 0.05),
            show_grid=st.checkbox("Grid", value=True),
            show_legend=st.checkbox("Legend", value=True),
        )

try:
    pdf = prepare_data(df, spec, mode)
    fig = build_figure(pdf, spec, style)
except PlotError as err:
    with canvas:
        st.error(err.message)
        if err.suggestion:
            st.info(f"What you can do: {err.suggestion}")
    st.stop()

with controls:
    names = series_names(fig)
    if names:
        with st.expander("Series colors (HEX)", expanded=False):
            palette = PALETTES[style.palette]
            colors = {
                name: st.color_picker(name, palette[i % len(palette)],
                                      key=f"clr|{style.palette}|{kind}|{name}")
                for i, name in enumerate(names[:12])
            }
        apply_series_colors(fig, colors)
        if len(names) > 12:
            st.caption("Colors can be changed for the first 12 series.")

finalize(fig, spec, live)
st.session_state["plot_used"] = True

with canvas:
    st.plotly_chart(fig, config=chart_config(live, filename=kind.lower().replace(" ", "_")))
    rows_note = f"{len(pdf):,} rows plotted"
    if dataset.sampled or len(pdf) < dataset.n_rows_loaded:
        rows_note += f" (limited by the '{mode.value}' mode or by aggregation)"
    st.caption(rows_note)

    left, middle, right = st.columns([1, 1, 2])
    fmt = left.selectbox("Export as", ["png", "svg", "pdf", "html"], label_visibility="collapsed")
    if middle.button("Prepare export"):
        try:
            with st.spinner("Exporting..."):
                st.session_state["plot_export"] = (fmt, export_figure(fig, fmt))
        except PlotError as err:
            st.error(err.message)
            st.info(err.suggestion)
    export = st.session_state.get("plot_export")
    if export:
        mime = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf", "html": "text/html"}
        right.download_button(f"Download {export[0].upper()}", data=export[1],
                              file_name=f"plot.{export[0]}", mime=mime[export[0]])
        st.caption("The prepared file shows the plot as it was when you clicked 'Prepare export'.")