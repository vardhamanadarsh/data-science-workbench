"""Plot settings, plot specification and column grouping."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

PLOT_KINDS: list[str] = [
    # Statistical
    "Histogram", "KDE", "Box plot", "Violin plot", "Strip plot", "ECDF", "Q-Q plot",
    # Relationship
    "Scatter", "Bubble", "Regression plot", "Joint plot", "Pair plot", "Correlation heatmap",
    # Comparison
    "Bar", "Grouped bar", "Stacked bar", "100% stacked bar", "Lollipop", "Dot plot", "Radar",
    # Ordered
    "Line", "Area",
]
ORDERED_KINDS = {"Line", "Area"}
BAR_KINDS = {"Bar", "Grouped bar", "Stacked bar", "100% stacked bar"}
SERIES_BAR_KINDS = {"Grouped bar", "Stacked bar", "100% stacked bar"}
AGG_KINDS = BAR_KINDS | {"Lollipop", "Dot plot"}
GROUPED_X_KINDS = AGG_KINDS | {"Box plot", "Violin plot", "Strip plot"}
NEEDS_Y = {
    "Scatter", "Bubble", "Regression plot", "Joint plot", "Pair plot", "Radar",
    "Box plot", "Violin plot", "Strip plot", "Line", "Area",
}
MAX_COLOR_GROUPS = 12
MAX_BAR_CATEGORIES = 50
MAX_HEATMAP_COLUMNS = 40
MAX_PAIR_COLUMNS = 6
PAIR_PLOT_MAX_ROWS = 2_000


class PlotError(Exception):
    """A user-facing plotting error with a suggestion."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


@dataclass
class PlotSpec:
    """WHAT to plot."""

    kind: str
    x: str | None = None
    y: list[str] = field(default_factory=list)
    color: str | None = None
    size: str | None = None
    bins: int = 30
    agg: str = "mean"
    corr_method: str = "pearson"


@dataclass
class PlotStyle:
    """HOW the plot looks."""

    theme: str = "Professional"
    palette: str = "Default"
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    font_size: int = 14
    line_width: float = 2.0
    marker_size: int = 6
    opacity: float = 0.85
    show_grid: bool = True
    show_legend: bool = True


def column_groups(df: pl.DataFrame) -> dict[str, list[str]]:
    """Split columns into numeric / categorical / datetime lists.

    Integer columns with at most 12 distinct values are also offered as categories.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    for name, dtype in df.schema.items():
        if dtype in (pl.Date, pl.Datetime):
            datetime.append(name)
        elif dtype == pl.Boolean or dtype in (pl.String, pl.Categorical, pl.Enum):
            categorical.append(name)
        elif dtype.is_numeric():
            numeric.append(name)
            if dtype.is_integer() and df[name].n_unique() <= MAX_COLOR_GROUPS:
                categorical.append(name)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}