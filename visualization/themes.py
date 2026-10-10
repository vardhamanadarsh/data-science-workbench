"""Colour palettes and Plotly templates."""

from __future__ import annotations

PALETTES: dict[str, list[str]] = {
    "Default": ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD",
                "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
    "Color-blind friendly": ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                             "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "Colorful": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                 "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
    "Monochrome": ["#111111", "#4D4D4D", "#808080", "#A6A6A6", "#CCCCCC"],
}

# Display name -> built-in Plotly template
THEMES: dict[str, str] = {
    "Professional": "plotly_white",
    "Publication": "simple_white",
    "Colorful": "seaborn",
    "Dark": "plotly_dark",
}