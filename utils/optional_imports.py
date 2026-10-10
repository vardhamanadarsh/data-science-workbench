"""Graceful handling of optional dependencies.

Heavy packages are imported only when a feature is opened, and a missing
package produces a friendly message with the exact install command instead
of a crash.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from loguru import logger


@dataclass(frozen=True)
class OptionalPackage:
    """Describes one optional dependency."""

    import_name: str   # name used in `import ...`
    pip_name: str      # name used with pip / uv
    extra: str         # extra group in pyproject.toml
    purpose: str       # shown to the user


REGISTRY: dict[str, OptionalPackage] = {
    p.import_name: p
    for p in [
        OptionalPackage("xgboost", "xgboost", "ml", "XGBoost models"),
        OptionalPackage("lightgbm", "lightgbm", "ml", "LightGBM models"),
        OptionalPackage("imblearn", "imbalanced-learn", "ml", "SMOTE and resampling"),
        OptionalPackage("optuna", "optuna", "ml", "Hyperparameter tuning"),
        OptionalPackage("shap", "shap", "explain", "SHAP explanations"),
        OptionalPackage("dice_ml", "dice-ml", "explain", "Counterfactual explanations"),
        OptionalPackage("fairlearn", "fairlearn", "explain", "Fairness and bias checks"),
        OptionalPackage("flaml", "flaml", "automl", "Lightweight AutoML"),
        OptionalPackage("lazypredict", "lazypredict", "automl", "Quick model comparison"),
        OptionalPackage("statsforecast", "statsforecast", "timeseries", "Auto-ARIMA and fast forecasting"),
        OptionalPackage("ruptures", "ruptures", "timeseries", "Change-point detection"),
        OptionalPackage("arch", "arch", "timeseries", "GARCH and Phillips-Perron test"),
        OptionalPackage("linearmodels", "linearmodels", "econometrics", "Panel data and IV models"),
        OptionalPackage("pingouin", "pingouin", "econometrics", "Statistical tests with effect sizes"),
        OptionalPackage("scikit_posthocs", "scikit-posthocs", "econometrics", "Post-hoc tests"),
        OptionalPackage("pandera", "pandera", "quality", "Schema validation"),
        OptionalPackage("ydata_profiling", "ydata-profiling", "quality", "Full EDA report"),
        OptionalPackage("evidently", "evidently", "quality", "Data drift reports"),
        OptionalPackage("missingno", "missingno", "quality", "Missing-value pattern plots"),
        OptionalPackage("pygwalker", "pygwalker", "explore", "Drag-and-drop chart builder"),
        OptionalPackage("jinja2", "jinja2", "reports", "HTML report templates"),
        OptionalPackage("xlsxwriter", "xlsxwriter", "reports", "Excel export"),
        OptionalPackage("skops", "skops", "reports", "Safe model saving"),
        OptionalPackage("dowhy", "dowhy", "causal", "Causal inference"),
    ]
}


class MissingDependencyError(ImportError):
    """Raised when an optional package is not installed or fails to import."""

    def __init__(self, import_name: str) -> None:
        self.import_name = import_name
        super().__init__(install_hint(import_name))


@lru_cache(maxsize=None)
def is_available(import_name: str) -> bool:
    """Check if a package is installed WITHOUT importing it (fast)."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def install_hint(import_name: str) -> str:
    """Return a human-readable install instruction for a package."""
    pkg = REGISTRY.get(import_name)
    if pkg is None:
        return f"The package '{import_name}' is not installed."
    return (
        f"'{pkg.pip_name}' is not installed ({pkg.purpose}). "
        f"Install it with:  uv sync --extra {pkg.extra}   "
        f'or  pip install "workbench[{pkg.extra}]"'
    )


def lazy_import(import_name: str) -> ModuleType:
    """Import an optional package on demand.

    Raises:
        MissingDependencyError: if the package is missing or cannot be imported.
    """
    if not is_available(import_name):
        raise MissingDependencyError(import_name)
    try:
        return importlib.import_module(import_name)
    except Exception as exc:  # broken install, binary mismatch, etc.
        logger.warning("Import of {} failed: {}", import_name, exc)
        raise MissingDependencyError(import_name) from exc


def require(import_name: str) -> ModuleType | None:
    """Streamlit-friendly import: shows a message and returns None if missing.

    Usage inside a page:
        shap = require("shap")
        if shap is None:
            return
    """
    try:
        return lazy_import(import_name)
    except MissingDependencyError as err:
        import streamlit as st  # imported here so this module works without a UI

        pkg = REGISTRY.get(import_name)
        st.warning(f"**Optional feature unavailable:** {err}")
        if pkg is not None:
            st.code(f"uv sync --extra {pkg.extra}", language="bash")
        return None


def availability_report() -> list[dict[str, str]]:
    """Status of every optional package (used on the Settings page)."""
    return [
        {
            "Package": p.pip_name,
            "Feature": p.purpose,
            "Extra": p.extra,
            "Status": "Installed" if is_available(p.import_name) else "Not installed",
        }
        for p in REGISTRY.values()
    ]