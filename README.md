# Data Science Workbench

A modular, local-first data science app built with Streamlit. Load a dataset, check its health, explore it,
clean it, train and compare models, explain them, look for leakage, anomalies and unfair group differences,
compare datasets for drift, and export a report. Free and open-source tools only, no cloud services.

![CI](https://github.com/vardhamanadarsh/data-science-workbench/actions/workflows/ci.yml/badge.svg)

## Features

| Area | What it does |
|---|---|
| Data loading | CSV, TSV, Excel, JSON, Parquet. Polars by default, DuckDB for very large files, three processing modes (50k / 500k / all rows) |
| Data Health | 0 to 100 score, missing values, duplicates, constant columns, outliers, suspicious placeholders, class imbalance |
| EDA and Plot Studio | Automated summaries and 20+ interactive Plotly chart types |
| Preprocessing | Cleaning, transforms, feature engineering, with undo and redo. Nothing is deleted automatically |
| Machine Learning | Classification and regression with scikit-learn, XGBoost and LightGBM, a Baseline for honesty, cross-validation, Optuna tuning, calibration and learning curves |
| AutoML | Native model comparison and optional FLAML |
| Explainability | SHAP, permutation importance, what-if analysis, counterfactual search |
| Checks | Leakage detection, fairness comparison across groups, anomaly detection |
| Compare & Drift | Schema, missing-value and distribution changes between two datasets (PSI and KS) |
| Guidance | "What should I do?" advice and plain-English explanations of every metric, computed from your actual results |
| Reports | HTML report (3 templates, print to PDF), JSON, Excel, CSV, LaTeX tables, models saved with skops, project save and open |
| Experiments | Local JSON experiment tracking |

## Design principles

- **No leakage by construction:** imputing, scaling and encoding happen inside a scikit-learn Pipeline, fitted on training rows only.
- **Honest language:** leakage and fairness findings are always "possible", never certain.
- **You decide:** the app suggests and never deletes or changes your data on its own.
- **Local:** your data stays on your machine.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --extra ml --extra automl --extra explain --extra reports
uv run streamlit run app.py
```

Open http://localhost:8501 and try one of the demo datasets from the Data Upload page.

## Optional extras

The core install is large because of the numeric stack (Polars, PyArrow, SciPy, scikit-learn, DuckDB, statsmodels).
Add only the extras you need. Sizes are the real installed size on disk (Python 3.11, without dev tools),
measured with `scripts/measure_sizes.ps1`. Each row is a full install, so it includes the core.

| Setup | Installed size | Added over core |
|---|---|---|
| Core only | 875 MB | |
| Core + `ml` (XGBoost, LightGBM, imbalanced-learn, Optuna) | 1,063 MB | +188 MB |
| Core + `explain` (SHAP) | 991 MB | +116 MB |
| Core + `automl` (FLAML) | 1,050 MB | +175 MB |
| Core + `reports` (xlsxwriter, skops, jinja2) | 879 MB | +4 MB |
| All of the above | 1,184 MB | +309 MB |

The extras overlap (FLAML brings its own LightGBM and XGBoost), so the total is less than the sum of the parts.
Features whose extra is missing show an install hint instead of crashing.

## Tests

```bash
uv run pytest -q
```

## Docker

```bash
docker build -t workbench .
docker run -p 8501:8501 workbench
```

Add extras at build time:

```bash
docker build --build-arg EXTRAS="--extra ml --extra explain --extra reports --extra automl" -t workbench .
```

## Project layout

```
app.py            entry point and navigation
config.py         typed settings
core/             analysis logic, no UI code (testable on its own)
views/            one Streamlit page per file
visualization/    plot builders and themes
utils/            caching, optional imports, helpers
tests/            pytest suite
```

## Limitations

- Trained models and plots are not stored in project files. Save models separately as `.skops` files.
- Explanations (SHAP, counterfactuals) describe model behaviour, not real-world causes.

## Licence

MIT. See the LICENSE file.
