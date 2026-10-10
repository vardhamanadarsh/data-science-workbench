"""Small built-in demo datasets (synthetic) so the app can be tried instantly."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from config import settings


def _churn(n: int = 1000) -> pl.DataFrame:
    rng = np.random.default_rng(settings.sample_seed)
    age = rng.integers(18, 80, n)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(20, 150).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    calls = rng.poisson(1.5, n)
    logit = -1.0 + 0.02 * (monthly - 70) - 0.03 * tenure + 0.4 * calls + 0.8 * (contract == "Month-to-month")
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    total = (monthly * tenure + rng.normal(0, 50, n)).round(2).tolist()
    for i in rng.choice(n, int(n * 0.03), replace=False):
        total[i] = None
    return pl.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract.tolist(),
            "support_calls": calls,
            "total_charges": total,
            "churned": churned,
        }
    )


def _monthly_sales() -> pl.DataFrame:
    rng = np.random.default_rng(settings.sample_seed)
    dates = pl.date_range(date(2015, 1, 1), date(2024, 12, 1), interval="1mo", eager=True)
    t = np.arange(dates.len())
    season = 15 * np.sin(2 * np.pi * t / 12)
    ad_spend = (50 + 10 * np.sin(2 * np.pi * t / 12 + 1) + rng.normal(0, 3, t.size)).round(2)
    sales = (100 + 0.8 * t + season + 0.3 * ad_spend + rng.normal(0, 4, t.size)).round(2)
    return pl.DataFrame({"date": dates, "sales": sales, "ad_spend": ad_spend})


SAMPLES = {
    "Customer churn (classification, synthetic)": ("demo_churn.parquet", _churn),
    "Monthly sales (time series, synthetic)": ("demo_monthly_sales.parquet", _monthly_sales),
}


def ensure_samples() -> dict[str, Path]:
    """Create the sample files if missing and return {label: path}."""
    settings.samples_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, (filename, builder) in SAMPLES.items():
        path = settings.samples_dir / filename
        if not path.exists():
            builder().write_parquet(path)
        paths[label] = path
    return paths