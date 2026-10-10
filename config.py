"""Central, typed configuration for the Workbench application."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent


class ProcessingMode(str, Enum):
    """How much data the app is allowed to process at once."""

    FAST = "Fast Preview"
    STANDARD = "Standard Analysis"
    FULL = "Full Analysis"


# Maximum rows used for plots / quick EDA in each mode (None = no limit).
MODE_ROW_LIMITS: dict[ProcessingMode, int | None] = {
    ProcessingMode.FAST: 50_000,
    ProcessingMode.STANDARD: 500_000,
    ProcessingMode.FULL: None,
}


class Settings(BaseSettings):
    """Application settings. Can be overridden with WORKBENCH_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="WORKBENCH_", extra="ignore")

    app_name: str = "Data Science Workbench"
    app_icon: str = "🔬"
    version: str = "0.1.0"

    # Privacy: no data leaves the machine unless the user enables it.
    local_processing_mode: bool = True

    # File handling
    max_upload_mb: int = Field(default=500, ge=1)
    max_memory_fraction: float = Field(default=0.6, gt=0, le=1)
    allowed_extensions: tuple[str, ...] = (
        ".csv", ".tsv", ".txt", ".xls", ".xlsx", ".json", ".parquet",
    )

    # Performance
    default_mode: ProcessingMode = ProcessingMode.STANDARD
    large_file_mb: int = 100          # above this, prefer DuckDB over eager loading
    sample_seed: int = 42

    # Paths
    data_dir: Path = BASE_DIR / "data"
    samples_dir: Path = BASE_DIR / "data" / "samples"
    experiments_dir: Path = BASE_DIR / "experiments"
    cache_dir: Path = BASE_DIR / ".cache"


settings = Settings()