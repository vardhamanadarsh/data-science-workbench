"""Safe, engine-aware data loading.

Engine selection logic (documented for the README "Design Decisions"):
  * Excel (.xls/.xlsx)                -> pandas readers, converted to Polars
  * JSON                              -> Polars (with fallbacks)
  * space / fixed-width text          -> Polars / pandas (small files only)
  * big CSV/TSV/TXT/Parquet + a row limit for the chosen mode
                                      -> DuckDB (scans the file, never loads it
                                         all; returns a sample)
  * everything else                   -> Polars eager read (fast, low memory)

The loader is UI-free. Caching is applied in the page layer.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import duckdb
import pandas as pd
import polars as pl
from loguru import logger

from config import MODE_ROW_LIMITS, ProcessingMode, settings

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EXT_TO_TYPE: dict[str, str] = {
    ".csv": "csv", ".tsv": "tsv", ".txt": "txt", ".xls": "xls",
    ".xlsx": "xlsx", ".json": "json", ".parquet": "parquet",
}
TEXT_TYPES = {"csv", "tsv", "txt"}
DUCKDB_TYPES = {"csv", "tsv", "txt", "parquet"}

DELIMITERS: dict[str, str] = {
    "Comma (,)": ",",
    "Tab": "\t",
    "Semicolon (;)": ";",
    "Pipe (|)": "|",
    "Space": " ",
    "Fixed-width": "fixed",
}

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# Rough RAM expansion factor when a file is fully loaded.
_MEMORY_FACTOR = {"csv": 3, "tsv": 3, "txt": 3, "json": 5, "parquet": 6, "xlsx": 8, "xls": 8}


class DataLoadError(Exception):
    """A user-facing loading error with a suggestion on what to do."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message} {suggestion}".strip())


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoadOptions:
    """User choices that influence loading."""

    delimiter: str | None = None      # None = auto-detect (text files only)
    sheet: str | None = None          # Excel sheet name (None = first sheet)
    mode: ProcessingMode = ProcessingMode.STANDARD


@dataclass(frozen=True)
class EngineChoice:
    engine: str   # "polars" | "duckdb" | "pandas"
    reason: str


@dataclass
class LoadedDataset:
    """A loaded dataset plus everything the UI needs to describe it."""

    name: str
    path: Path
    file_type: str
    file_size_bytes: int
    engine: str
    engine_reason: str
    df: pl.DataFrame
    total_rows: int                   # rows in the FILE (df may be a sample)
    sampled: bool
    delimiter: str | None
    sheet: str | None
    mode: ProcessingMode
    load_seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows_loaded(self) -> int:
        return self.df.height

    @property
    def n_cols(self) -> int:
        return self.df.width

    @property
    def memory_bytes(self) -> int:
        return int(self.df.estimated_size())

    def summary(self) -> dict[str, str | int]:
        return {
            "File": self.name,
            "Type": self.file_type.upper(),
            "File size": human_size(self.file_size_bytes),
            "Rows in file": self.total_rows,
            "Rows loaded": self.n_rows_loaded,
            "Columns": self.n_cols,
            "Memory (loaded)": human_size(self.memory_bytes),
            "Engine": self.engine,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def human_size(num_bytes: float) -> str:
    """Format bytes as a readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _sql_str(value: str) -> str:
    """Quote a value as a SQL string literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


def _available_memory() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # psutil missing or unsupported
        return None


# --------------------------------------------------------------------------- #
# Validation and saving uploads
# --------------------------------------------------------------------------- #

def detect_file_type(path: Path) -> str:
    """Return the file type from the extension (validated later by content)."""
    ext = path.suffix.lower()
    if ext not in EXT_TO_TYPE or ext not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise DataLoadError(f"Unsupported file type '{ext or 'none'}'.", f"Allowed types: {allowed}.")
    return EXT_TO_TYPE[ext]


def validate_file(path: Path) -> str:
    """Check extension, size and file signature. Returns the file type.

    Nothing in the file is ever executed; we only read the first bytes.
    """
    if not path.is_file():
        raise DataLoadError("File not found.", "Upload the file again.")
    ftype = detect_file_type(path)
    size = path.stat().st_size
    if size == 0:
        raise DataLoadError("The file is empty.", "Check that the export finished correctly.")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise DataLoadError(
            f"File is {human_size(size)}, above the {settings.max_upload_mb} MB limit.",
            "Raise WORKBENCH_MAX_UPLOAD_MB or convert the file to Parquet.",
        )

    with path.open("rb") as handle:
        head = handle.read(8192)

    if ftype == "parquet" and head[:4] != b"PAR1":
        raise DataLoadError("This is not a valid Parquet file.", "Re-export it as Parquet or CSV.")
    if ftype == "xlsx" and head[:2] != b"PK":
        raise DataLoadError("This is not a valid .xlsx file.", "Open it in Excel and save it again.")
    if ftype == "xls" and head[:8] != _OLE_MAGIC:
        raise DataLoadError(
            "This does not look like a real .xls file (it may be HTML or CSV renamed).",
            "Open it in Excel and save as .xlsx or .csv.",
        )
    if ftype in TEXT_TYPES | {"json"} and b"\x00" in head:
        raise DataLoadError(
            "The file contains binary data or uses UTF-16 encoding.",
            "Save it as UTF-8 text and upload again.",
        )
    return ftype


def save_upload(filename: str, fileobj: BinaryIO) -> Path:
    """Stream an uploaded file to disk (never held fully in RAM).

    The file name is sanitised and prefixed with a content hash, so the same
    upload always maps to the same path (good for caching).
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or "upload"
    detect_file_type(Path(safe))  # fail early on bad extensions

    upload_dir = settings.cache_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / f".tmp_{safe}"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    hasher = hashlib.blake2b(digest_size=8)
    written = 0

    if hasattr(fileobj, "seek"):
        fileobj.seek(0)
    try:
        with tmp.open("wb") as out:
            while chunk := fileobj.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise DataLoadError(
                        f"File exceeds the {settings.max_upload_mb} MB limit.",
                        "Convert it to Parquet or raise WORKBENCH_MAX_UPLOAD_MB.",
                    )
                hasher.update(chunk)
                out.write(chunk)
        final = upload_dir / f"{hasher.hexdigest()}_{safe}"
        tmp.replace(final)
        return final
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Inspecting files
# --------------------------------------------------------------------------- #

def sniff_delimiter(path: Path, file_type: str) -> str:
    """Guess the delimiter from the first lines of a text file."""
    if file_type == "tsv":
        return "\t"
    with path.open("rb") as handle:
        head = handle.read(65536).decode("utf-8", errors="replace")
    lines = [ln for ln in head.splitlines() if ln.strip()][:20]
    if not lines:
        return ","

    best, best_score = ",", 0.0
    for delim in (",", "\t", ";", "|"):
        counts = [ln.count(delim) for ln in lines]
        if min(counts) == 0:
            continue
        score = min(counts) * (1.0 if len(set(counts)) == 1 else 0.5)
        if score > best_score:
            best, best_score = delim, score
    if best_score > 0:
        return best

    space_counts = [len(ln.split()) for ln in lines]
    if min(space_counts) > 1 and len(set(space_counts)) == 1:
        return " "
    return ","


def list_excel_sheets(path: Path) -> list[str]:
    """Sheet names of an Excel workbook."""
    try:
        with pd.ExcelFile(path) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except Exception as exc:
        raise DataLoadError("Could not open the Excel file.", f"Details: {exc}") from exc


def choose_engine(
    file_type: str, size_bytes: int, mode: ProcessingMode, delimiter: str | None = None
) -> EngineChoice:
    """Pick the best engine for this file and processing mode."""
    size_mb = size_bytes / (1024 * 1024)
    if file_type in {"xls", "xlsx"}:
        return EngineChoice("pandas", "Excel readers return pandas; the result is converted to Polars.")
    if file_type == "json":
        return EngineChoice("polars", "Polars reads JSON and NDJSON directly.")
    if file_type in TEXT_TYPES and delimiter in {" ", "fixed"}:
        return EngineChoice("polars", "Whitespace-separated text needs pre-processing (small files only).")
    limit = MODE_ROW_LIMITS[mode]
    if file_type in DUCKDB_TYPES and size_mb >= settings.large_file_mb and limit is not None:
        return EngineChoice(
            "duckdb",
            f"File is {size_mb:.0f} MB (>= {settings.large_file_mb} MB): DuckDB scans it and "
            f"returns a sample of about {limit:,} rows instead of loading everything.",
        )
    return EngineChoice("polars", "Polars eager read: fastest option for files that fit in memory.")


def _check_memory(file_type: str, size_bytes: int) -> None:
    available = _available_memory()
    if available is None:
        return
    estimate = size_bytes * _MEMORY_FACTOR.get(file_type, 4)
    if estimate > available * settings.max_memory_fraction:
        raise DataLoadError(
            f"Loading this file fully needs about {human_size(estimate)} of RAM, "
            f"but only {human_size(available)} is free.",
            "Switch to Fast Preview or Standard Analysis, or convert the file to Parquet.",
        )


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def _read_delimited(source: Path | io.BytesIO, sep: str) -> pl.DataFrame:
    kwargs = {"separator": sep, "try_parse_dates": True, "encoding": "utf8-lossy"}
    try:
        return pl.read_csv(source, infer_schema_length=10_000, **kwargs)
    except pl.exceptions.ComputeError:
        # A later row did not match the inferred type: re-scan the whole file.
        if isinstance(source, io.BytesIO):
            source.seek(0)
        return pl.read_csv(source, infer_schema_length=None, **kwargs)


def _from_pandas(pdf: pd.DataFrame) -> pl.DataFrame:
    try:
        return pl.from_pandas(pdf)
    except Exception:
        # Mixed-type object columns: store them as text.
        pdf = pdf.copy()
        for col in pdf.select_dtypes(include="object").columns:
            pdf[col] = pdf[col].map(lambda v: None if pd.isna(v) else str(v))
        return pl.from_pandas(pdf)


def _read_whitespace(path: Path) -> pl.DataFrame:
    if path.stat().st_size > settings.large_file_mb * 1024 * 1024:
        raise DataLoadError(
            "Space-separated files above the large-file limit are not supported.",
            "Convert the file to CSV or Parquet.",
        )
    data = re.sub(rb"(?m)^[ \t]+", b"", path.read_bytes())
    data = re.sub(rb"[ \t]+", b" ", data)
    return _read_delimited(io.BytesIO(data), " ")


def _read_json(path: Path) -> pl.DataFrame:
    for reader in (pl.read_json, pl.read_ndjson):
        try:
            return reader(path)
        except Exception:
            continue
    try:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except Exception as exc:
        raise DataLoadError("The JSON file could not be parsed.", f"Details: {exc}") from exc

    records = obj if isinstance(obj, list) else None
    if records is None and isinstance(obj, dict):
        records = next(
            (v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None
        )
    if not records:
        raise DataLoadError(
            "No table-like records were found in this JSON file.",
            "Use a list of objects, e.g. [{...}, {...}], or one JSON object per line.",
        )
    return pl.DataFrame(records, infer_schema_length=None)


def _duckdb_source(path: Path, file_type: str, sep: str | None) -> str:
    quoted = _sql_str(path.as_posix())
    if file_type == "parquet":
        return f"read_parquet({quoted})"
    return f"read_csv({quoted}, delim={_sql_str(sep or ',')}, header=true)"


def _duckdb_load(
    path: Path, file_type: str, sep: str | None, limit: int | None
) -> tuple[pl.DataFrame, int, bool]:
    """Scan with DuckDB; return (frame, total rows in file, was_sampled).

    Uses Bernoulli sampling, which streams through the file and keeps the
    original row order (important for time series).
    """
    source = _duckdb_source(path, file_type, sep)
    con = duckdb.connect(":memory:")
    try:
        total = int(con.execute(f"SELECT count(*) FROM {source}").fetchone()[0])
        if limit is None or total <= limit:
            return con.execute(f"SELECT * FROM {source}").pl(), total, False
        percent = min(100.0, limit / total * 100)
        query = f"SELECT * FROM {source} USING SAMPLE {percent:.6f}% (bernoulli, {settings.sample_seed})"
        return con.execute(query).pl(), total, True
    finally:
        con.close()


def _clean_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace from column names and name empty columns."""
    new_names: list[str] = []
    seen: set[str] = set()
    for i, name in enumerate(df.columns):
        clean = str(name).strip() or f"column_{i + 1}"
        base, n = clean, 2
        while clean in seen:
            clean, n = f"{base}_{n}", n + 1
        seen.add(clean)
        new_names.append(clean)
    return df.rename(dict(zip(df.columns, new_names))) if new_names != df.columns else df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_dataset(
    path: str | Path, options: LoadOptions | None = None, display_name: str | None = None
) -> LoadedDataset:
    """Validate and load a file into a Polars DataFrame.

    Raises:
        DataLoadError: with a user-friendly message and suggestion.
    """
    options = options or LoadOptions()
    path = Path(path)
    file_type = validate_file(path)
    size = path.stat().st_size
    started = time.perf_counter()
    warnings: list[str] = []

    sep = (options.delimiter or sniff_delimiter(path, file_type)) if file_type in TEXT_TYPES else None
    choice = choose_engine(file_type, size, options.mode, sep)
    limit = MODE_ROW_LIMITS[options.mode]
    sampled = False
    sheet = options.sheet

    try:
        if choice.engine == "duckdb":
            df, total_rows, sampled = _duckdb_load(path, file_type, sep, limit)
            if sampled:
                warnings.append(
                    f"Large file: a representative sample of {df.height:,} of {total_rows:,} rows "
                    f"was loaded. Switch to Full Analysis to load everything."
                )
        else:
            _check_memory(file_type, size)
            if file_type in {"xls", "xlsx"}:
                sheet = sheet or list_excel_sheets(path)[0]
                df = _from_pandas(pd.read_excel(path, sheet_name=sheet))
            elif file_type == "json":
                df = _read_json(path)
            elif file_type == "parquet":
                df = pl.read_parquet(path)
            elif sep == "fixed":
                df = _from_pandas(pd.read_fwf(path))
            elif sep == " ":
                df = _read_whitespace(path)
            else:
                df = _read_delimited(path, sep or ",")
            total_rows = df.height
    except DataLoadError:
        raise
    except MemoryError as exc:
        raise DataLoadError(
            "Not enough memory to load this file.", "Use Fast Preview or convert the file to Parquet."
        ) from exc
    except Exception as exc:
        logger.exception("Failed to load {}", path.name)
        raise DataLoadError(
            "The file could not be read.",
            f"Check the delimiter and encoding. Details: {exc}",
        ) from exc

    if df.width == 0 or df.height == 0:
        raise DataLoadError(
            "The file was read but contains no data.",
            "Check the delimiter, or that the first row is a header.",
        )
    if df.width == 1 and file_type in TEXT_TYPES and sep not in {"fixed"}:
        warnings.append("Only one column was detected. The delimiter may be wrong; try another one.")

    df = _clean_columns(df)
    elapsed = time.perf_counter() - started
    logger.info("Loaded {} with {} in {:.2f}s ({} rows)", path.name, choice.engine, elapsed, df.height)

    return LoadedDataset(
        name=display_name or path.name,
        path=path,
        file_type=file_type,
        file_size_bytes=size,
        engine=choice.engine,
        engine_reason=choice.reason,
        df=df,
        total_rows=total_rows,
        sampled=sampled,
        delimiter=sep,
        sheet=sheet,
        mode=options.mode,
        load_seconds=elapsed,
        warnings=warnings,
    )


def describe_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Column names, data types, missing counts and memory per column."""
    nulls = list(df.null_count().row(0))
    height = max(df.height, 1)
    return pl.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing": nulls,
            "missing_%": [round(n / height * 100, 2) for n in nulls],
            "memory_mb": [round(df[c].estimated_size() / 1e6, 3) for c in df.columns],
        }
    )


def analysis_sample(
    df: pl.DataFrame, mode: ProcessingMode, preserve_order: bool = False
) -> pl.DataFrame:
    """Row-limited frame for plots and quick EDA, depending on the mode.

    preserve_order=True uses evenly spaced rows (use it for time series);
    otherwise a seeded random sample is drawn.
    """
    limit = MODE_ROW_LIMITS[mode]
    if limit is None or df.height <= limit:
        return df
    if preserve_order:
        return df.gather_every(math.ceil(df.height / limit))
    return df.sample(n=limit, seed=settings.sample_seed)