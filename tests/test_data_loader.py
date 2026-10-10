from pathlib import Path

import polars as pl

from core.data_loader import LoadOptions, describe_schema, load_dataset, sniff_delimiter


def test_csv_semicolon_autodetect(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_text("x;y\n1;a\n2;b\n3;c\n", encoding="utf-8")
    assert sniff_delimiter(f, "csv") == ";"
    ds = load_dataset(f)
    assert ds.df.shape == (3, 2)
    assert ds.engine == "polars"


def test_parquet_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "a.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}).write_parquet(f)
    ds = load_dataset(f, LoadOptions())
    assert ds.n_rows_loaded == 3
    assert describe_schema(ds.df).height == 2


def test_wrong_extension_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "bad.exe"
    f.write_bytes(b"MZ")
    try:
        load_dataset(f)
    except Exception as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected an error")