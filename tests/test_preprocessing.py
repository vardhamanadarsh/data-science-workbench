from datetime import date

import numpy as np
import polars as pl
import pytest

from core.feature_engineering import (
    add_date_parts, add_interaction, add_lags, add_rolling, leakage_notes,
)
from core.history import DataHistory
from core.preprocessing import (
    PreprocessError, convert_type, drop_duplicate_rows, fill_missing, handle_outliers,
    outlier_table, transform_numeric,
)


def _df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, None, 4, 5, 5], "c": ["x", "y", "x", None, "y", "y"]})


def test_drop_duplicates() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    res = drop_duplicate_rows(df)
    assert res.df.height == 2
    assert not drop_duplicate_rows(res.df).changed


def test_fill_median_does_not_touch_original() -> None:
    df = _df()
    res = fill_missing(df, ["a"], "median")
    assert res.df["a"].null_count() == 0 and res.df["a"][2] == 4.0
    assert df["a"].null_count() == 1


def test_fill_mean_on_text_raises() -> None:
    with pytest.raises(PreprocessError):
        fill_missing(_df(), ["c"], "mean")


def test_fill_mode_text() -> None:
    assert fill_missing(_df(), ["c"], "mode").df["c"][3] == "y"


def test_convert_reports_failed_values() -> None:
    res = convert_type(pl.DataFrame({"v": ["1", "2", "oops"]}), "v", "Number (decimal)")
    assert res.df["v"].dtype == pl.Float64
    assert res.df["v"].null_count() == 1 and "could not be converted" in res.message


def test_outliers_clip_and_remove() -> None:
    df = pl.DataFrame({"x": [float(i) for i in range(1, 21)] + [1000.0]})
    assert outlier_table(df, ["x"])["outliers"][0] == 1
    assert handle_outliers(df, ["x"], 1.5, "clip").df["x"].max() == pytest.approx(31.0)
    assert handle_outliers(df, ["x"], 1.5, "remove").df.height == 20


def test_log1p_and_domain_check() -> None:
    res = transform_numeric(pl.DataFrame({"v": [0.0, 1.0, 3.0]}), ["v"], "log1p")
    assert res.new_columns == ["v_log1p"]
    assert res.df["v_log1p"][2] == pytest.approx(np.log(4))
    with pytest.raises(PreprocessError):
        transform_numeric(pl.DataFrame({"v": [-5.0, 1.0]}), ["v"], "log1p")


def test_history_undo_redo() -> None:
    h = DataHistory(pl.DataFrame({"a": [1, 2, 3]}))
    h.push("step 1", pl.DataFrame({"a": [1, 2]}))
    assert h.index == 1 and h.current.df.height == 2
    h.undo()
    assert h.current.df.height == 3 and h.can_redo
    h.push("step 2", pl.DataFrame({"a": [9]}))
    assert not h.can_redo


def test_history_limit_keeps_original() -> None:
    h = DataHistory(pl.DataFrame({"a": [1]}), max_snapshots=3)
    for i in range(5):
        h.push(f"s{i}", pl.DataFrame({"a": [i]}))
    assert len(h.snapshots) == 3 and h.snapshots[0].label == "Original data"
    assert h.current.label == "s4"


def test_lags_and_rolling() -> None:
    df = pl.DataFrame({"t": [1, 2, 3, 4, 5, 6], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert add_lags(df, ["v"], [1], "t").df["v_lag1"].to_list() == [None, 1.0, 2.0, 3.0, 4.0, 5.0]
    safe = add_rolling(df, ["v"], [3], ["mean"], True, "t").df["v_roll3_mean"].to_list()
    assert safe == [None, None, None, 2.0, 3.0, 4.0]
    incl = add_rolling(df, ["v"], [3], ["mean"], False, "t").df["v_roll3_mean"].to_list()
    assert incl == [None, None, 2.0, 3.0, 4.0, 5.0]


def test_unsorted_rows_rejected() -> None:
    df = pl.DataFrame({"t": [2, 1, 3], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(PreprocessError):
        add_lags(df, ["v"], [1], "t")


def test_date_parts() -> None:
    df = pl.DataFrame({"d": [date(2024, 1, 15), date(2024, 7, 4)]})
    res = add_date_parts(df, "d", ["year", "month"])
    assert res.df["d_year"].to_list() == [2024, 2024] and res.df["d_month"].to_list() == [1, 7]
    with pytest.raises(PreprocessError):
        add_date_parts(df, "d", ["hour"])


def test_ratio_by_zero_is_missing() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0]})
    assert add_interaction(df, "a", "b", "ratio").df["a_per_b"].to_list() == [None, 0.5]


def test_leakage_notes() -> None:
    assert any("time-based" in n for n in leakage_notes(["y"], "y", "lag"))
    assert leakage_notes(["x"], "y", "interaction") == []