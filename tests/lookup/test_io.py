"""Atomic table and sidecar writes of the lookup products."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from idd_spatial_analysis.lookup import _io
from idd_spatial_analysis.lookup._io import (
    check_file_stem,
    preview,
    read_parquet_checked,
    write_parquet_atomic,
    write_yaml_atomic,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = pa.schema([("zone", pa.int32()), ("pixel_count", pa.int64())])


def _table() -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()), pa.array([10, 20, 30], type=pa.int64())],
        schema=SCHEMA,
    )


def test_parquet_round_trip_keeps_dtypes_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "table.parquet"
    assert write_parquet_atomic(_table(), target) == target
    assert [path.name for path in target.parent.iterdir()] == ["table.parquet"]
    back = read_parquet_checked(target, SCHEMA)
    assert back.equals(_table())
    assert back.schema.equals(SCHEMA)


def test_parquet_write_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "table.parquet"
    write_parquet_atomic(_table(), target)
    shorter = _table().slice(0, 1)
    write_parquet_atomic(shorter, target)
    assert pq.read_metadata(target).num_rows == 1


def test_footer_mismatch_leaves_the_target_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "table.parquet"
    write_parquet_atomic(_table(), target)
    real = pq.read_metadata

    class _Wrong:
        num_rows = 99
        schema = real(target).schema

    monkeypatch.setattr(_io.pq, "read_metadata", lambda _path: _Wrong())
    with pytest.raises(ValueError, match="99 rows != 1"):
        write_parquet_atomic(_table().slice(0, 1), target)
    monkeypatch.undo()
    assert pq.read_metadata(target).num_rows == 3
    assert [path.name for path in tmp_path.iterdir()] == ["table.parquet"]


def test_read_checked_refuses_other_types_and_columns(tmp_path: Path) -> None:
    target = tmp_path / "table.parquet"
    write_parquet_atomic(_table(), target)
    wider = pa.schema([("zone", pa.int64()), ("pixel_count", pa.int64())])
    with pytest.raises(ValueError, match="is not the expected"):
        read_parquet_checked(target, wider)
    renamed = pa.schema([("zone", pa.int32()), ("pixels", pa.int64())])
    with pytest.raises(ValueError, match="is not the expected"):
        read_parquet_checked(target, renamed)


def test_yaml_sidecar_keeps_key_order(tmp_path: Path) -> None:
    data = {"product": "zone_lookup", "zone_set": "x", "all_touched": False, "resolution": [1.0]}
    target = write_yaml_atomic(data, tmp_path / "side" / "build.yaml")
    assert list(yaml.safe_load(target.read_text())) == list(data)
    assert yaml.safe_load(target.read_text()) == data
    assert [path.name for path in target.parent.iterdir()] == ["build.yaml"]


def test_file_stems_must_be_plain_names() -> None:
    for good in ("B-0000X-0000Y", "2020", "2020q1", "static.v2"):
        check_file_stem(good, "label")
    for bad in ("", ".", "..", "a/b", "/abs", "dir/"):
        with pytest.raises(ValueError, match="label must be a plain file stem"):
            check_file_stem(bad, "label")


def test_preview_sorts_and_counts_what_it_leaves_out() -> None:
    assert preview({"b", "a"}) == "['a', 'b']"
    many = preview(range(25, 0, -1))
    assert many == "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (+15 more)"
