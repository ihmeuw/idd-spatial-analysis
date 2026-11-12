"""Atomic writes and checked reads for the tables and sidecars of the lookup products."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import yaml

from idd_spatial_analysis.raster.io import atomic_write

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa

_SHOW = 10


def check_file_stem(value: str, name: str) -> None:
    """Raise unless ``value`` can be a file name without its suffix: no separators, not empty."""
    if not value or Path(value).name != value or value in {".", ".."}:
        msg = f"{name} must be a plain file stem, got {value!r}"
        raise ValueError(msg)


def preview(values: Iterable[Any]) -> str:
    """The first few of ``values``, sorted, and how many more there are; for error messages."""
    items = sorted(values)
    more = "" if len(items) <= _SHOW else f" (+{len(items) - _SHOW} more)"
    return f"{items[:_SHOW]}{more}"


def write_parquet_atomic(table: pa.Table, path: Path | str) -> Path:
    """Write ``table`` to ``path`` through ``atomic_write`` and check the file's footer.

    The file carries the table's schema as it stands, so the caller's explicit dtypes are what
    lands on disk. Before the rename the temp file's footer is compared with the table (row
    count, column names and types); the data is never read back. Same contract as idd-tools'
    ``atomic_write_parquet``, local for the reason ``raster.io.atomic_write`` is.
    """
    target = Path(path)

    def _write(tmp: Path) -> None:
        pq.write_table(table, tmp)
        footer = pq.read_metadata(tmp)
        problems: list[str] = []
        if footer.num_rows != table.num_rows:
            problems.append(f"{footer.num_rows} rows != {table.num_rows}")
        written = footer.schema.to_arrow_schema()
        if not written.equals(table.schema, check_metadata=False):
            problems.append(f"schema {written} != {table.schema}")
        if problems:
            msg = f"{target}: footer mismatch after write: {'; '.join(problems)}"
            raise ValueError(msg)

    atomic_write(target, _write)
    return target


def read_parquet_checked(path: Path | str, schema: pa.Schema) -> pa.Table:
    """The table at ``path``, once its columns and types are known to be those of ``schema``.

    Types are asserted, never assumed: a file whose columns or types differ raises
    ``ValueError`` naming both schemas. Schema metadata is not part of the comparison.
    """
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        msg = f"{path}: schema {table.schema} is not the expected {schema}"
        raise ValueError(msg)
    return table


def write_yaml_atomic(data: dict[str, Any], path: Path | str) -> Path:
    """Write ``data`` as YAML to ``path`` through ``atomic_write``, keys in the order given."""
    target = Path(path)
    text = yaml.safe_dump(data, sort_keys=False)
    atomic_write(target, lambda tmp: tmp.write_text(text))
    return target
