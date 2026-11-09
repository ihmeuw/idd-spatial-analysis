"""Small checks on pandas tables shared by the zonal functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


def require_columns(table: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise ``KeyError`` naming ``name`` when any of ``columns`` is missing from ``table``."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        msg = f"{name} lacks columns {missing}"
        raise KeyError(msg)
