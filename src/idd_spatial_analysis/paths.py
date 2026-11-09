"""Absolute roots from the gitignored ``paths.yaml`` (STANDARDS §Path safety).

A function, not top-level code, so importing the package never requires ``paths.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PATHS_FILE = Path(__file__).resolve().parents[2] / "paths.yaml"
_WRITE_KEY = "write"

REQUIRED_KEYS: tuple[str, ...] = (
    "admin_shapefile_root",
    "tile_index_root",
    "population_grid_root",
    "raking_root",
    "project_output_root",
)


class PathsFileMissingError(FileNotFoundError):
    """``paths.yaml`` is absent; the message says what to do."""


def load_paths(paths_file: Path | None = None) -> dict[str, Any]:
    """Read ``paths.yaml`` and return its mapping with every root as a ``Path``.

    ``write`` comes back as a list of key names (empty when absent or blank): the roots a
    session, and the CLI, may write under. Anything else under ``write`` is an error.
    """
    file = paths_file if paths_file is not None else _PATHS_FILE
    if not file.exists():
        msg = f"{file} missing. Run: cp paths.yaml.example paths.yaml; then edit."
        raise PathsFileMissingError(msg)
    with file.open() as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        msg = f"{file} lacks required keys: {missing}"
        raise KeyError(msg)
    resolved: dict[str, Any] = {
        key: (Path(value) if key != _WRITE_KEY and isinstance(value, str) else value)
        for key, value in data.items()
    }
    write = resolved.get(_WRITE_KEY)
    if write is None:
        write = []
    if not isinstance(write, list) or not all(isinstance(key, str) for key in write):
        msg = f"{file}: write must be a list of key names, got {write!r}"
        raise TypeError(msg)
    resolved[_WRITE_KEY] = write
    return resolved
