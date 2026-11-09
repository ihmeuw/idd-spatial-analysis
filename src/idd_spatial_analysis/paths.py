"""Absolute roots from the gitignored ``paths.yaml`` (STANDARDS §Path safety).

A function, not top-level code, so importing the package never requires ``paths.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PATHS_FILE = Path(__file__).resolve().parents[2] / "paths.yaml"

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

    ``write`` is returned as-is (a list of keys a Claude session may write under); pipeline
    code ignores it.
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
        key: (Path(value) if key != "write" and isinstance(value, str) else value)
        for key, value in data.items()
    }
    resolved.setdefault("write", [])
    return resolved
