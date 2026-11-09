"""paths.yaml loading: a function, explicit about what is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from idd_spatial_analysis.paths import REQUIRED_KEYS, PathsFileMissingError, load_paths


def _write(tmp_path: Path, body: str) -> Path:
    file = tmp_path / "paths.yaml"
    file.write_text(body)
    return file


def test_missing_file_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(PathsFileMissingError, match=r"cp paths.yaml.example paths.yaml"):
        load_paths(tmp_path / "paths.yaml")


def test_roots_become_paths_and_write_defaults_to_empty(tmp_path: Path) -> None:
    body = "\n".join(f"{key}: /some/where/{key}" for key in REQUIRED_KEYS) + "\n"
    paths = load_paths(_write(tmp_path, body))
    for key in REQUIRED_KEYS:
        assert isinstance(paths[key], Path)
        assert paths[key] == Path(f"/some/where/{key}")
    assert paths["write"] == []


def test_write_list_is_kept_as_list(tmp_path: Path) -> None:
    body = "\n".join(f"{key}: /x/{key}" for key in REQUIRED_KEYS) + "\nwrite: [raking_root]\n"
    paths = load_paths(_write(tmp_path, body))
    assert paths["write"] == ["raking_root"]


def test_missing_required_key_raises(tmp_path: Path) -> None:
    body = "admin_shapefile_root: /x/a\n"
    with pytest.raises(KeyError, match="tile_index_root"):
        load_paths(_write(tmp_path, body))


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_paths(_write(tmp_path, ""))
