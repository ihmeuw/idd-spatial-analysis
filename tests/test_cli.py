"""The idd-spatial command line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from click.testing import CliRunner

from idd_spatial_analysis import __version__
from idd_spatial_analysis.boundaries import AdminRelease, load_admin
from idd_spatial_analysis.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _paths_yaml(tmp_path: Path, release: AdminRelease, write: list[str]) -> Path:
    config = {
        "admin_shapefile_root": str(release.root),
        "tile_index_root": str(tmp_path),
        "population_grid_root": str(tmp_path),
        "raking_root": str(tmp_path),
        "project_output_root": str(tmp_path / "node"),
        "write": write,
    }
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_materialize_copies_the_release_once(tmp_path: Path, release: AdminRelease) -> None:
    paths = _paths_yaml(tmp_path, release, ["project_output_root"])
    args = ["materialize-admin-release", "--release", "2024_07_29", "--paths-file", str(paths)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "copied release 2024_07_29" in result.output
    copied = AdminRelease(tmp_path / "node" / "admin_shapefiles", "2024_07_29", simplified=False)
    assert len(load_admin(copied, 2)) == 4
    again = CliRunner().invoke(main, args)
    assert again.exit_code == 1
    assert "refusing" in again.output


def test_materialize_needs_write_permission(tmp_path: Path, release: AdminRelease) -> None:
    paths = _paths_yaml(tmp_path, release, [])
    result = CliRunner().invoke(
        main, ["materialize-admin-release", "--release", "2024_07_29", "--paths-file", str(paths)]
    )
    assert result.exit_code == 1
    assert "write list" in result.output
    assert not (tmp_path / "node").exists()


def test_materialize_reports_missing_release_bad_levels_and_simplified_only(
    tmp_path: Path, release: AdminRelease
) -> None:
    paths = _paths_yaml(tmp_path, release, ["project_output_root"])
    missing = CliRunner().invoke(
        main, ["materialize-admin-release", "--release", "2099_01_01", "--paths-file", str(paths)]
    )
    assert missing.exit_code == 1
    assert "2099_01_01" in missing.output
    bad = CliRunner().invoke(
        main,
        [
            "materialize-admin-release",
            "--release",
            "2024_07_29",
            "--levels",
            "a,b",
            "--paths-file",
            str(paths),
        ],
    )
    assert bad.exit_code == 1
    assert "comma-separated" in bad.output
    only = CliRunner().invoke(
        main,
        [
            "materialize-admin-release",
            "--release",
            "2024_07_29",
            "--levels",
            "0",
            "--simplified-only",
            "--paths-file",
            str(paths),
        ],
    )
    assert only.exit_code == 0, only.output
    written = sorted(
        p.name for p in (tmp_path / "node" / "admin_shapefiles" / "2024_07_29").iterdir()
    )
    assert written
    assert all(name.startswith("lbd_standard_admin_0_simplified") for name in written)


def test_levels_naming_nothing_is_an_error(tmp_path: Path, release: AdminRelease) -> None:
    paths = _paths_yaml(tmp_path, release, ["project_output_root"])
    result = CliRunner().invoke(
        main,
        [
            "materialize-admin-release",
            "--release",
            "2024_07_29",
            "--levels",
            ",",
            "--paths-file",
            str(paths),
        ],
    )
    assert result.exit_code == 1
    assert "no level" in result.output
    assert not (tmp_path / "node").exists()


def test_missing_paths_file_is_a_clean_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "materialize-admin-release",
            "--release",
            "x",
            "--paths-file",
            str(tmp_path / "none.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "paths.yaml" in result.output


def test_help_and_version() -> None:
    assert "materialize-admin-release" in CliRunner().invoke(main, ["--help"]).output
    assert __version__ in CliRunner().invoke(main, ["--version"]).output
