"""The idd-spatial command line."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from idd_spatial_analysis import __version__
from idd_spatial_analysis.boundaries import AdminRelease, load_admin
from idd_spatial_analysis.cli import main
from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.lookup import (
    load_zone_lookup_index,
    load_zone_lookup_metadata,
    load_zone_lookup_summary,
    read_zone_lookup_block,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

WEST, EAST = "B-0000X-0000Y", "B-0001X-0000Y"
LOOKUP_NODE = "node/02-processed_data/zone_lookup/lsae_1/100m"


@pytest.fixture(autouse=True)
def _restore_umask() -> Iterator[None]:
    """The build commands set the process umask; put it back so other tests are unaffected."""
    before = os.umask(0)
    os.umask(before)
    yield
    os.umask(before)


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
    listing = CliRunner().invoke(main, ["--help"]).output
    for command in ("materialize-admin-release", "build-zone-lookup", "collect-zone-lookup"):
        assert command in listing
    assert __version__ in CliRunner().invoke(main, ["--version"]).output


def _lookup_paths(tmp_path: Path, tile_index_gdf: gpd.GeoDataFrame, write: list[str]) -> Path:
    """A tile index and an lsae-style raking set on disk, and the paths.yaml that names them.

    Three zones over the two synthetic blocks: 11 the top half of the west block and the first
    32 columns of the east one, 12 the rest of the east block's top half, 13 the bottom half.
    """
    tile_dir = tmp_path / "tiles" / "100m"
    tile_dir.mkdir(parents=True)
    tile_index_gdf.to_parquet(tile_dir / "tile_index.parquet")
    shapes_dir = tmp_path / "raking" / "gbd-inputs"
    shapes_dir.mkdir(parents=True)
    shapes = gpd.GeoDataFrame(
        {"location_id": np.array([11, 12, 13], dtype="int64")},
        geometry=[box(0, 3200, 9600, 6400), box(9600, 3200, 12800, 6400), box(0, 0, 12800, 3200)],
        crs=WORLD_CYLINDRICAL_EQUAL_AREA,
    ).to_crs(WGS84)
    shapes.to_parquet(shapes_dir / "shapes_lsae_1_a2.parquet")
    config = {
        "admin_shapefile_root": str(tmp_path),
        "tile_index_root": str(tmp_path / "tiles"),
        "population_grid_root": str(tmp_path),
        "raking_root": str(tmp_path / "raking"),
        "project_output_root": str(tmp_path / "node"),
        "write": write,
    }
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_build_all_blocks_then_collect(tmp_path: Path, tile_index_gdf: gpd.GeoDataFrame) -> None:
    paths = _lookup_paths(tmp_path, tile_index_gdf, ["project_output_root"])
    common = ["--hierarchy", "lsae_1", "--paths-file", str(paths)]
    built = CliRunner().invoke(main, ["build-zone-lookup", "--all-blocks", *common])
    assert built.exit_code == 0, built.output
    assert f"{WEST}: 2 zones, 4096 zoned pixels of 4096" in built.output
    assert "built 2 block(s)" in built.output
    working = tmp_path / LOOKUP_NODE / "working"
    assert sorted(p.name for p in (working / "blocks").iterdir()) == [f"{WEST}.tif", f"{EAST}.tif"]
    mask, _ = read_zone_lookup_block(working, EAST)
    assert (mask[:32, :32] == 11).all()
    assert (mask[:32, 32:] == 12).all()
    assert (mask[32:, :] == 13).all()

    collected = CliRunner().invoke(main, ["collect-zone-lookup", *common])
    assert collected.exit_code == 0, collected.output
    assert "collected 2 block(s) and 3 zones (0 without a pixel)" in collected.output
    assert load_zone_lookup_summary(working)["pixel_count"].tolist() == [3072, 1024, 4096]
    assert len(load_zone_lookup_index(working)) == 5
    metadata = load_zone_lookup_metadata(working)
    assert metadata["zone_set"] == "lsae_1"
    assert metadata["zone_col"] == "location_id"
    assert metadata["zone_source"] == "population-model raking shapes of lsae_1"


def test_named_block_into_scratch_leaves_working_alone(
    tmp_path: Path, tile_index_gdf: gpd.GeoDataFrame
) -> None:
    paths = _lookup_paths(tmp_path, tile_index_gdf, ["project_output_root"])
    common = ["--hierarchy", "lsae_1", "--scratch", "probe", "--paths-file", str(paths)]
    built = CliRunner().invoke(main, ["build-zone-lookup", "--block-key", EAST, *common])
    assert built.exit_code == 0, built.output
    node = tmp_path / LOOKUP_NODE
    assert (node / "scratch" / "probe" / "blocks" / f"{EAST}.tif").exists()
    assert not (node / "working").exists()
    collected = CliRunner().invoke(main, ["collect-zone-lookup", "--block-key", EAST, *common])
    assert collected.exit_code == 0, collected.output
    assert "collected 1 block(s) and 3 zones (0 without a pixel)" in collected.output
    assert load_zone_lookup_summary(node / "scratch" / "probe")["block_count"].tolist() == [1, 1, 1]


def test_zone_lookup_commands_refuse_bad_requests(
    tmp_path: Path, tile_index_gdf: gpd.GeoDataFrame
) -> None:
    paths = _lookup_paths(tmp_path, tile_index_gdf, ["project_output_root"])
    common = ["--hierarchy", "lsae_1", "--paths-file", str(paths)]
    neither = CliRunner().invoke(main, ["build-zone-lookup", *common])
    assert neither.exit_code == 1
    assert "one or the other" in neither.output
    both = CliRunner().invoke(
        main, ["build-zone-lookup", "--all-blocks", "--block-key", WEST, *common]
    )
    assert both.exit_code == 1
    assert "one or the other" in both.output
    unknown_block = CliRunner().invoke(main, ["build-zone-lookup", "--block-key", "B-9", *common])
    assert unknown_block.exit_code == 1
    assert "B-9" in unknown_block.output
    unknown_hierarchy = CliRunner().invoke(
        main,
        ["build-zone-lookup", "--all-blocks", "--hierarchy", "nope", "--paths-file", str(paths)],
    )
    assert unknown_hierarchy.exit_code == 1
    assert "unknown hierarchy" in unknown_hierarchy.output
    assert not (tmp_path / "node").exists()
    too_early = CliRunner().invoke(main, ["collect-zone-lookup", *common])
    assert too_early.exit_code == 1
    assert "not built" in too_early.output


def test_zone_lookup_commands_need_write_permission(
    tmp_path: Path, tile_index_gdf: gpd.GeoDataFrame
) -> None:
    paths = _lookup_paths(tmp_path, tile_index_gdf, [])
    for command in (["build-zone-lookup", "--all-blocks"], ["collect-zone-lookup"]):
        result = CliRunner().invoke(
            main, [*command, "--hierarchy", "lsae_1", "--paths-file", str(paths)]
        )
        assert result.exit_code == 1
        assert "write list" in result.output
    assert not (tmp_path / "node").exists()
