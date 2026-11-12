"""Zone lookup rasters: build per block, read back, collect."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import rasterio
import yaml
from rasterio.windows import Window
from shapely.geometry import box

from idd_spatial_analysis import __version__
from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.lookup import (
    build_zone_lookup_block,
    collect_zone_lookup,
    load_zone_lookup_index,
    load_zone_lookup_metadata,
    load_zone_lookup_summary,
    read_zone_lookup_block,
    zone_lookup_block_path,
)
from idd_spatial_analysis.raster import rasterize_zones, write_geotiff_atomic
from idd_spatial_analysis.zonal import collate_blocks, weighted_sums

from .conftest import EAST, WEST, WIDE_ZONE_IDS

if TYPE_CHECKING:
    from pathlib import Path

    from idd_spatial_analysis.frame import BlockGrid


@pytest.fixture
def built(tmp_path: Path, blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame) -> Path:
    """A lookup directory with both blocks built and nothing collected yet."""
    for key, grid in blocks.items():
        build_zone_lookup_block(wide_zones_gdf, "zone", grid, tmp_path, key)
    return tmp_path


def test_block_raster_and_table(
    tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    table = build_zone_lookup_block(zones_gdf, "zone", grid, tmp_path, "B-0000X-0000Y")
    expected, _ = rasterize_zones(zones_gdf, "zone", grid)
    path = zone_lookup_block_path(tmp_path, "B-0000X-0000Y")
    assert path == tmp_path / "blocks" / "B-0000X-0000Y.tif"
    with rasterio.open(path) as src:
        assert src.dtypes == ("int32",)
        assert src.nodata == 0
        assert src.compression is not None
        np.testing.assert_array_equal(src.read(1), expected)
    assert list(table.columns) == ["zone", "pixel_count", "row_off", "col_off", "height", "width"]
    assert table["zone"].tolist() == [1, 2, 3, 4]
    assert table["pixel_count"].tolist() == [1024, 1024, 1024, 1024]
    assert table.loc[table["zone"] == 4, ["row_off", "col_off", "height", "width"]].to_numpy()[
        0
    ].tolist() == [32, 32, 32, 32]
    assert table["zone"].dtype == np.int32
    assert table["pixel_count"].dtype == np.int64
    assert table["row_off"].dtype == np.int32
    on_disk = pq.read_table(tmp_path / "block_zones" / "B-0000X-0000Y.parquet")
    assert on_disk.to_pandas().equals(table)
    assert not list(tmp_path.rglob("*.tmp"))


def test_read_block_whole_and_by_window(
    tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    build_zone_lookup_block(zones_gdf, "zone", grid, tmp_path, "b")
    expected, _ = rasterize_zones(zones_gdf, "zone", grid)
    mask, read_grid = read_zone_lookup_block(tmp_path, "b")
    assert mask.dtype == np.int32
    assert read_grid == grid
    np.testing.assert_array_equal(mask, expected)
    window = Window(32, 32, 32, 32)
    part, part_grid = read_zone_lookup_block(tmp_path, "b", window=window)
    assert (part == 4).all()
    assert part.shape == (32, 32)
    assert part_grid == grid.sub(window)


def test_read_refuses_missing_foreign_and_out_of_range(
    tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    with pytest.raises(FileNotFoundError, match="no raster for block 'absent'"):
        read_zone_lookup_block(tmp_path, "absent")
    build_zone_lookup_block(zones_gdf, "zone", grid, tmp_path, "b")
    with pytest.raises(ValueError, match="does not lie inside"):
        read_zone_lookup_block(tmp_path, "b", window=Window(60, 0, 8, 8))
    with pytest.raises(ValueError, match="does not lie inside"):
        read_zone_lookup_block(tmp_path, "b", window=Window(0, 0, 0, 4))
    write_geotiff_atomic(
        zone_lookup_block_path(tmp_path, "floats"),
        np.ones(grid.shape, "float32"),
        grid,
        nodata=None,
    )
    with pytest.raises(ValueError, match="not a zone lookup raster"):
        read_zone_lookup_block(tmp_path, "floats")
    for bad in ("", "..", "a/b"):
        with pytest.raises(ValueError, match="plain file stem"):
            zone_lookup_block_path(tmp_path, bad)


def test_block_without_zones_is_written_empty(tmp_path: Path, grid: BlockGrid) -> None:
    far = gpd.GeoDataFrame(
        {"zone": np.array([7], dtype="int32")},
        geometry=[box(50_000, 50_000, 51_000, 51_000)],
        crs=grid.crs,
    )
    table = build_zone_lookup_block(far, "zone", grid, tmp_path, "empty")
    assert table.empty
    assert table["zone"].dtype == np.int32
    mask, _ = read_zone_lookup_block(tmp_path, "empty")
    assert mask.shape == grid.shape
    assert not mask.any()


def test_rebuilding_a_block_replaces_it(
    tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    build_zone_lookup_block(zones_gdf, "zone", grid, tmp_path, "b")
    fewer = zones_gdf.loc[zones_gdf["zone"] != 4]
    table = build_zone_lookup_block(fewer, "zone", grid, tmp_path, "b")
    assert table["zone"].tolist() == [1, 2, 3]
    mask, _ = read_zone_lookup_block(tmp_path, "b")
    assert not (mask == 4).any()
    assert (mask[32:, 32:] == 0).all()


def test_zones_in_lonlat_build_the_same_block(
    tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    build_zone_lookup_block(zones_gdf.to_crs(WGS84), "zone", grid, tmp_path, "b")
    expected, _ = rasterize_zones(zones_gdf, "zone", grid)
    mask, _ = read_zone_lookup_block(tmp_path, "b")
    np.testing.assert_array_equal(mask, expected)


def test_collect_writes_index_summary_and_sidecar(built: Path) -> None:
    summary = collect_zone_lookup(
        built, [EAST, WEST], WIDE_ZONE_IDS, zone_set="synthetic", zone_source="unit test"
    )
    assert summary["zone"].tolist() == WIDE_ZONE_IDS
    assert summary["pixel_count"].tolist() == [3072, 1024, 4096, 0, 0]
    assert summary["block_count"].tolist() == [2, 1, 2, 0, 0]
    assert summary["zone"].dtype == np.int32
    assert summary["pixel_count"].dtype == np.int64
    assert summary["block_count"].dtype == np.int32
    assert load_zone_lookup_summary(built).equals(summary)

    index = load_zone_lookup_index(built)
    assert list(index.columns) == [
        "block_key",
        "zone",
        "pixel_count",
        "row_off",
        "col_off",
        "height",
        "width",
    ]
    assert index[["block_key", "zone"]].to_numpy().tolist() == [
        [WEST, 1],
        [WEST, 3],
        [EAST, 1],
        [EAST, 2],
        [EAST, 3],
    ]
    assert index["pixel_count"].tolist() == [2048, 2048, 1024, 1024, 2048]
    east_one = index.loc[(index["block_key"] == EAST) & (index["zone"] == 1)].iloc[0]
    window = Window(east_one["col_off"], east_one["row_off"], east_one["width"], east_one["height"])
    assert window == Window(0, 0, 32, 32)
    part, _ = read_zone_lookup_block(built, EAST, window=window)
    assert (part == 1).all()

    metadata = load_zone_lookup_metadata(built)
    assert metadata["product"] == "zone_lookup"
    assert metadata["zone_set"] == "synthetic"
    assert metadata["zone_source"] == "unit test"
    assert metadata["zone_col"] == "zone"
    assert metadata["crs"] == "ESRI:54034"
    assert metadata["resolution"] == [100.0, 100.0]
    assert metadata["all_touched"] is False
    assert metadata["block_count"] == 2
    assert metadata["zone_count"] == 5
    assert metadata["zones_without_pixels"] == 2
    assert metadata["zoned_pixel_count"] == 2 * 64 * 64
    assert metadata["package_version"] == __version__
    assert not list(built.rglob("*.tmp"))


def test_collect_refuses_missing_and_foreign_blocks(built: Path) -> None:
    with pytest.raises(FileNotFoundError, match="1 blocks are not built"):
        collect_zone_lookup(built, [WEST, EAST, "B-0002X-0000Y"], WIDE_ZONE_IDS, zone_set="s")
    with pytest.raises(ValueError, match="not in block_keys"):
        collect_zone_lookup(built, [WEST], WIDE_ZONE_IDS, zone_set="s")
    (built / "block_zones" / f"{EAST}.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=EAST):
        collect_zone_lookup(built, [WEST, EAST], WIDE_ZONE_IDS, zone_set="s")
    assert not (built / "zones.parquet").exists()


def test_collect_refuses_mixed_parameters_and_unknown_zones(
    built: Path, blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame
) -> None:
    with pytest.raises(ValueError, match="not in zone_ids"):
        collect_zone_lookup(built, [WEST, EAST], [1, 2, 5, 9], zone_set="s")
    build_zone_lookup_block(wide_zones_gdf, "zone", blocks[EAST], built, EAST, all_touched=True)
    with pytest.raises(ValueError, match="was built with"):
        collect_zone_lookup(built, [WEST, EAST], WIDE_ZONE_IDS, zone_set="s")


def test_collect_refuses_bad_keys_and_ids(built: Path) -> None:
    with pytest.raises(ValueError, match="block_keys is empty"):
        collect_zone_lookup(built, [], WIDE_ZONE_IDS, zone_set="s")
    with pytest.raises(ValueError, match="block_keys holds duplicates"):
        collect_zone_lookup(built, [WEST, WEST, EAST], WIDE_ZONE_IDS, zone_set="s")
    with pytest.raises(ValueError, match="zone_ids is empty"):
        collect_zone_lookup(built, [WEST, EAST], [], zone_set="s")
    with pytest.raises(ValueError, match="zone_ids holds duplicates"):
        collect_zone_lookup(built, [WEST, EAST], [1, 1, 2, 3], zone_set="s")
    with pytest.raises(TypeError, match="must be integers"):
        collect_zone_lookup(built, [WEST, EAST], [1.0, 2.0, 3.0], zone_set="s")
    with pytest.raises(ValueError, match="0 is nodata"):
        collect_zone_lookup(built, [WEST, EAST], [0, 1, 2, 3], zone_set="s")


def test_block_table_without_parameters_is_refused(built: Path) -> None:
    path = built / "block_zones" / f"{WEST}.parquet"
    pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    with pytest.raises(ValueError, match="no zone lookup build parameters"):
        collect_zone_lookup(built, [WEST, EAST], WIDE_ZONE_IDS, zone_set="s")


def test_loaders_need_a_collected_lookup_of_this_product(built: Path) -> None:
    for loader in (load_zone_lookup_index, load_zone_lookup_summary, load_zone_lookup_metadata):
        with pytest.raises(FileNotFoundError, match="collect_zone_lookup"):
            loader(built)
    collect_zone_lookup(built, [WEST, EAST], WIDE_ZONE_IDS, zone_set="s")
    sidecar = built / "build.yaml"
    assert load_zone_lookup_metadata(built)["zone_source"] is None
    sidecar.write_text(yaml.safe_dump({"product": "something_else", "schema_version": 1}))
    with pytest.raises(ValueError, match="something_else"):
        load_zone_lookup_metadata(built)


def test_sums_from_the_lookup_match_rasterizing_again(
    built: Path, blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame
) -> None:
    """Population-weighted means per zone from the stored masks, block by block, added up."""
    rng = np.random.default_rng(7)
    from_lookup, direct = [], []
    for key, grid in blocks.items():
        population = rng.gamma(2.0, 50.0, size=grid.shape).astype("float32")
        value = rng.normal(size=grid.shape).astype("float32")
        stored, stored_grid = read_zone_lookup_block(built, key)
        assert stored_grid == grid
        fresh, _ = rasterize_zones(wide_zones_gdf, "zone", grid)
        from_lookup.append(weighted_sums(stored, population, value, nan_policy="keep_weight"))
        direct.append(weighted_sums(fresh, population, value, nan_policy="keep_weight"))
    got, expected = collate_blocks(from_lookup), collate_blocks(direct)
    assert got["zone"].tolist() == [1, 2, 3]
    assert got.equals(expected)
    numerator, denominator = got["weighted_sum"], got["weight_sum"]
    assert np.isfinite(numerator / denominator).all()
