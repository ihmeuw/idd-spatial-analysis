"""Zone polygons to id masks."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import box

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import BlockGrid
from idd_spatial_analysis.raster import rasterize_zones
from idd_spatial_analysis.raster.zones import _near_grid


def test_quadrants_mask_and_windows(grid: BlockGrid, zones_gdf: gpd.GeoDataFrame) -> None:
    mask, windows = rasterize_zones(zones_gdf, "zone", grid)
    assert mask.dtype == np.int32
    assert (mask[:32, :32] == 1).all()
    assert (mask[:32, 32:] == 2).all()
    assert (mask[32:, :32] == 3).all()
    assert (mask[32:, 32:] == 4).all()
    assert set(windows) == {1, 2, 3, 4}
    assert windows[1] == Window(0, 0, 32, 32)
    assert windows[4] == Window(32, 32, 32, 32)


def test_all_touched_includes_slivers(grid: BlockGrid) -> None:
    sliver = gpd.GeoDataFrame(
        {"zone": np.array([7], dtype="int32")}, geometry=[box(100, 100, 140, 140)], crs=grid.crs
    )
    mask, windows = rasterize_zones(sliver, "zone", grid)
    assert mask.sum() == 0
    assert windows == {}
    mask, windows = rasterize_zones(sliver, "zone", grid, all_touched=True)
    assert (mask == 7).sum() == 1
    assert windows == {7: Window(1, 62, 1, 1)}


def test_zones_in_another_crs_give_the_same_mask(
    grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    mask, _ = rasterize_zones(zones_gdf, "zone", grid)
    lonlat, _ = rasterize_zones(zones_gdf.to_crs(WGS84), "zone", grid)
    np.testing.assert_array_equal(lonlat, mask)


def test_near_grid_keeps_the_nearby_rows_in_their_order(
    grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    """Only the four quadrant zones reach the block; the fifth lies 14 km away."""
    lonlat = zones_gdf.to_crs(WGS84).iloc[::-1]
    near = _near_grid(lonlat, grid)
    assert near["zone"].tolist() == [4, 3, 2, 1]
    mask, _ = rasterize_zones(lonlat, "zone", grid)
    expected, _ = rasterize_zones(zones_gdf, "zone", grid)
    np.testing.assert_array_equal(mask, expected)


def test_near_grid_keeps_every_row_when_the_box_cannot_be_drawn() -> None:
    """A grid of the whole world has no lon/lat box narrower than 180 degrees."""
    world = BlockGrid(
        crs=WORLD_CYLINDRICAL_EQUAL_AREA,
        transform=Affine(100_000.0, 0, -20_000_000.0, 0, -100_000.0, 6_300_000.0),
        height=126,
        width=400,
    )
    zones = gpd.GeoDataFrame(
        {"zone": np.array([1, 2], dtype="int32")},
        geometry=[box(-170, -10, -150, 10), box(150, -10, 170, 10)],
        crs=WGS84,
    )
    assert _near_grid(zones, world)["zone"].tolist() == [1, 2]
    mask, windows = rasterize_zones(zones, "zone", world)
    assert set(windows) == {1, 2}
    assert (mask == 1).sum() == (mask == 2).sum() > 0
    assert (mask[:, :200] == 2).sum() == 0
    assert (mask[:, 200:] == 1).sum() == 0


def test_block_without_zones_is_empty(grid: BlockGrid) -> None:
    far = gpd.GeoDataFrame(
        {"zone": np.array([1], dtype="int32")},
        geometry=[box(50_000, 50_000, 51_000, 51_000)],
        crs=grid.crs,
    )
    mask, windows = rasterize_zones(far, "zone", grid)
    assert mask.sum() == 0
    assert windows == {}


def test_overlapping_zones_last_wins(grid: BlockGrid) -> None:
    two = gpd.GeoDataFrame(
        {"zone": np.array([1, 2], dtype="int32")},
        geometry=[box(0, 0, 6400, 6400), box(0, 0, 3200, 3200)],
        crs=grid.crs,
    )
    mask, windows = rasterize_zones(two, "zone", grid)
    assert (mask[32:, :32] == 2).all()
    assert (mask[:32, :] == 1).all()
    assert windows[2] == Window(0, 32, 32, 32)


def test_empty_zones_and_0_360_longitudes_are_refused(
    grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> None:
    with pytest.raises(ValueError, match="empty"):
        rasterize_zones(zones_gdf.iloc[:0], "zone", grid)
    wrapped = gpd.GeoDataFrame(
        {"zone": np.array([1], dtype="int32")}, geometry=[box(170, -20, 190, -10)], crs=WGS84
    )
    with pytest.raises(ValueError, match="to_lon_180"):
        rasterize_zones(wrapped, "zone", grid)


def test_zones_already_in_the_grid_crs_are_burned_as_they_are() -> None:
    lon360 = BlockGrid(
        crs=WGS84, transform=Affine(0.5, 0, 160.0, 0, -0.5, 0.0), height=60, width=80
    )
    zones = gpd.GeoDataFrame(
        {"zone": np.array([1], dtype="int32")}, geometry=[box(170, -20, 190, -10)], crs=WGS84
    )
    mask, windows = rasterize_zones(zones, "zone", lon360)
    assert (mask == 1).sum() == 800
    assert windows == {1: Window(20, 20, 40, 20)}


def test_bad_ids_and_inputs(grid: BlockGrid, zones_gdf: gpd.GeoDataFrame) -> None:
    with pytest.raises(KeyError, match="nope"):
        rasterize_zones(zones_gdf, "nope", grid)
    with pytest.raises(ValueError, match="CRS"):
        rasterize_zones(zones_gdf.set_crs(None, allow_override=True), "zone", grid)
    with pytest.raises(TypeError, match="integers"):
        rasterize_zones(zones_gdf.assign(zone=zones_gdf["zone"].astype(float)), "zone", grid)
    with pytest.raises(ValueError, match="duplicate"):
        rasterize_zones(
            zones_gdf.assign(zone=np.array([1, 1, 3, 4, 9], dtype="int32")), "zone", grid
        )
    with pytest.raises(ValueError, match="nodata"):
        rasterize_zones(
            zones_gdf.assign(zone=np.array([0, 2, 3, 4, 9], dtype="int32")), "zone", grid
        )
