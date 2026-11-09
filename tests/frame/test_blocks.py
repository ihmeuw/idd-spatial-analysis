"""Tile indexes and block selection."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import (
    block_bounds,
    block_for_point,
    block_grid,
    blocks_intersecting,
    load_tile_index,
)


def test_load_tile_index_checks_crs_and_columns(tmp_path, tile_index_gdf: gpd.GeoDataFrame) -> None:
    path = tmp_path / "tile_index.parquet"
    tile_index_gdf.to_parquet(path)
    tiles = load_tile_index(path)
    assert len(tiles) == len(tile_index_gdf)
    with pytest.raises(ValueError, match="CRS"):
        load_tile_index(path, expected_crs=WGS84)
    tile_index_gdf.drop(columns=["block_key"]).to_parquet(path)
    with pytest.raises(KeyError, match="block_key"):
        load_tile_index(path)


def test_block_bounds_and_grid(tile_index_gdf: gpd.GeoDataFrame) -> None:
    assert block_bounds(tile_index_gdf, "B-0000X-0000Y") == (0.0, 0.0, 6400.0, 6400.0)
    assert block_bounds(tile_index_gdf, "B-0001X-0000Y") == (6400.0, 0.0, 12800.0, 6400.0)
    grid = block_grid(tile_index_gdf, "B-0001X-0000Y", 100.0)
    assert grid.shape == (64, 64)
    assert grid.bounds == (6400.0, 0.0, 12800.0, 6400.0)
    with pytest.raises(KeyError, match="B-9999X"):
        block_bounds(tile_index_gdf, "B-9999X-9999Y")


def test_blocks_intersecting_reprojects_and_sorts(tile_index_gdf: gpd.GeoDataFrame) -> None:
    east = gpd.GeoDataFrame(geometry=[box(7000, 1000, 8000, 2000)], crs=tile_index_gdf.crs)
    assert blocks_intersecting(tile_index_gdf, east) == ["B-0001X-0000Y"]
    both = gpd.GeoDataFrame(geometry=[box(6000, 1000, 8000, 2000)], crs=tile_index_gdf.crs)
    assert blocks_intersecting(tile_index_gdf, both) == ["B-0000X-0000Y", "B-0001X-0000Y"]
    lonlat = east.to_crs(WGS84)
    assert blocks_intersecting(tile_index_gdf, lonlat) == ["B-0001X-0000Y"]


def test_blocks_intersecting_refuses_to_skip_everything(tile_index_gdf: gpd.GeoDataFrame) -> None:
    far = gpd.GeoDataFrame(geometry=[box(50000, 50000, 60000, 60000)], crs=tile_index_gdf.crs)
    with pytest.raises(ValueError, match="no block"):
        blocks_intersecting(tile_index_gdf, far)
    no_crs = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="CRS"):
        blocks_intersecting(tile_index_gdf, no_crs)


def test_blocks_intersecting_refuses_0_360_polygons(tile_index_gdf: gpd.GeoDataFrame) -> None:
    wrapped = gpd.GeoDataFrame(geometry=[box(170, 0, 190, 0.05)], crs=WGS84)
    with pytest.raises(ValueError, match="to_lon_180"):
        blocks_intersecting(tile_index_gdf, wrapped)


def test_block_for_point(tile_index_gdf: gpd.GeoDataFrame) -> None:
    assert block_for_point(tile_index_gdf, 100.0, 100.0, str(tile_index_gdf.crs)) == "B-0000X-0000Y"
    assert (
        block_for_point(tile_index_gdf, 7000.0, 100.0, str(tile_index_gdf.crs)) == "B-0001X-0000Y"
    )
    with pytest.raises(ValueError, match="no tile"):
        block_for_point(tile_index_gdf, -1.0, -1.0, str(tile_index_gdf.crs))
