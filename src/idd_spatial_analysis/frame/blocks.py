"""Tile indexes and blocks: the partition of a frame into fixed-size rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from shapely.geometry import Point

from idd_spatial_analysis.constants import BLOCK_KEY_COLUMN, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.crs import crs_equal, crs_string
from idd_spatial_analysis.frame.grid import BlockGrid, grid_from_bounds
from idd_spatial_analysis.geometry.antimeridian import check_lon_180

if TYPE_CHECKING:
    from pathlib import Path

    from idd_spatial_analysis.frame.bounds import Bounds


def load_tile_index(
    path: Path | str, *, expected_crs: str = WORLD_CYLINDRICAL_EQUAL_AREA
) -> gpd.GeoDataFrame:
    """Read a tile index (one row per tile, a ``block_key`` column) and check its CRS."""
    tiles = gpd.read_parquet(path)
    _check_tile_index(tiles, expected_crs)
    return tiles


def _check_tile_index(tiles: gpd.GeoDataFrame, expected_crs: str | None) -> None:
    if BLOCK_KEY_COLUMN not in tiles.columns:
        msg = f"tile index lacks a {BLOCK_KEY_COLUMN!r} column"
        raise KeyError(msg)
    if tiles.empty:
        msg = "tile index is empty"
        raise ValueError(msg)
    if tiles.crs is None:
        msg = "tile index carries no CRS"
        raise ValueError(msg)
    if expected_crs is not None and not crs_equal(tiles.crs, expected_crs):
        msg = f"tile index CRS is {tiles.crs}, expected {expected_crs}"
        raise ValueError(msg)


def block_bounds(tiles: gpd.GeoDataFrame, block_key: str) -> Bounds:
    """Union bounds of the tiles of one block."""
    rows = tiles.loc[tiles[BLOCK_KEY_COLUMN] == block_key]
    if rows.empty:
        msg = f"no tile carries block_key {block_key!r}"
        raise KeyError(msg)
    left, bottom, right, top = rows.total_bounds
    return (float(left), float(bottom), float(right), float(top))


def block_grid(tiles: gpd.GeoDataFrame, block_key: str, resolution: float) -> BlockGrid:
    """The raster grid of one block at ``resolution`` in the tile index's CRS."""
    return grid_from_bounds(block_bounds(tiles, block_key), resolution, crs_string(tiles.crs))


def blocks_intersecting(tiles: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> list[str]:
    """Keys of the blocks whose footprint intersects at least one polygon, sorted.

    Polygons are reprojected to the tile index's CRS; lon/lat polygons must be in the -180..180
    convention (``geometry.to_lon_180``), or reprojection wraps them around the world. Raises
    when no block qualifies, so a bad input cannot silently skip every block.
    """
    _check_tile_index(tiles, None)
    if polygons.crs is None:
        msg = "polygons carry no CRS"
        raise ValueError(msg)
    check_lon_180(polygons)
    blocks = tiles[[BLOCK_KEY_COLUMN, "geometry"]].dissolve(by=BLOCK_KEY_COLUMN).reset_index()
    shapes = polygons[["geometry"]]
    if not crs_equal(polygons.crs, tiles.crs):
        shapes = shapes.to_crs(tiles.crs)
    hits = gpd.sjoin(blocks, shapes, how="inner", predicate="intersects")
    keys = sorted({str(key) for key in hits[BLOCK_KEY_COLUMN]})
    if not keys:
        msg = "no block intersects any of the polygons"
        raise ValueError(msg)
    return keys


def block_for_point(tiles: gpd.GeoDataFrame, x: float, y: float, crs: str) -> str:
    """The block containing a point given in ``crs``.

    A point on a shared tile edge belongs to the first matching tile in index order.
    """
    point = gpd.GeoSeries([Point(x, y)], crs=crs).to_crs(tiles.crs).iloc[0]
    hits = tiles.loc[tiles.intersects(point), BLOCK_KEY_COLUMN]
    if hits.empty:
        msg = f"point ({x}, {y}) in {crs} falls in no tile"
        raise ValueError(msg)
    return str(hits.iloc[0])
