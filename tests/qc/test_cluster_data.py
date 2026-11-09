"""Checks against the cluster data named in paths.yaml.

Schema and dtype checks on a real LBD release, whole-pixel block grids on the real tile
index, raking shapes against the hierarchy's leaves, and population conservation of the
block weights on one real block. The comparison of a block's zone weights against
climate-aggregates' own population parquet (PROPOSED_STRUCTURE §Testing) is not here yet; see
STATUS.md's parking lot.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from idd_spatial_analysis.boundaries import load_admin, load_hierarchy, load_raking_shapes
from idd_spatial_analysis.constants import ADM_CODE_COLUMN, ADM_NAME_COLUMN, LOC_ID_COLUMN
from idd_spatial_analysis.frame import (
    block_bounds,
    block_grid,
    blocks_intersecting,
    load_tile_index,
)
from idd_spatial_analysis.raster import rasterize_zones, read_window
from idd_spatial_analysis.zonal import weighted_sums, zone_weights

if TYPE_CHECKING:
    from pathlib import Path

    from idd_spatial_analysis.boundaries import AdminRelease

pytestmark = pytest.mark.qc

MIN_CELLS = 150


def test_lbd_release_schema_and_dtypes(admin_release: AdminRelease) -> None:
    countries = load_admin(admin_release, 0, columns=[LOC_ID_COLUMN])
    assert len(countries) > MIN_CELLS
    assert str(countries[ADM_CODE_COLUMN.format(level=0)].dtype) == "int32"
    assert str(countries[LOC_ID_COLUMN].dtype) == "Int32"
    assert countries.geometry.is_valid.all()
    code = int(countries[ADM_CODE_COLUMN.format(level=0)].iloc[0])
    for level in (1, 2):
        units = load_admin(admin_release, level, countries=[code])
        assert (units[ADM_CODE_COLUMN.format(level=0)] == code).all()
        assert str(units[ADM_CODE_COLUMN.format(level=level)].dtype) == "int32"
        assert ADM_NAME_COLUMN.format(level=level) in units.columns
        assert units[ADM_CODE_COLUMN.format(level=level)].is_unique
        assert (units[ADM_CODE_COLUMN.format(level=level)] > 0).all()


def test_tile_index_blocks_are_whole_pixels(tile_index_path: Path) -> None:
    tiles = load_tile_index(tile_index_path)
    assert len(tiles) > MIN_CELLS
    resolution = float(os.environ.get("IDD_SPATIAL_QC_RESOLUTION", "100"))
    keys = tiles["block_key"].drop_duplicates().iloc[:5]
    for key in keys:
        left, bottom, right, top = block_bounds(tiles, key)
        grid = block_grid(tiles, key, resolution)
        assert grid.width == pytest.approx((right - left) / resolution)
        assert grid.height == pytest.approx((top - bottom) / resolution)
        assert grid.bounds[0] == left
        assert grid.bounds[3] == top


def test_raking_shapes_are_hierarchy_leaves(raking_root: Path, hierarchy_name: str) -> None:
    hierarchy = load_hierarchy(raking_root, hierarchy_name)
    shapes = load_raking_shapes(raking_root, hierarchy_name)
    assert len(shapes) > MIN_CELLS
    assert shapes["location_id"].is_unique
    parents = set(hierarchy.loc[hierarchy["level"] > 0, "parent_id"].tolist())
    leaves = set(hierarchy["location_id"].tolist()) - parents
    missing = set(shapes["location_id"].tolist()) - leaves
    assert not missing, f"{len(missing)} shape ids are not most-detailed locations"


def test_block_weights_conserve_population(
    tile_index_path: Path, raking_root: Path, hierarchy_name: str, population_tif: Path
) -> None:
    resolution = float(os.environ.get("IDD_SPATIAL_QC_RESOLUTION", "100"))
    tiles = load_tile_index(tile_index_path)
    shapes = load_raking_shapes(raking_root, hierarchy_name)
    block_key = os.environ.get("IDD_SPATIAL_QC_BLOCK_KEY") or blocks_intersecting(tiles, shapes)[0]
    grid = block_grid(tiles, block_key, resolution)
    population, sub = read_window(population_tif, grid.bounds, boundless=True)
    assert sub.shape == grid.shape
    mask, _ = rasterize_zones(shapes, "location_id", grid)
    assert (mask > 0).sum() > MIN_CELLS
    sums = weighted_sums(mask, population, np.zeros(grid.shape), nan_policy="keep_weight")
    coarse = block_grid(tiles, block_key, resolution * 32)
    weights = zone_weights(mask, population, grid, coarse)
    assert len(weights) > MIN_CELLS
    per_zone = weights.groupby("zone")["weight"].sum()
    expect = sums.set_index("zone")["weight_sum"]
    assert per_zone.index.tolist() == expect.index.tolist()
    np.testing.assert_allclose(per_zone.to_numpy(), expect.to_numpy(), rtol=1e-9)
    zoned = np.nansum(population[mask > 0].astype(np.float64))
    assert weights["weight"].sum() == pytest.approx(zoned, rel=1e-9)
