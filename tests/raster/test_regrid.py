"""Nearest lookups and rasterio warps."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid, grid_from_bounds
from idd_spatial_analysis.raster import NearestIndexMap, reproject_to_grid, resample_masked


@pytest.fixture
def coarse(grid: BlockGrid) -> BlockGrid:
    return BlockGrid(
        crs=grid.crs, transform=Affine(200.0, 0, 0, 0, -200.0, 6400.0), height=32, width=32
    )


def test_same_crs_lookup_is_separable_and_picks_centre_pixels(
    grid: BlockGrid, coarse: BlockGrid, population: np.ndarray
) -> None:
    lookup = NearestIndexMap.build(grid, coarse)
    assert lookup.rows.ndim == 1
    assert not lookup.masked
    out = lookup.apply(population)
    np.testing.assert_array_equal(out, population[1::2, 1::2])
    rows, cols = lookup.indices()
    assert rows.shape == coarse.shape
    assert rows[0, 0] == 1
    assert cols[0, 5] == 11


def test_target_pixels_outside_source_get_fill(grid: BlockGrid, population: np.ndarray) -> None:
    target = grid_from_bounds((-1000.0, -1000.0, 1000.0, 1000.0), 100.0, grid.crs)
    out = NearestIndexMap.build(grid, target).apply(population, fill=-9.0)
    assert (out[10:, :] == -9.0).all()
    assert (out[:, :10] == -9.0).all()
    np.testing.assert_array_equal(out[:10, 10:], population[54:64, 0:10])


def test_other_crs_lookup_matches_brute_force(grid: BlockGrid, population: np.ndarray) -> None:
    target = grid_from_bounds(grid.lonlat_bounds(), 0.001, WGS84)
    lookup = NearestIndexMap.build(grid, target)
    assert lookup.rows.ndim == 2
    xs, ys = target.pixel_centers()
    grid_x, grid_y = np.meshgrid(xs, ys)
    tx, ty = Transformer.from_crs(WGS84, grid.crs, always_xy=True).transform(grid_x, grid_y)
    rows = np.floor((ty - 6400.0) / -100.0)
    cols = np.floor(tx / 100.0)
    inside = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
    assert inside.sum() > 1000
    got_rows, got_cols = lookup.indices()
    np.testing.assert_array_equal(got_rows[inside], rows[inside])
    np.testing.assert_array_equal(got_cols[inside], cols[inside])
    assert (got_rows[~inside] == -1).all()
    expect = np.full(target.shape, np.nan, dtype=np.float32)
    expect[inside] = population[rows[inside].astype(int), cols[inside].astype(int)]
    np.testing.assert_array_equal(lookup.apply(population), expect)


def test_masked_lookup_fills_only_the_mask(
    grid: BlockGrid, coarse: BlockGrid, population: np.ndarray
) -> None:
    rng = np.random.default_rng(1)
    mask = rng.random(coarse.shape) < 0.3
    out = resample_masked(population, grid, coarse, mask)
    full = NearestIndexMap.build(grid, coarse).apply(population)
    np.testing.assert_array_equal(out[mask], full[mask])
    assert np.isnan(out[~mask]).all()
    lookup = NearestIndexMap.build(grid, coarse, target_mask=mask)
    assert lookup.masked
    with pytest.raises(ValueError, match="masked pixels"):
        lookup.indices()
    with pytest.raises(ValueError, match="target_mask shape"):
        NearestIndexMap.build(grid, coarse, target_mask=np.ones((3, 3), dtype=bool))


def test_masked_lookup_in_another_crs(grid: BlockGrid, population: np.ndarray) -> None:
    target = grid_from_bounds(grid.lonlat_bounds(), 0.001, WGS84)
    rng = np.random.default_rng(3)
    mask = rng.random(target.shape) < 0.2
    out = resample_masked(population, grid, target, mask)
    full = NearestIndexMap.build(grid, target).apply(population)
    np.testing.assert_array_equal(out[mask], full[mask])
    assert np.isnan(out[~mask]).all()


def test_apply_handles_stacks_and_rejects_bad_inputs(
    grid: BlockGrid, coarse: BlockGrid, population: np.ndarray
) -> None:
    lookup = NearestIndexMap.build(grid, coarse)
    stack = np.stack([population, 2 * population])
    out = lookup.apply(stack)
    assert out.shape == (2, 32, 32)
    np.testing.assert_array_equal(out[1], 2 * population[1::2, 1::2])
    with pytest.raises(ValueError, match="source shape"):
        lookup.apply(np.ones((10, 10)))
    with pytest.raises(ValueError, match="representable"):
        lookup.apply(np.ones(grid.shape, dtype="int32"))
    ints = lookup.apply(np.ones(grid.shape, dtype="int32"), fill=0)
    assert ints.dtype == np.int32
    assert ints.sum() == 32 * 32


def test_reproject_identity_and_nearest_agree_with_lookup(
    grid: BlockGrid, population: np.ndarray
) -> None:
    same = reproject_to_grid(population, grid, grid)
    np.testing.assert_array_equal(same, population)
    offset = BlockGrid(
        crs=grid.crs, transform=Affine(200.0, 0, 50.0, 0, -200.0, 6350.0), height=31, width=31
    )
    warped = reproject_to_grid(population, grid, offset, resampling="nearest")
    looked = NearestIndexMap.build(grid, offset).apply(population)
    np.testing.assert_array_equal(warped, looked)


def test_reproject_to_lonlat_and_bad_inputs(grid: BlockGrid, population: np.ndarray) -> None:
    target = grid_from_bounds(grid.lonlat_bounds(), 0.002, WGS84)
    out = reproject_to_grid(population, grid, target, resampling="average")
    assert out.shape == target.shape
    assert np.isfinite(out).sum() > 100
    stack = reproject_to_grid(np.stack([population, population]), grid, target)
    assert stack.shape == (2, *target.shape)
    with pytest.raises(ValueError, match="integer nodata"):
        reproject_to_grid(np.ones(grid.shape, dtype="int32"), grid, target)
    with pytest.raises(ValueError, match="grid shape"):
        reproject_to_grid(np.ones((3, 3)), grid, target)
    with pytest.raises(KeyError):
        reproject_to_grid(population, grid, target, resampling="nope")
