"""Nearest lookups and rasterio warps."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import BlockGrid, grid_from_bounds
from idd_spatial_analysis.raster import NearestIndexMap, reproject_to_grid, resample_masked
from idd_spatial_analysis.raster.regrid import _row_and_column_centres

UTM33 = "EPSG:32633"
WEB_MERCATOR = "EPSG:3857"
MOLLWEIDE = "ESRI:54009"


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
    assert lookup.centre_transform == "same_crs"
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
    # Between the equal-area frame and lon/lat, x depends on the column alone and y on the row.
    assert lookup.centre_transform == "row_and_column"
    assert lookup.rows.ndim == 1
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


def test_reproject_without_nodata_leaves_uncovered_pixels_nan(
    grid: BlockGrid, population: np.ndarray
) -> None:
    target = grid_from_bounds((-2000.0, -2000.0, 2000.0, 2000.0), 100.0, grid.crs)
    out = reproject_to_grid(population, grid, target, nodata=None)
    assert np.isnan(out[:, :20]).all()
    assert np.isnan(out[20:, :]).all()
    np.testing.assert_array_equal(out[:20, 20:], population[44:64, 0:20])


def test_lookup_edge_tolerance_on_a_non_binary_lattice() -> None:
    fine = BlockGrid(
        crs=WGS84, transform=Affine(1 / 120, 0, -73.3, 0, -1 / 120, 45.7), height=64, width=64
    )
    coarse = BlockGrid(
        crs=WGS84, transform=Affine(1 / 60, 0, -73.3, 0, -1 / 60, 45.7), height=32, width=32
    )
    values = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    np.testing.assert_array_equal(
        NearestIndexMap.build(fine, coarse).apply(values), values[1::2, 1::2]
    )


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


def test_row_and_column_route_equals_transforming_every_pixel(
    grid: BlockGrid, population: np.ndarray
) -> None:
    target = grid_from_bounds(grid.lonlat_bounds(), 0.001, WGS84)
    fast = NearestIndexMap.build(grid, target)
    full = NearestIndexMap.build(grid, target, allow_row_and_column=False)
    assert full.centre_transform == "every_pixel"
    assert full.rows.ndim == 2
    for got, expected in zip(fast.indices(), full.indices(), strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(fast.apply(population), full.apply(population))

    mask = np.random.default_rng(5).random(target.shape) < 0.3
    fast_masked = NearestIndexMap.build(grid, target, target_mask=mask)
    full_masked = NearestIndexMap.build(grid, target, target_mask=mask, allow_row_and_column=False)
    assert fast_masked.centre_transform == "row_and_column"
    assert full_masked.centre_transform == "every_pixel"
    assert fast_masked.selected is not None
    assert full_masked.selected is not None
    np.testing.assert_array_equal(fast_masked.selected, full_masked.selected)
    np.testing.assert_array_equal(fast_masked.rows, full_masked.rows)
    np.testing.assert_array_equal(fast_masked.cols, full_masked.cols)
    assert (fast_masked.rows == -1).any()
    assert (fast_masked.rows >= 0).any()


def test_fine_frame_pixels_into_a_lonlat_grid_take_the_row_and_column_route(
    grid: BlockGrid,
) -> None:
    """The direction the weight table uses: block pixels looked up in a coarse lon/lat grid."""
    left, bottom, right, top = grid.lonlat_bounds()
    coarse = grid_from_bounds((left, bottom, right, top), 0.01, WGS84)
    mask = np.ones(grid.shape, dtype=bool)
    fast = NearestIndexMap.build(coarse, grid, target_mask=mask)
    full = NearestIndexMap.build(coarse, grid, target_mask=mask, allow_row_and_column=False)
    assert fast.centre_transform == "row_and_column"
    np.testing.assert_array_equal(fast.rows, full.rows)
    np.testing.assert_array_equal(fast.cols, full.cols)
    assert len(np.unique(fast.rows)) > 1
    assert len(np.unique(fast.cols)) > 1


def test_crs_pair_that_is_not_separable_transforms_every_pixel() -> None:
    source = BlockGrid(
        crs=UTM33,
        transform=Affine(1000.0, 0, 400_000.0, 0, -1000.0, 5_600_000.0),
        height=64,
        width=64,
    )
    values = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    target = grid_from_bounds(source.lonlat_bounds(), 0.01, WGS84)
    lookup = NearestIndexMap.build(source, target)
    assert lookup.centre_transform == "every_pixel"
    assert lookup.rows.ndim == 2
    xs, ys = target.pixel_centers()
    grid_x, grid_y = np.meshgrid(xs, ys)
    tx, ty = Transformer.from_crs(WGS84, UTM33, always_xy=True).transform(grid_x, grid_y)
    rows = np.floor((ty - 5_600_000.0) / -1000.0)
    cols = np.floor((tx - 400_000.0) / 1000.0)
    inside = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
    assert inside.sum() > 1000
    assert (~inside).any()
    got_rows, got_cols = lookup.indices()
    np.testing.assert_array_equal(got_rows[inside], rows[inside])
    np.testing.assert_array_equal(got_cols[inside], cols[inside])
    assert (got_rows[~inside] == -1).all()

    mask = np.random.default_rng(11).random(target.shape) < 0.25
    masked = NearestIndexMap.build(source, target, target_mask=mask)
    assert masked.centre_transform == "every_pixel"
    np.testing.assert_array_equal(masked.apply(values)[mask], lookup.apply(values)[mask])


def test_row_and_column_centres_are_given_only_when_they_are_exact() -> None:
    xs = np.linspace(-10.0, 10.0, 41)
    ys = np.linspace(60.0, 40.0, 41)
    found = _row_and_column_centres(WGS84, WEB_MERCATOR, xs, ys)
    assert found is not None
    x_of_col, y_of_row = found
    grid_x, grid_y = np.meshgrid(xs, ys)
    tx, ty = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform(grid_x, grid_y)
    np.testing.assert_array_equal(tx, np.broadcast_to(x_of_col, tx.shape))
    np.testing.assert_array_equal(ty, np.broadcast_to(y_of_row[:, np.newaxis], ty.shape))
    # Mollweide's x depends on the latitude, UTM's on both coordinates.
    assert _row_and_column_centres(WGS84, MOLLWEIDE, xs, ys) is None
    assert _row_and_column_centres(WGS84, UTM33, np.linspace(12.0, 18.0, 41), ys) is None
    # North of the equal-area projection's last row there is no latitude, so the middle column
    # does not transform everywhere and the lattice is left to the pointwise method.
    metres_x = np.linspace(0.0, 4_000_000.0, 41)
    past_the_pole = np.linspace(7_000_000.0, 5_000_000.0, 41)
    assert (
        _row_and_column_centres(WORLD_CYLINDRICAL_EQUAL_AREA, WGS84, metres_x, past_the_pole)
        is None
    )
    within = np.linspace(6_000_000.0, 5_000_000.0, 41)
    assert (
        _row_and_column_centres(WORLD_CYLINDRICAL_EQUAL_AREA, WGS84, metres_x, within) is not None
    )
    # A lattice of one row and one column is its own sample.
    single = _row_and_column_centres(WGS84, WEB_MERCATOR, xs[:1], ys[:1])
    assert single is not None
