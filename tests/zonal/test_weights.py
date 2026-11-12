"""Sparse zone-by-coarse-cell weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid, grid_from_bounds
from idd_spatial_analysis.raster import NearestIndexMap
from idd_spatial_analysis.zonal import (
    ZoneCellPairs,
    apply_zone_weights,
    weighted_sums,
    zone_weights,
)


@pytest.fixture
def coarse(grid: BlockGrid) -> BlockGrid:
    """16 x 16 cells of 400 m; every cell lies inside one quadrant zone."""
    return BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=16
    )


def test_weights_conserve_population_per_zone(
    grid: BlockGrid, coarse: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    table = zone_weights(mask, population, grid, coarse)
    assert list(table.columns) == ["zone", "coarse_row", "coarse_col", "weight"]
    assert table["zone"].dtype == np.int32
    assert table["coarse_row"].dtype == np.int32
    assert table["weight"].dtype == np.float64
    per_zone = table.groupby("zone")["weight"].sum()
    sums = weighted_sums(
        mask, population, np.zeros(grid.shape), nan_policy="keep_weight"
    ).set_index("zone")["weight_sum"]
    assert per_zone.index.tolist() == sums.index.tolist()
    np.testing.assert_allclose(per_zone.to_numpy(), sums.to_numpy(), rtol=1e-9)
    assert table["coarse_row"].between(0, 15).all()
    assert table["coarse_col"].between(0, 15).all()
    # 256 cells minus the four whose 4 x 4 fine pixels are all NaN (the top-left 8 x 8 corner)
    assert len(table) == 252
    cells = table.set_index(["coarse_row", "coarse_col"])["weight"]
    assert (0, 0) not in cells.index
    assert cells.loc[(5, 5)] == 0.0


def test_cells_straddling_zones_match_brute_force(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    coarse = BlockGrid(
        crs=grid.crs, transform=Affine(700.0, 0, -150.0, 0, -700.0, 6550.0), height=10, width=10
    )
    table = zone_weights(mask, population, grid, coarse)
    xs, ys = grid.pixel_centers()
    expect: dict[tuple[int, int, int], float] = {}
    for row in range(grid.height):
        for col in range(grid.width):
            zone, weight = int(mask[row, col]), float(population[row, col])
            if zone == 0 or not np.isfinite(weight):
                continue
            coarse_row = int(np.floor((ys[row] - 6550.0) / -700.0))
            coarse_col = int(np.floor((xs[col] + 150.0) / 700.0))
            key = (zone, coarse_row, coarse_col)
            expect[key] = expect.get(key, 0.0) + weight
    got = {
        (int(r.zone), int(r.coarse_row), int(r.coarse_col)): float(r.weight)
        for r in table.itertuples(index=False)
    }
    assert got.keys() == expect.keys()
    for key, weight in expect.items():
        assert got[key] == pytest.approx(weight, rel=1e-9)
    assert table.groupby(["coarse_row", "coarse_col"])["zone"].nunique().max() > 1


def test_uncovered_pixels_are_an_error_unless_allowed(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    half = BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=8
    )
    with pytest.raises(ValueError, match="outside the coarse grid"):
        zone_weights(mask, population, grid, half)
    table = zone_weights(mask, population, grid, half, allow_uncovered=True)
    assert set(table["zone"]) == {1, 3}
    assert table["weight"].sum() == pytest.approx(np.nansum(population[:, :32].astype(np.float64)))


def test_coarse_grid_in_another_crs(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    lonlat = grid_from_bounds(grid.lonlat_bounds(pad=500.0), 0.005, WGS84)
    table = zone_weights(mask, population, grid, lonlat)
    assert table["weight"].sum() == pytest.approx(np.nansum(population.astype(np.float64)))
    assert len(table) > 4


def test_bad_inputs(grid: BlockGrid, population: np.ndarray) -> None:
    with pytest.raises(ValueError, match="must match"):
        zone_weights(np.zeros((3, 3), dtype="int32"), population, grid, grid)
    with pytest.raises(TypeError, match="integer"):
        zone_weights(np.zeros(grid.shape), population, grid, grid)


@pytest.fixture
def straddling(grid: BlockGrid) -> BlockGrid:
    """10 x 10 cells of 700 m, offset from the block so that cells straddle the zone borders."""
    return BlockGrid(
        crs=grid.crs, transform=Affine(700.0, 0, -150.0, 0, -700.0, 6550.0), height=10, width=10
    )


def _field(shape: tuple[int, ...], seed: int = 3) -> np.ndarray:
    """A float32 field with two missing cells in every slice."""
    field = np.random.default_rng(seed).normal(20.0, 5.0, size=shape).astype("float32")
    field[..., 2, 3] = np.nan
    field[..., 7, 7] = np.nan
    return field


def test_pairs_group_the_zoned_pixels_by_zone_and_cell(
    grid: BlockGrid, coarse: BlockGrid, mask: np.ndarray
) -> None:
    pairs = ZoneCellPairs.build(mask, grid, coarse)
    assert len(pairs) == 256
    assert pairs.pixel_counts.tolist() == [16] * 256
    assert pairs.pixels.size == grid.height * grid.width
    assert pairs.uncovered.size == 0
    assert pairs.fine_shape == grid.shape
    table = pairs.table()
    assert list(table.columns) == ["zone", "coarse_row", "coarse_col", "pixel_count"]
    assert table["zone"].dtype == np.int32
    assert table["coarse_row"].dtype == np.int32
    assert table["pixel_count"].dtype == np.int64
    ordered = table.sort_values(["zone", "coarse_row", "coarse_col"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(table, ordered)
    # every pixel sits in the pair that names its own zone and its own 400 m cell
    rows, cols = np.divmod(pairs.pixels, grid.width)
    np.testing.assert_array_equal(pairs.zones[pairs.pair_of_pixel], mask[rows, cols])
    np.testing.assert_array_equal(pairs.coarse_rows[pairs.pair_of_pixel], rows // 4)
    np.testing.assert_array_equal(pairs.coarse_cols[pairs.pair_of_pixel], cols // 4)


def test_one_pairing_serves_several_weight_rasters(
    grid: BlockGrid, straddling: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    """Two "years" of population on the same pairs give what zone_weights gives for each."""
    pairs = ZoneCellPairs.build(mask, grid, straddling)
    later = population * 1.5
    later[40:44, 40:44] = np.nan
    for raster in (population, later):
        sums = pairs.sum_weight(raster)
        present = pairs.finite_count(raster) > 0
        table = zone_weights(mask, raster, grid, straddling)
        assert sums.dtype == np.float64
        assert int(present.sum()) == len(table)
        assert pairs.zones[present].tolist() == table["zone"].tolist()
        assert pairs.coarse_rows[present].tolist() == table["coarse_row"].tolist()
        np.testing.assert_allclose(sums[present], table["weight"].to_numpy(), rtol=1e-12)
        assert (sums[~present] == 0).all()
        assert sums.sum() == pytest.approx(np.nansum(raster.astype(np.float64)), rel=1e-12)
    assert pairs.finite_count(population).sum() == np.isfinite(population).sum()
    with pytest.raises(ValueError, match="does not match the block"):
        pairs.sum_weight(population[:10])


def test_pairs_report_the_pixels_a_coarse_grid_does_not_cover(
    grid: BlockGrid, mask: np.ndarray
) -> None:
    half = BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=8
    )
    pairs = ZoneCellPairs.build(mask, grid, half)
    assert set(pairs.zones.tolist()) == {1, 3}
    assert pairs.pixels.size == 64 * 32
    assert pairs.uncovered.size == 64 * 32
    _, uncovered_cols = np.divmod(pairs.uncovered, grid.width)
    assert uncovered_cols.min() == 32


def test_pairs_of_a_block_without_zones_are_empty_and_bad_masks_are_refused(
    grid: BlockGrid, coarse: BlockGrid
) -> None:
    pairs = ZoneCellPairs.build(np.zeros(grid.shape, dtype="int32"), grid, coarse)
    assert len(pairs) == 0
    assert pairs.table().empty
    assert pairs.sum_weight(np.ones(grid.shape)).size == 0
    assert pairs.finite_count(np.ones(grid.shape)).size == 0
    with pytest.raises(ValueError, match="must match the grid"):
        ZoneCellPairs.build(np.zeros((3, 3), dtype="int32"), grid, coarse)
    with pytest.raises(TypeError, match="integer"):
        ZoneCellPairs.build(np.zeros(grid.shape), grid, coarse)


@pytest.mark.parametrize("policy", ["keep_weight", "drop_pixel"])
def test_apply_equals_weighted_sums_on_the_resampled_field(
    grid: BlockGrid, straddling: BlockGrid, mask: np.ndarray, population: np.ndarray, policy: str
) -> None:
    """The coarse product and the pixel pass are the same two sums per zone."""
    field = _field(straddling.shape)
    table = zone_weights(mask, population, grid, straddling)
    got = apply_zone_weights(table, field, nan_policy=policy)  # type: ignore[arg-type]
    on_pixels = NearestIndexMap.build(straddling, grid).apply(field)
    expected = weighted_sums(mask, population, on_pixels, nan_policy=policy)  # type: ignore[arg-type]
    assert list(got.columns) == ["zone", "weighted_sum", "weight_sum"]
    assert got["zone"].dtype == np.int32
    pd.testing.assert_frame_equal(got, expected, rtol=1e-12)
    assert np.isnan(on_pixels[mask == 4]).any()


def test_policies_differ_only_in_the_weight_of_missing_cells(
    grid: BlockGrid, straddling: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    field = _field(straddling.shape)
    table = zone_weights(mask, population, grid, straddling)
    keep = apply_zone_weights(table, field)
    drop = apply_zone_weights(table, field, nan_policy="drop_pixel")
    pd.testing.assert_series_equal(keep["weighted_sum"], drop["weighted_sum"])
    assert (keep["weight_sum"] >= drop["weight_sum"]).all()
    assert (keep["weight_sum"] > drop["weight_sum"]).any()
    pd.testing.assert_frame_equal(keep, apply_zone_weights(table, field, nan_policy="keep_weight"))
    as_zero = apply_zone_weights(table, np.nan_to_num(field, nan=0.0), nan_policy="drop_pixel")
    pd.testing.assert_frame_equal(keep, as_zero)


def test_apply_to_a_stack_and_to_a_named_weight_column(
    grid: BlockGrid, straddling: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    """A table with one weight column per year, applied to a stack of fields."""
    pairs = ZoneCellPairs.build(mask, grid, straddling)
    later = population * 1.5
    table = pairs.table().assign(
        weight_2000=pairs.sum_weight(population), weight_2010=pairs.sum_weight(later)
    )
    stack = _field((3, *straddling.shape))
    on_pixels = NearestIndexMap.build(straddling, grid).apply(stack)
    for column, raster in (("weight_2000", population), ("weight_2010", later)):
        got = apply_zone_weights(table, stack, weight_col=column)
        expected = weighted_sums(mask, raster, on_pixels)
        assert list(got.columns) == ["zone", "slice", "weighted_sum", "weight_sum"]
        assert got["slice"].tolist() == [0] * 4 + [1] * 4 + [2] * 4
        pd.testing.assert_frame_equal(got, expected, rtol=1e-12)


def test_apply_with_a_coarse_grid_in_another_crs(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    lonlat = grid_from_bounds(grid.lonlat_bounds(pad=500.0), 0.005, WGS84)
    field = np.random.default_rng(5).normal(size=lonlat.shape).astype("float32")
    got = apply_zone_weights(zone_weights(mask, population, grid, lonlat), field)
    expected = weighted_sums(mask, population, NearestIndexMap.build(lonlat, grid).apply(field))
    pd.testing.assert_frame_equal(got, expected, rtol=1e-12)


def test_apply_refuses_tables_and_fields_that_do_not_fit(
    grid: BlockGrid, straddling: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    table = zone_weights(mask, population, grid, straddling)
    field = _field(straddling.shape)
    with pytest.raises(ValueError, match="built for another grid"):
        apply_zone_weights(table, field[:5])
    with pytest.raises(ValueError, match="built for another grid"):
        apply_zone_weights(table.assign(coarse_col=table["coarse_col"] - 1), field)
    with pytest.raises(ValueError, match=r"\(rows, cols\) or \(slices, rows, cols\)"):
        apply_zone_weights(table, field.ravel())
    with pytest.raises(KeyError, match="weight_1999"):
        apply_zone_weights(table, field, weight_col="weight_1999")
    with pytest.raises(KeyError, match="coarse_row"):
        apply_zone_weights(table.drop(columns=["coarse_row"]), field)
    with pytest.raises(ValueError, match="NaN or infinite weights"):
        apply_zone_weights(table.assign(weight=np.nan), field)
    with pytest.raises(ValueError, match="nan_policy must be one of"):
        apply_zone_weights(table, field, nan_policy="skip")  # type: ignore[arg-type]


def test_apply_to_an_empty_table_gives_empty_sums(straddling: BlockGrid) -> None:
    empty = pd.DataFrame(
        {
            "zone": pd.Series([], dtype="int32"),
            "coarse_row": pd.Series([], dtype="int32"),
            "coarse_col": pd.Series([], dtype="int32"),
            "weight": pd.Series([], dtype="float64"),
        }
    )
    flat = apply_zone_weights(empty, _field(straddling.shape))
    assert list(flat.columns) == ["zone", "weighted_sum", "weight_sum"]
    assert flat.empty
    assert flat["zone"].dtype == np.int32
    stacked = apply_zone_weights(empty, _field((2, *straddling.shape)))
    assert list(stacked.columns) == ["zone", "slice", "weighted_sum", "weight_sum"]
    assert stacked.empty


def test_pairs_remember_their_grids_and_how_the_centres_were_transformed(
    grid: BlockGrid, coarse: BlockGrid, mask: np.ndarray
) -> None:
    pairs = ZoneCellPairs.build(mask, grid, coarse)
    assert pairs.grid == grid
    assert pairs.coarse == coarse
    assert pairs.centre_transform == "same_crs"
    lonlat = grid_from_bounds(grid.lonlat_bounds(), 0.005, WGS84)
    assert ZoneCellPairs.build(mask, grid, lonlat).centre_transform == "row_and_column"


def test_weight_table_is_zone_weights_for_a_pairing_built_once(
    grid: BlockGrid,
    coarse: BlockGrid,
    straddling: BlockGrid,
    mask: np.ndarray,
    population: np.ndarray,
) -> None:
    pairs = ZoneCellPairs.build(mask, grid, straddling)
    pd.testing.assert_frame_equal(
        pairs.weight_table(population), zone_weights(mask, population, grid, straddling)
    )
    assert pairs.uncovered_weight(population) == (0, 0.0)

    # A pair whose pixels are all NaN is left out; one whose pixels are all zero stays.
    aligned = ZoneCellPairs.build(mask, grid, coarse)
    weight = np.ones(grid.shape, dtype="float32")
    weight[0:4, 0:4] = np.nan
    weight[0:4, 4:8] = 0.0
    table = aligned.weight_table(weight)
    cells = set(zip(table["coarse_row"].tolist(), table["coarse_col"].tolist(), strict=True))
    assert (0, 0) not in cells
    assert len(table) == len(aligned) - 1
    zero = table.loc[(table["coarse_row"] == 0) & (table["coarse_col"] == 1), "weight"]
    assert zero.tolist() == [0.0]
    assert table["weight"].dtype == np.float64
    assert table["zone"].dtype == np.int32

    with pytest.raises(ValueError, match="does not match the block"):
        pairs.weight_table(population[:10])
    with pytest.raises(ValueError, match="does not match the block"):
        pairs.uncovered_weight(population[:10])


def test_uncovered_weight_counts_what_a_small_coarse_grid_loses(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    west_half = BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=8
    )
    pairs = ZoneCellPairs.build(mask, grid, west_half)
    lost = population[:, 32:].astype(np.float64)
    count, total = pairs.uncovered_weight(population)
    assert count == int(np.isfinite(lost).sum())
    assert total == pytest.approx(np.nansum(lost), rel=1e-12)
    with pytest.raises(ValueError, match="fall outside the coarse grid"):
        pairs.weight_table(population)
    kept = pairs.weight_table(population, allow_uncovered=True)
    assert kept["weight"].sum() == pytest.approx(
        np.nansum(population[:, :32].astype(np.float64)), rel=1e-12
    )
