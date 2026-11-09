"""Sparse zone-by-coarse-cell weights."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid, grid_from_bounds
from idd_spatial_analysis.zonal import weighted_sums, zone_weights


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


def test_pixels_outside_the_coarse_grid_are_dropped(
    grid: BlockGrid, mask: np.ndarray, population: np.ndarray
) -> None:
    half = BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=8
    )
    table = zone_weights(mask, population, grid, half)
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
