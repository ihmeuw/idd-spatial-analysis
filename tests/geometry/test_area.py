"""Cell areas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from idd_spatial_analysis.constants import EARTH_RADIUS_KM, WGS84
from idd_spatial_analysis.frame import BlockGrid
from idd_spatial_analysis.geometry import cell_area_km2, grid_cell_areas_km2


def test_one_degree_rows_sum_to_the_sphere() -> None:
    areas = cell_area_km2(np.arange(90, -91, -1), 1.0)
    assert areas.shape == (180,)
    assert areas.sum() * 360 == pytest.approx(4 * math.pi * EARTH_RADIUS_KM**2, rel=1e-12)
    assert areas[0] < areas[89]


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="1-D"):
        cell_area_km2(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="positive"):
        cell_area_km2(np.array([1.0, 0.0]), 0.0)


def test_grid_cell_areas_geographic_and_metric(grid: BlockGrid) -> None:
    metric = grid_cell_areas_km2(grid)
    assert metric.shape == (grid.height,)
    assert np.allclose(metric, 0.01)
    lonlat = BlockGrid(crs=WGS84, transform=Affine(0.1, 0, 0, 0, -0.1, 90), height=1800, width=3600)
    areas = grid_cell_areas_km2(lonlat)
    assert areas.sum() * lonlat.width == pytest.approx(4 * math.pi * EARTH_RADIUS_KM**2, rel=1e-12)


def test_grid_cell_areas_rejects_other_units() -> None:
    feet = BlockGrid(crs="EPSG:2227", transform=Affine(100, 0, 0, 0, -100, 100), height=1, width=1)
    with pytest.raises(ValueError, match="neither"):
        grid_cell_areas_km2(feet)
