"""rasterra adapters (skipped without the extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterra")

from typing import TYPE_CHECKING

from idd_spatial_analysis.crs import crs_equal
from idd_spatial_analysis.raster.adapters import from_raster_array, to_raster_array

if TYPE_CHECKING:
    from idd_spatial_analysis.frame import BlockGrid


def test_round_trip(grid: BlockGrid, population: np.ndarray) -> None:
    raster = to_raster_array(population, grid, nodata=np.nan)
    assert raster.x_resolution == 100.0
    assert abs(raster.y_resolution) == 100.0
    data, back = from_raster_array(raster)
    np.testing.assert_array_equal(data, population)
    assert back.transform == grid.transform
    assert crs_equal(back.crs, grid.crs)
    assert back.shape == grid.shape


def test_shape_mismatch_raises(grid: BlockGrid) -> None:
    with pytest.raises(ValueError, match="does not match"):
        to_raster_array(np.ones((3, 3)), grid)
