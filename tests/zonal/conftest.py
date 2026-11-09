"""Zone mask and a value field for the zonal tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from idd_spatial_analysis.raster import rasterize_zones

if TYPE_CHECKING:
    import geopandas as gpd
    from rasterio.windows import Window

    from idd_spatial_analysis.frame import BlockGrid


@pytest.fixture
def mask_windows(
    grid: BlockGrid, zones_gdf: gpd.GeoDataFrame
) -> tuple[np.ndarray, dict[int, Window]]:
    return rasterize_zones(zones_gdf, "zone", grid)


@pytest.fixture
def mask(mask_windows: tuple[np.ndarray, dict[int, Window]]) -> np.ndarray:
    return mask_windows[0]


@pytest.fixture
def value(grid: BlockGrid) -> np.ndarray:
    """A float32 field with a NaN patch inside zone 2 (rows 10..19, cols 40..49)."""
    rng = np.random.default_rng(2)
    field = rng.normal(size=grid.shape).astype("float32")
    field[10:20, 40:50] = np.nan
    return field
