"""GeoTIFFs of the synthetic block written to tmp_path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import rasterio

from idd_spatial_analysis.raster import rasterize_zones

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    import numpy as np

    from idd_spatial_analysis.frame import BlockGrid


@pytest.fixture
def population_tif(tmp_path: Path, grid: BlockGrid, population: np.ndarray) -> Path:
    path = tmp_path / "population.tif"
    with rasterio.open(path, "w", **grid.profile()) as dst:
        dst.write(population, 1)
    return path


@pytest.fixture
def zone_tif(tmp_path: Path, grid: BlockGrid, zones_gdf: gpd.GeoDataFrame) -> Path:
    mask, _ = rasterize_zones(zones_gdf, "zone", grid)
    path = tmp_path / "zones.tif"
    with rasterio.open(path, "w", **grid.profile(dtype="int32", nodata=0)) as dst:
        dst.write(mask, 1)
    return path
