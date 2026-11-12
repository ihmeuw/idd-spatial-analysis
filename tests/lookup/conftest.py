"""Two adjoining blocks and zones that straddle them, for the lookup tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from idd_spatial_analysis.constants import WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import block_grid

if TYPE_CHECKING:
    from idd_spatial_analysis.frame import BlockGrid

WEST, EAST = "B-0000X-0000Y", "B-0001X-0000Y"
WIDE_ZONE_IDS = [1, 2, 3, 5, 9]


@pytest.fixture
def blocks(tile_index_gdf: gpd.GeoDataFrame) -> dict[str, BlockGrid]:
    """The two 64 x 64 blocks of the synthetic tile index, side by side, at 100 m."""
    return {key: block_grid(tile_index_gdf, key, 100.0) for key in (WEST, EAST)}


@pytest.fixture
def wide_zones_gdf() -> gpd.GeoDataFrame:
    """Zones over both blocks (x 0..12800 m, y 0..6400 m).

    Zone 1 covers the top half of the west block and the first 32 columns of the east block,
    zone 2 the rest of the east block's top half, zone 3 the whole bottom half. Zone 5 is a
    40 m square inside one pixel, so it holds no pixel centre; zone 9 lies far away.
    """
    return gpd.GeoDataFrame(
        {"zone": np.array(WIDE_ZONE_IDS, dtype="int32")},
        geometry=[
            box(0, 3200, 9600, 6400),
            box(9600, 3200, 12800, 6400),
            box(0, 0, 12800, 3200),
            box(110, 110, 140, 140),
            box(50_000, 50_000, 51_000, 51_000),
        ],
        crs=WORLD_CYLINDRICAL_EQUAL_AREA,
    )
