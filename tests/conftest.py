"""Synthetic fixtures shared across the test tree: a small equal-area block and its zones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Point, box

from idd_spatial_analysis.boundaries import AdminRelease
from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import BlockGrid

if TYPE_CHECKING:
    from pathlib import Path

BLOCK_M = 6400.0
RES_M = 100.0
RELEASE = "2024_07_29"


def _write_release(root: Path, release: str) -> None:
    """A three-level LBD-style release: two countries, three admin 1, four admin 2 units."""
    directory = root / release
    directory.mkdir(parents=True)
    adm0 = gpd.GeoDataFrame(
        {"ADM0_CODE": [1, 2], "ADM0_NAME": ["Aland", "Bland"], "loc_id": [10, None]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs=WGS84,
    )
    adm1 = gpd.GeoDataFrame(
        {
            "ADM0_CODE": [1, 1, 2],
            "ADM1_CODE": [11, 12, 21],
            "ADM1_NAME": ["A-north", "A-south", "B-all"],
            "loc_id": [101, 102, 201],
        },
        geometry=[box(0, 5, 10, 10), box(0, 0, 10, 5), box(10, 0, 20, 10)],
        crs=WGS84,
    )
    adm2 = gpd.GeoDataFrame(
        {
            "ADM0_CODE": [1, 1, 1, 2],
            "ADM1_CODE": [11, 11, 12, 21],
            "ADM2_CODE": [111, 112, 121, 211],
            "ADM2_NAME": ["A-nw", "A-ne", "A-s", "B"],
            "loc_id": [1001, 1002, 1003, 2001],
        },
        geometry=[box(0, 5, 5, 10), box(5, 5, 10, 10), box(0, 0, 10, 5), box(10, 0, 20, 10)],
        crs=WGS84,
    )
    for level, gdf in enumerate((adm0, adm1, adm2)):
        for suffix in ("", "_simplified"):
            gdf.to_file(
                directory / f"lbd_standard_admin_{level}{suffix}.shp", driver="ESRI Shapefile"
            )


@pytest.fixture
def release(tmp_path: Path) -> AdminRelease:
    """The full-geometry variant of a release written under tmp_path."""
    _write_release(tmp_path / "admin_shapefiles", RELEASE)
    return AdminRelease(tmp_path / "admin_shapefiles", RELEASE, simplified=False)


@pytest.fixture
def grid() -> BlockGrid:
    """A 64 x 64 block of 100 m pixels in the equal-area frame, origin at (0, 6400)."""
    return BlockGrid(
        crs=WORLD_CYLINDRICAL_EQUAL_AREA,
        transform=Affine(RES_M, 0.0, 0.0, 0.0, -RES_M, BLOCK_M),
        height=64,
        width=64,
    )


@pytest.fixture
def zones_gdf(grid: BlockGrid) -> gpd.GeoDataFrame:
    """Four zones tiling the block in quadrants (1 NW, 2 NE, 3 SW, 4 SE) and one zone outside."""
    half = BLOCK_M / 2
    return gpd.GeoDataFrame(
        {
            "zone": np.array([1, 2, 3, 4, 9], dtype="int32"),
            "name": ["nw", "ne", "sw", "se", "elsewhere"],
        },
        geometry=[
            box(0, half, half, BLOCK_M),
            box(half, half, BLOCK_M, BLOCK_M),
            box(0, 0, half, half),
            box(half, 0, BLOCK_M, half),
            box(20_000, 20_000, 21_000, 21_000),
        ],
        crs=grid.crs,
    )


@pytest.fixture
def population(grid: BlockGrid) -> np.ndarray:
    """Positive float32 population with a zero band, NaN outside the top-left 8 x 8 corner."""
    rng = np.random.default_rng(0)
    pop = rng.gamma(shape=2.0, scale=50.0, size=grid.shape).astype("float32")
    pop[:, 20:24] = 0.0
    pop[:8, :8] = np.nan
    return pop


@pytest.fixture
def facilities_gdf(grid: BlockGrid) -> gpd.GeoDataFrame:
    """Six facilities with known positions: two in zone 1, one in zone 2, three in zone 4, none in 3."""
    coords = [
        (500.0, 6000.0),
        (2500.0, 4000.0),
        (5000.0, 5000.0),
        (4000.0, 1000.0),
        (6000.0, 3000.0),
        (3300.0, 300.0),
    ]
    return gpd.GeoDataFrame(
        {
            "facility_id": np.arange(6, dtype="int32"),
            "zone": np.array([1, 1, 2, 4, 4, 4], dtype="int32"),
        },
        geometry=[Point(x, y) for x, y in coords],
        crs=grid.crs,
    )


@pytest.fixture
def tile_index_gdf() -> gpd.GeoDataFrame:
    """Two blocks of four 3200 m tiles each, side by side, in the equal-area frame."""
    rows: list[dict[str, object]] = []
    tile = BLOCK_M / 2
    for bx, block_key in enumerate(("B-0000X-0000Y", "B-0001X-0000Y")):
        for tx in range(2):
            for ty in range(2):
                left = bx * BLOCK_M + tx * tile
                bottom = ty * tile
                rows.append(
                    {
                        "block_key": block_key,
                        "tile_key": f"T-{bx * 2 + tx:04d}X-{ty:04d}Y",
                        "geometry": box(left, bottom, left + tile, bottom + tile),
                    }
                )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WORLD_CYLINDRICAL_EQUAL_AREA)
