"""A tiny population-model raking layout written to tmp_path (the LBD release is in tests/conftest.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from idd_spatial_analysis.constants import WGS84

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def raking_root(tmp_path: Path) -> Path:
    root = tmp_path / "raking"
    (root / "gbd-inputs").mkdir(parents=True)
    shapes = gpd.GeoDataFrame(
        {"location_id": [100, 101, 102]},
        geometry=[box(0, 0, 10, 10), box(0, 0, 5, 10), box(5, 0, 10, 10)],
        crs=WGS84,
    )
    shapes.to_parquet(root / "shapes_gbd_2023.parquet")
    population = pd.DataFrame(
        {
            "location_id": [100, 101, 102, 100, 101, 102],
            "year_id": [2020, 2020, 2020, 2023, 2023, 2023],
            "most_detailed": [0, 1, 1, 0, 1, 1],
        }
    )
    population.to_parquet(root / "population_gbd_2023.parquet")
    shapes.iloc[1:].to_parquet(root / "gbd-inputs" / "shapes_lsae_1285_a2.parquet")
    hierarchy = pd.DataFrame(
        {
            "location_id": [100, 101, 102],
            "parent_id": [0, 100, 100],
            "level": [0, 1, 1],
            "extra": "x",
        }
    )
    hierarchy.to_parquet(root / "gbd-inputs" / "hierarchy_gbd_2023.parquet")
    return root
