"""A tiny population-model raking layout written to tmp_path (the LBD release is in tests/conftest.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from idd_spatial_analysis.constants import WGS84

if TYPE_CHECKING:
    from pathlib import Path


def write_population(root: Path, flags: dict[int, list[int]], years: list[int]) -> None:
    """``population_gbd_2023.parquet`` with ``flags[year] = most_detailed`` per location in order."""
    rows = []
    for year in years:
        for location_id, flag in zip([100, 101, 102, 103], flags[year], strict=True):
            rows.append({"location_id": location_id, "year_id": year, "most_detailed": flag})
    pd.DataFrame(rows).to_parquet(root / "population_gbd_2023.parquet")


@pytest.fixture
def raking_root(tmp_path: Path) -> Path:
    """gbd_2023 shapes with a GeoParquet covering column, lsae_1285 shapes without one.

    Locations: 100 (the whole 10 x 10 box), 101 (its west half), 102 (its east half) and 103, a
    triangle whose bounding box reaches where the triangle does not, so a bbox pre-filter and
    the exact intersects test can be told apart.
    """
    root = tmp_path / "raking"
    (root / "gbd-inputs").mkdir(parents=True)
    shapes = gpd.GeoDataFrame(
        {"location_id": [100, 101, 102, 103]},
        geometry=[
            box(0, 0, 10, 10),
            box(0, 0, 5, 10),
            box(5, 0, 10, 10),
            Polygon([(5, 0), (10, 0), (10, 5)]),
        ],
        crs=WGS84,
    )
    shapes.to_parquet(root / "shapes_gbd_2023.parquet", write_covering_bbox=True)
    write_population(root, {2020: [0, 1, 1, 1], 2023: [0, 1, 1, 1]}, [2020, 2023])
    shapes.iloc[1:].to_parquet(root / "gbd-inputs" / "shapes_lsae_1285_a2.parquet")
    hierarchy = pd.DataFrame(
        {
            "location_id": [100, 101, 102, 103],
            "parent_id": [0, 100, 100, 100],
            "level": [0, 1, 1, 1],
            "extra": "x",
        }
    )
    hierarchy.to_parquet(root / "gbd-inputs" / "hierarchy_gbd_2023.parquet")
    return root
