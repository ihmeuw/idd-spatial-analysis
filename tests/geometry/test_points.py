"""Points to zones."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.geometry import points_to_zones


def test_points_get_their_zone_and_outside_is_na(zones_gdf: gpd.GeoDataFrame) -> None:
    pts = gpd.GeoDataFrame(
        geometry=[Point(100, 100), Point(5000, 5000), Point(10_000, 10_000)],
        crs=zones_gdf.crs,
        index=[3, 1, 2],
    )
    zones = points_to_zones(pts, zones_gdf, "zone")
    assert list(zones.index) == [3, 1, 2]
    assert str(zones.dtype) == "Int32"
    assert zones.loc[3] == 3
    assert zones.loc[1] == 2
    assert pd.isna(zones.loc[2])


def test_boundary_points_need_intersects_and_take_the_last_listed_zone(
    zones_gdf: gpd.GeoDataFrame,
) -> None:
    # (3200, 100) lies on the edge shared by zone 3 (listed third) and zone 4 (listed fourth).
    on_edge = gpd.GeoDataFrame(geometry=[Point(3200, 100)], crs=zones_gdf.crs)
    assert pd.isna(points_to_zones(on_edge, zones_gdf, "zone").iloc[0])
    assert points_to_zones(on_edge, zones_gdf, "zone", predicate="intersects").iloc[0] == 4
    # Reversed row order with the original index kept apart: position, not index label, wins.
    reversed_zones = zones_gdf.iloc[::-1]
    assert list(reversed_zones.index) == [4, 3, 2, 1, 0]
    assert points_to_zones(on_edge, reversed_zones, "zone", predicate="intersects").iloc[0] == 3


def test_points_are_reprojected(zones_gdf: gpd.GeoDataFrame) -> None:
    pts = gpd.GeoDataFrame(geometry=[Point(100, 100)], crs=zones_gdf.crs).to_crs(WGS84)
    assert points_to_zones(pts, zones_gdf, "zone").iloc[0] == 3


def test_bad_inputs_raise(zones_gdf: gpd.GeoDataFrame) -> None:
    pts = gpd.GeoDataFrame(geometry=[Point(100, 100)], crs=zones_gdf.crs)
    with pytest.raises(KeyError, match="nope"):
        points_to_zones(pts, zones_gdf, "nope")
    with pytest.raises(ValueError, match="CRS"):
        points_to_zones(gpd.GeoDataFrame(geometry=[Point(0, 0)]), zones_gdf, "zone")
    dup = gpd.GeoDataFrame(
        geometry=[Point(100, 100), Point(200, 200)], crs=zones_gdf.crs, index=[1, 1]
    )
    with pytest.raises(ValueError, match="unique"):
        points_to_zones(dup, zones_gdf, "zone")
    with pytest.raises(TypeError, match="integers"):
        points_to_zones(pts, zones_gdf.assign(zone=zones_gdf["zone"].astype(float)), "zone")
