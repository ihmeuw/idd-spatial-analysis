"""Nearest-facility distances against geodesic brute force, and shares within thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Point

from idd_spatial_analysis.constants import EARTH_RADIUS_KM, WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import BlockGrid, bbox_in_crs, grid_from_bounds
from idd_spatial_analysis.zonal import nearest_facility_distance, share_within
from idd_spatial_analysis.zonal.distance import _chord_to_arc

if TYPE_CHECKING:
    from rasterio.windows import Window

GEOD = Geod(ellps="WGS84")
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0


def test_chord_to_arc_is_pinned() -> None:
    for arc in (1_000_000.0, 5_000_000.0):
        chord = 2 * EARTH_RADIUS_M * np.sin(arc / (2 * EARTH_RADIUS_M))
        assert _chord_to_arc(np.array([chord]))[0] == pytest.approx(arc, rel=1e-9)
    half_turn = np.pi * EARTH_RADIUS_M
    assert _chord_to_arc(np.array([2 * EARTH_RADIUS_M]))[0] == pytest.approx(half_turn, rel=1e-9)
    assert _chord_to_arc(np.array([3 * EARTH_RADIUS_M]))[0] == pytest.approx(half_turn, rel=1e-9)


def test_a_far_facility_needs_the_arc_correction(grid: BlockGrid, mask: np.ndarray) -> None:
    # About 500 km from the block: the chord is shorter than the arc by one part in 4000, so a
    # chord returned as is would miss the tolerance here by a factor of ten.
    far = gpd.GeoDataFrame(
        {"zone": np.array([1], dtype="int32")}, geometry=[Point(4.5, 0.03)], crs=WGS84
    )
    got = nearest_facility_distance(mask, far, grid, within_zone=False)
    np.testing.assert_allclose(got, _brute_force(grid, mask, far, None), rtol=2e-5)
    assert 450_000 < np.nanmin(got) < 550_000


def test_facility_zone_zero_is_refused(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    zeroed = facilities_gdf.assign(zone=np.array([0, 1, 2, 4, 4, 4], dtype="int32"))
    with pytest.raises(ValueError, match="no zone"):
        nearest_facility_distance(mask, zeroed, grid, within_zone=True, facility_zone_col="zone")


def _brute_force(
    grid: BlockGrid,
    mask: np.ndarray,
    facilities_gdf: gpd.GeoDataFrame,
    facility_zone: np.ndarray | None,
) -> np.ndarray:
    """Geodesic distance to the nearest (same-zone) facility, pixel by pixel, via pyproj.Geod."""
    lonlat = facilities_gdf.to_crs(WGS84)
    flon, flat = lonlat.geometry.x.to_numpy(), lonlat.geometry.y.to_numpy()
    xs, ys = grid.pixel_centers()
    to_lonlat = Transformer.from_crs(grid.crs, WGS84, always_xy=True)
    out = np.full(grid.shape, np.nan)
    for row in range(grid.height):
        for col in range(grid.width):
            zone = mask[row, col]
            if zone == 0:
                continue
            if facility_zone is None:
                candidates_lon, candidates_lat = flon, flat
            else:
                keep = facility_zone == zone
                candidates_lon, candidates_lat = flon[keep], flat[keep]
            if candidates_lon.size == 0:
                continue
            lon, lat = to_lonlat.transform(xs[col], ys[row])
            _, _, distance = GEOD.inv(
                np.full(candidates_lon.size, lon),
                np.full(candidates_lon.size, lat),
                candidates_lon,
                candidates_lat,
            )
            out[row, col] = distance.min()
    return out


def test_any_facility_matches_geodesic_brute_force(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    got = nearest_facility_distance(mask, facilities_gdf, grid, within_zone=False)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _brute_force(grid, mask, facilities_gdf, None), rtol=1e-5)


def test_distances_are_surface_metres_away_from_the_equator() -> None:
    # A 100 m block near 20E 50N. Planar distances in the equal-area frame would be off by
    # cos(latitude): 1.56 times too long east-west, 0.64 times too short north-south.
    bounds = bbox_in_crs((20.0, 50.0, 20.1, 50.05), WGS84, WORLD_CYLINDRICAL_EQUAL_AREA)
    grid = grid_from_bounds(bounds, 100.0, WORLD_CYLINDRICAL_EQUAL_AREA)
    mask = np.ones(grid.shape, dtype="int32")
    facility = gpd.GeoDataFrame(
        {"zone": np.array([1], dtype="int32")}, geometry=[Point(20.05, 50.02)], crs=WGS84
    )
    got = nearest_facility_distance(mask, facility, grid, within_zone=False)
    np.testing.assert_allclose(got, _brute_force(grid, mask, facility, None), rtol=1e-5)
    xs, ys = grid.pixel_centers()
    projected = facility.to_crs(grid.crs).geometry.iloc[0]
    planar = np.hypot(xs[np.newaxis, :] - projected.x, ys[:, np.newaxis] - projected.y)
    assert np.nanmax(np.abs(planar - got) / np.maximum(got, 1.0)) > 0.1


def test_a_lonlat_grid_is_accepted_and_agrees_with_brute_force(
    grid: BlockGrid, facilities_gdf: gpd.GeoDataFrame
) -> None:
    lonlat_grid = grid_from_bounds(grid.lonlat_bounds(), 0.0009, WGS84)
    mask = np.ones(lonlat_grid.shape, dtype="int32")
    got = nearest_facility_distance(mask, facilities_gdf, lonlat_grid, within_zone=False)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(
        got, _brute_force(lonlat_grid, mask, facilities_gdf, None), rtol=1e-5
    )


def test_within_zone_uses_the_facility_zone_column(
    grid: BlockGrid,
    mask_windows: tuple[np.ndarray, dict[int, Window]],
    facilities_gdf: gpd.GeoDataFrame,
) -> None:
    mask, windows = mask_windows
    # The first facility sits in the zone-1 quadrant but is labelled 2: the column, not the
    # mask pixel under the facility, decides which zone it serves.
    labels = np.array([2, 1, 2, 4, 4, 4], dtype="int32")
    labelled = facilities_gdf.assign(zone=labels)
    got = nearest_facility_distance(
        mask, labelled, grid, within_zone=True, facility_zone_col="zone", windows=windows
    )
    expect = _brute_force(grid, mask, labelled, labels)
    np.testing.assert_allclose(got, expect, rtol=1e-5)
    assert np.isnan(got[mask == 3]).all()
    assert np.isfinite(got[mask == 1]).all()
    under_mask = _brute_force(grid, mask, facilities_gdf, facilities_gdf["zone"].to_numpy())
    assert not np.array_equal(got, under_mask, equal_nan=True)
    without_windows = nearest_facility_distance(
        mask, labelled, grid, within_zone=True, facility_zone_col="zone"
    )
    np.testing.assert_array_equal(got, without_windows)
    nullable = labelled.assign(zone=pd.array(labels, dtype="Int32"))
    again = nearest_facility_distance(
        mask, nullable, grid, within_zone=True, facility_zone_col="zone"
    )
    np.testing.assert_array_equal(got, again)


def test_facilities_outside_the_grid_or_on_a_border_count_for_their_zone(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    beyond = gpd.GeoDataFrame(
        {"zone": np.array([3], dtype="int32")}, geometry=[Point(-1000.0, 1000.0)], crs=grid.crs
    )
    with_outside = gpd.GeoDataFrame(
        pd.concat([facilities_gdf[["zone", "geometry"]], beyond], ignore_index=True),
        crs=grid.crs,
    )
    got = nearest_facility_distance(
        mask, with_outside, grid, within_zone=True, facility_zone_col="zone"
    )
    assert np.isfinite(got[mask == 3]).all()
    expect = _brute_force(grid, mask, with_outside, with_outside["zone"].to_numpy())
    np.testing.assert_allclose(got, expect, rtol=1e-5)
    border = gpd.GeoDataFrame(
        {"zone": np.array([3], dtype="int32")}, geometry=[Point(3200.0, 3200.0)], crs=grid.crs
    )
    got = nearest_facility_distance(mask, border, grid, within_zone=True, facility_zone_col="zone")
    assert np.isfinite(got[mask == 3]).all()
    assert np.isnan(got[mask == 1]).all()


def test_zone_column_rules(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    with pytest.raises(ValueError, match="facility_zone_col"):
        nearest_facility_distance(mask, facilities_gdf, grid, within_zone=True)
    with pytest.raises(ValueError, match="only applies"):
        nearest_facility_distance(
            mask, facilities_gdf, grid, within_zone=False, facility_zone_col="zone"
        )
    with pytest.raises(KeyError, match="nope"):
        nearest_facility_distance(
            mask, facilities_gdf, grid, within_zone=True, facility_zone_col="nope"
        )
    unzoned = facilities_gdf.assign(zone=pd.array([1, None, 2, 4, 4, 4], dtype="Int32"))
    with pytest.raises(ValueError, match="no zone"):
        nearest_facility_distance(mask, unzoned, grid, within_zone=True, facility_zone_col="zone")
    floats = facilities_gdf.assign(zone=facilities_gdf["zone"].astype(float))
    with pytest.raises(TypeError, match="integers"):
        nearest_facility_distance(mask, floats, grid, within_zone=True, facility_zone_col="zone")


def test_bad_inputs(grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame) -> None:
    with pytest.raises(ValueError, match="CRS"):
        nearest_facility_distance(
            mask, facilities_gdf.set_crs(None, allow_override=True), grid, within_zone=False
        )
    with pytest.raises(ValueError, match="no facilities"):
        nearest_facility_distance(mask, facilities_gdf.iloc[:0], grid, within_zone=False)
    polygons = facilities_gdf.copy()
    polygons["geometry"] = polygons.buffer(10)
    with pytest.raises(TypeError, match="Point"):
        nearest_facility_distance(mask, polygons, grid, within_zone=False)
    with pytest.raises(ValueError, match="does not match"):
        nearest_facility_distance(mask[:10], facilities_gdf, grid, within_zone=False)
    with pytest.raises(TypeError, match="integer"):
        nearest_facility_distance(mask.astype(float), facilities_gdf, grid, within_zone=False)


def test_share_within_matches_manual_sums(
    grid: BlockGrid,
    mask_windows: tuple[np.ndarray, dict[int, Window]],
    facilities_gdf: gpd.GeoDataFrame,
    population: np.ndarray,
) -> None:
    mask, windows = mask_windows
    distance = nearest_facility_distance(
        mask, facilities_gdf, grid, within_zone=True, facility_zone_col="zone", windows=windows
    )
    table = share_within(distance, population, mask, [3000.0, 1000.0])
    assert list(table.columns) == ["zone", "threshold", "weight_within", "weight_total", "share"]
    assert table["threshold"].tolist() == [1000.0] * 4 + [3000.0] * 4
    weight = population.astype(np.float64)
    for zone in (1, 2, 3, 4):
        in_zone = (mask == zone) & np.isfinite(weight)
        total = weight[in_zone].sum()
        for threshold in (1000.0, 3000.0):
            row = table.loc[(table["zone"] == zone) & (table["threshold"] == threshold)].iloc[0]
            within = weight[in_zone & (distance <= threshold)].sum()
            assert row["weight_total"] == pytest.approx(total)
            assert row["weight_within"] == pytest.approx(within)
            assert row["share"] == pytest.approx(within / total)
    assert (table.loc[table["zone"] == 3, "weight_within"] == 0).all()
    assert (table.loc[table["zone"] == 3, "share"] == 0).all()
    assert (table.loc[table["zone"] == 1, "share"] > 0).all()


def test_share_within_bad_inputs_zero_weight_and_large_ids(
    mask: np.ndarray, population: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="empty"):
        share_within(population, population, mask, [])
    with pytest.raises(ValueError, match="share a shape"):
        share_within(population[:10], population, mask, [1.0])
    table = share_within(population, np.zeros_like(population), mask, [1.0])
    assert table["share"].isna().all()
    huge = mask.copy()
    huge[huge == 4] = 2_000_000_000
    table = share_within(population, population, huge, [1.0])
    assert table["zone"].tolist() == [1, 2, 3, 2_000_000_000]
