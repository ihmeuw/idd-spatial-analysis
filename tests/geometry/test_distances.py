"""Distances along the Earth."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Geod

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.geometry import (
    chord_to_arc,
    geocentric_points,
    nearest_surface_distance,
    surface_distance,
    to_lonlat,
)
from idd_spatial_analysis.geometry.distances import EARTH_RADIUS_M

ONE_DEGREE_AT_EQUATOR_M = 111_319.4908  # WGS84 semi-major axis times pi / 180


def test_surface_distance_is_geodesic_metres() -> None:
    assert surface_distance(0.0, 0.0, 1.0, 0.0, WGS84)[()] == pytest.approx(
        ONE_DEGREE_AT_EQUATOR_M, rel=1e-9
    )
    assert surface_distance(10.0, 50.0, 10.0, 50.0, WGS84)[()] == 0.0
    many = surface_distance(0.0, 0.0, np.array([1.0, 2.0, 3.0]), np.zeros(3), WGS84)
    assert many.shape == (3,)
    np.testing.assert_allclose(many, ONE_DEGREE_AT_EQUATOR_M * np.array([1, 2, 3]), rtol=1e-9)


def test_surface_distance_accepts_projected_inputs_and_mixed_crs() -> None:
    # One degree east of Greenwich on the equator, in the equal-area frame (x = a * lon).
    projected = surface_distance(
        0.0, 0.0, ONE_DEGREE_AT_EQUATOR_M, 0.0, WORLD_CYLINDRICAL_EQUAL_AREA
    )
    assert projected[()] == pytest.approx(ONE_DEGREE_AT_EQUATOR_M, rel=1e-6)
    mixed = surface_distance(0.0, 0.0, 1.0, 0.0, WORLD_CYLINDRICAL_EQUAL_AREA, other_crs=WGS84)
    assert mixed[()] == pytest.approx(ONE_DEGREE_AT_EQUATOR_M, rel=1e-6)
    # Planar distance in the equal-area frame is wrong away from the equator; the geodesic is not.
    lon1, lat1, lon2, lat2 = 20.0, 60.0, 20.1, 60.0
    expect = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)[2]
    assert surface_distance(lon1, lat1, lon2, lat2, WGS84)[()] == pytest.approx(expect, rel=1e-9)


def test_surface_distance_bad_inputs() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        surface_distance(np.zeros(2), np.zeros(3), 0.0, 0.0, WGS84)
    with pytest.raises(ValueError, match="finite"):
        surface_distance(np.nan, 0.0, 0.0, 0.0, WGS84)


def test_to_lonlat_identity_and_projection() -> None:
    lon, lat = to_lonlat(np.array([5.0]), np.array([6.0]), "EPSG:4326")
    assert lon.tolist() == [5.0]
    assert lat.tolist() == [6.0]
    lon, lat = to_lonlat(np.array([0.0]), np.array([0.0]), WORLD_CYLINDRICAL_EQUAL_AREA)
    assert lon[0] == pytest.approx(0.0, abs=1e-9)
    assert lat[0] == pytest.approx(0.0, abs=1e-9)


def test_geocentric_points_on_the_axes() -> None:
    points = geocentric_points(np.array([0.0, 90.0, 0.0]), np.array([0.0, 0.0, 90.0]), WGS84)
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[0], [6_378_137.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(points[1], [0.0, 6_378_137.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(points[2], [0.0, 0.0, 6_356_752.3142], atol=1e-3)
    with pytest.raises(ValueError, match="finite"):
        geocentric_points(np.array([np.inf]), np.array([0.0]), WGS84)


def test_chord_to_arc_is_pinned() -> None:
    for arc in (1_000_000.0, 5_000_000.0):
        chord = 2 * EARTH_RADIUS_M * np.sin(arc / (2 * EARTH_RADIUS_M))
        assert chord_to_arc(np.array([chord]))[0] == pytest.approx(arc, rel=1e-9)
    half_turn = np.pi * EARTH_RADIUS_M
    assert chord_to_arc(np.array([2 * EARTH_RADIUS_M]))[0] == pytest.approx(half_turn, rel=1e-9)
    assert chord_to_arc(np.array([3 * EARTH_RADIUS_M]))[0] == pytest.approx(half_turn, rel=1e-9)


def test_nearest_surface_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    query_lon, query_lat = rng.uniform(60, 80, 50), rng.uniform(20, 40, 50)
    target_lon, target_lat = rng.uniform(55, 85, 12), rng.uniform(15, 45, 12)
    distance, index = nearest_surface_distance(
        query_lon, query_lat, WGS84, target_lon, target_lat, WGS84
    )
    geod = Geod(ellps="WGS84")
    for i in range(50):
        _, _, metres = geod.inv(
            np.full(12, query_lon[i]), np.full(12, query_lat[i]), target_lon, target_lat
        )
        assert index[i] == int(np.argmin(metres))
        assert distance[i] == pytest.approx(metres.min(), rel=2e-5)
    with pytest.raises(ValueError, match="no target"):
        nearest_surface_distance(query_lon, query_lat, WGS84, np.array([]), np.array([]), WGS84)
