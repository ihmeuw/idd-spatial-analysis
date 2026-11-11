"""Distances along the Earth: what "actual distance" means in this package.

Every distance the package reports is metres along the WGS84 ellipsoid (``surface_distance``)
or, for nearest-point searches, the Earth-centred chord turned into an arc
(``nearest_surface_distance``), which agrees with the geodesic to a few parts in a million at
the scales a block loop meets. Planar distances in a projected CRS are never used: in the
equal-area block frame they are off by the cosine of the latitude, 15 percent at 30 degrees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree

from idd_spatial_analysis.constants import EARTH_RADIUS_KM, GEOCENTRIC_CRS, WGS84
from idd_spatial_analysis.crs import crs_equal

if TYPE_CHECKING:
    from numpy.typing import NDArray

EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0
_GEOD = Geod(ellps="WGS84")
_POINT_DIMENSIONS = 3


def _finite_pair(
    xs: NDArray[Any], ys: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.shape != y.shape:
        msg = f"x and y must share a shape, got {x.shape} and {y.shape}"
        raise ValueError(msg)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        msg = "coordinates must be finite"
        raise ValueError(msg)
    return x, y


def to_lonlat(
    xs: NDArray[Any], ys: NDArray[Any], crs: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Longitude and latitude (degrees) of points given in ``crs``."""
    x, y = _finite_pair(xs, ys)
    if crs_equal(crs, WGS84):
        return x, y
    lon, lat = Transformer.from_crs(crs, WGS84, always_xy=True).transform(x, y)
    lon_arr, lat_arr = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if not (np.isfinite(lon_arr).all() and np.isfinite(lat_arr).all()):
        msg = f"some points have no position on the Earth (outside the valid area of {crs})"
        raise ValueError(msg)
    return lon_arr, lat_arr


def surface_distance(  # noqa: PLR0913  (two points, each with a CRS)
    x1: NDArray[Any],
    y1: NDArray[Any],
    x2: NDArray[Any],
    y2: NDArray[Any],
    crs: Any,
    *,
    other_crs: Any = None,
) -> NDArray[np.float64]:
    """Metres along the WGS84 ellipsoid between paired points, element by element.

    ``(x1, y1)`` are in ``crs``; ``(x2, y2)`` too unless ``other_crs`` is given. Inputs
    broadcast as numpy arrays do (one point against many, say).
    """
    lon1, lat1 = to_lonlat(x1, y1, crs)
    lon2, lat2 = to_lonlat(x2, y2, crs if other_crs is None else other_crs)
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(lon1, lat1, lon2, lat2)
    _, _, metres = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(metres, dtype=np.float64)


def geocentric_points(xs: NDArray[Any], ys: NDArray[Any], crs: Any) -> NDArray[np.float64]:
    """``(n, 3)`` Earth-centred Cartesian metres (EPSG:4978) of points on the ellipsoid."""
    x, y = _finite_pair(np.ravel(xs), np.ravel(ys))
    transformer = Transformer.from_crs(crs, GEOCENTRIC_CRS, always_xy=True)
    gx, gy, gz = transformer.transform(x, y, np.zeros_like(x))
    points = np.column_stack([gx, gy, gz]).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != _POINT_DIMENSIONS:  # noqa: PLR2004
        msg = f"expected (n, 3) geocentric points, got shape {points.shape}"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        bad = int((~np.isfinite(points).all(axis=1)).sum())
        msg = f"{bad} points have no position on the Earth (outside the valid area of {crs})"
        raise ValueError(msg)
    return points


def chord_to_arc(chord: NDArray[Any]) -> NDArray[np.float64]:
    """Surface distance for a straight-line distance between two points on the sphere."""
    ratio = np.clip(np.asarray(chord, dtype=np.float64) / (2.0 * EARTH_RADIUS_M), 0.0, 1.0)
    return np.asarray(2.0 * EARTH_RADIUS_M * np.arcsin(ratio), dtype=np.float64)


def nearest_surface_distance(  # noqa: PLR0913, PLR0917  (two point sets, each with a CRS)
    xs: NDArray[Any],
    ys: NDArray[Any],
    crs: Any,
    target_x: NDArray[Any],
    target_y: NDArray[Any],
    target_crs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """For each query point, metres along the Earth to its nearest target and that target's index.

    Both sets may be in any CRS. The search runs on Earth-centred coordinates (a KD-tree on
    chords), so the nearest target by chord is the nearest by surface distance too.
    """
    queries = geocentric_points(xs, ys, crs)
    targets = geocentric_points(target_x, target_y, target_crs)
    if targets.shape[0] == 0:
        msg = "no target points"
        raise ValueError(msg)
    chord, index = cKDTree(targets).query(queries)
    return chord_to_arc(chord), np.asarray(index, dtype=np.intp)
