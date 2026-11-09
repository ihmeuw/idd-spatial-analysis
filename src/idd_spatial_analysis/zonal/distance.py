"""Nearest-facility distances on a zone mask, and the share of weight within thresholds.

Distances are metres along the Earth's surface whatever the grid's CRS: pixel centres and
facilities are converted to Earth-centred Cartesian coordinates (EPSG:4978), the nearest
facility is found in that space (a chord through the Earth) and the chord is turned into an
arc. Planar distances in the equal-area block frame would be off by the cosine of the
latitude away from the equator: 15 percent at 30 degrees, a factor of two at 60.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from idd_spatial_analysis.constants import (
    EARTH_RADIUS_KM,
    GEOCENTRIC_CRS,
    SHARE_COLUMN,
    THRESHOLD_COLUMN,
    WEIGHT_TOTAL_COLUMN,
    WEIGHT_WITHIN_COLUMN,
    ZONE_COLUMN,
    ZONE_NODATA,
)
from idd_spatial_analysis.zonal.sums import check_zone_mask, zone_index

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import geopandas as gpd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid

_EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0
_POINT_DIMENSIONS = 3


def _geocentric(
    transformer: Transformer, xs: NDArray[Any], ys: NDArray[Any]
) -> NDArray[np.float64]:
    """``(n, 3)`` Earth-centred metres of points on the ellipsoid, through ``transformer``."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    gx, gy, gz = transformer.transform(x, y, np.zeros_like(x))
    points = np.column_stack([gx, gy, gz]).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != _POINT_DIMENSIONS:  # noqa: PLR2004
        msg = f"expected (n, 3) geocentric points, got shape {points.shape}"
        raise ValueError(msg)
    if not np.isfinite(points).all():
        bad = int((~np.isfinite(points).all(axis=1)).sum())
        msg = f"{bad} points have no position on the Earth (outside the CRS's valid area)"
        raise ValueError(msg)
    return points


def _chord_to_arc(chord: NDArray[Any]) -> NDArray[np.float64]:
    """Surface distance for a straight-line distance between two points on the sphere."""
    ratio = np.clip(np.asarray(chord, dtype=np.float64) / (2.0 * _EARTH_RADIUS_M), 0.0, 1.0)
    return np.asarray(2.0 * _EARTH_RADIUS_M * np.arcsin(ratio), dtype=np.float64)


def _facility_points(facilities_gdf: gpd.GeoDataFrame) -> NDArray[np.float64]:
    if facilities_gdf.crs is None:
        msg = "facilities carry no CRS"
        raise ValueError(msg)
    if facilities_gdf.empty:
        msg = "no facilities"
        raise ValueError(msg)
    if not (facilities_gdf.geometry.geom_type == "Point").all():
        msg = "facilities must be Point geometries"
        raise TypeError(msg)
    to_earth = Transformer.from_crs(facilities_gdf.crs, GEOCENTRIC_CRS, always_xy=True)
    return _geocentric(
        to_earth,
        facilities_gdf.geometry.x.to_numpy(dtype=np.float64),
        facilities_gdf.geometry.y.to_numpy(dtype=np.float64),
    )


def _facility_zones(facilities_gdf: gpd.GeoDataFrame, column: str) -> NDArray[np.int64]:
    if column not in facilities_gdf.columns:
        msg = f"facilities lack column {column!r}"
        raise KeyError(msg)
    zones = facilities_gdf[column]
    if zones.isna().any():
        msg = (
            f"{int(zones.isna().sum())} facilities have no zone in {column!r}; assign one "
            "(geometry.points_to_zones) or drop them"
        )
        raise ValueError(msg)
    if not pd.api.types.is_integer_dtype(zones.dtype):
        msg = f"facility zones must be integers, got {zones.dtype}"
        raise TypeError(msg)
    return np.asarray(zones.to_numpy(dtype=np.int64))


def _zone_pixels(
    zone: NDArray[Any], value: int, windows: Mapping[int, Window] | None
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if windows is not None and value in windows:
        window = windows[value]
        row0, col0 = int(window.row_off), int(window.col_off)
        sub = zone[row0 : row0 + int(window.height), col0 : col0 + int(window.width)]
        rows, cols = np.nonzero(sub == value)
        return rows + row0, cols + col0
    rows, cols = np.nonzero(zone == value)
    return rows, cols


def _nearest(
    tree: cKDTree,
    to_earth: Transformer,
    grid: BlockGrid,
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
) -> NDArray[np.float64]:
    xs, ys = grid.pixel_centers()
    chord, _ = tree.query(_geocentric(to_earth, xs[cols], ys[rows]))
    return _chord_to_arc(chord)


def nearest_facility_distance(  # noqa: PLR0913  (keyword-only options)
    mask: NDArray[Any],
    facilities_gdf: gpd.GeoDataFrame,
    grid: BlockGrid,
    *,
    within_zone: bool,
    facility_zone_col: str | None = None,
    windows: Mapping[int, Window] | None = None,
) -> NDArray[np.float32]:
    """Metres along the Earth from each zoned pixel's centre to the nearest facility.

    Any grid CRS works: pixel centres and facilities are compared in Earth-centred coordinates,
    so the result is surface distance, never grid units. Pixels with zone ``0`` are NaN.

    ``within_zone=True`` counts only facilities whose zone equals the pixel's mask value. The
    facility's zone is read from ``facility_zone_col``, an integer column such as
    ``geometry.points_to_zones`` produces from the same polygons the mask came from; the
    facilities may lie anywhere, including outside the grid (a zone that straddles blocks
    keeps all its facilities), and a facility without a zone is an error. A zone with no
    facility is NaN throughout. ``windows`` (from ``rasterize_zones``) bounds each zone's
    pixel search.
    """
    zone = np.asarray(mask)
    check_zone_mask(zone)
    if zone.shape != grid.shape:
        msg = f"mask shape {zone.shape} does not match the grid {grid.shape}"
        raise ValueError(msg)
    if within_zone and facility_zone_col is None:
        msg = "within_zone=True needs facility_zone_col, the facilities' zone column"
        raise ValueError(msg)
    if not within_zone and facility_zone_col is not None:
        msg = "facility_zone_col only applies with within_zone=True"
        raise ValueError(msg)
    facilities = _facility_points(facilities_gdf)
    to_earth = Transformer.from_crs(grid.crs, GEOCENTRIC_CRS, always_xy=True)
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    if not within_zone:
        rows, cols = np.nonzero(zone != ZONE_NODATA)
        if rows.size:
            out[rows, cols] = _nearest(cKDTree(facilities), to_earth, grid, rows, cols)
        return out
    facility_zone = _facility_zones(facilities_gdf, str(facility_zone_col))
    present = np.unique(zone[zone != ZONE_NODATA])
    for value in present:
        chosen = facility_zone == value
        if not chosen.any():
            continue  # no facility on this zone: its pixels stay NaN by design
        rows, cols = _zone_pixels(zone, int(value), windows)
        tree = cKDTree(facilities[chosen])
        out[rows, cols] = _nearest(tree, to_earth, grid, rows, cols)
    return out


def share_within(
    distance: NDArray[Any],
    weight: NDArray[Any],
    mask: NDArray[Any],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Per zone and threshold: the share of ``weight`` at pixels with ``distance <= threshold``.

    Pixels with a NaN weight never count; a NaN distance counts in the total but never within
    (nobody there is within reach). ``share`` is NaN for a zone whose total weight is zero.
    """
    d = np.asarray(distance, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    zone = np.asarray(mask)
    check_zone_mask(zone)
    if not d.shape == w.shape == zone.shape:
        msg = f"distance {d.shape}, weight {w.shape} and mask {zone.shape} must share a shape"
        raise ValueError(msg)
    cutoffs = sorted({float(t) for t in thresholds})
    if not cutoffs:
        msg = "thresholds is empty"
        raise ValueError(msg)
    present, index = zone_index(zone)
    w_flat = w.ravel()
    d_flat = d.ravel()
    valid = (zone.ravel() != ZONE_NODATA) & np.isfinite(w_flat)
    total = np.bincount(index[valid], weights=w_flat[valid], minlength=present.size)
    frames = []
    for cutoff in cutoffs:
        within = valid & (d_flat <= cutoff)
        reached = np.bincount(index[within], weights=w_flat[within], minlength=present.size)
        frames.append(
            pd.DataFrame(
                {
                    ZONE_COLUMN: present,
                    THRESHOLD_COLUMN: np.full(present.size, cutoff, dtype=np.float64),
                    WEIGHT_WITHIN_COLUMN: reached,
                    WEIGHT_TOTAL_COLUMN: total,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out[SHARE_COLUMN] = out[WEIGHT_WITHIN_COLUMN] / out[WEIGHT_TOTAL_COLUMN].where(
        out[WEIGHT_TOTAL_COLUMN] > 0, np.nan
    )
    return out
