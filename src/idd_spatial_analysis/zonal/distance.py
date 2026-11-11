"""Nearest-facility distances on a zone mask, and the share of weight within thresholds.

Distances are metres along the Earth's surface whatever the grid's CRS (see
``geometry.distances``): pixel centres and facilities are compared in Earth-centred
coordinates and the chord is turned into an arc. Planar distances in the equal-area block
frame would be off by the cosine of the latitude away from the equator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from idd_spatial_analysis.constants import (
    SHARE_COLUMN,
    THRESHOLD_COLUMN,
    WEIGHT_TOTAL_COLUMN,
    WEIGHT_WITHIN_COLUMN,
    ZONE_COLUMN,
    ZONE_NODATA,
)
from idd_spatial_analysis.geometry.distances import chord_to_arc, geocentric_points
from idd_spatial_analysis.zonal.sums import check_zone_mask, zone_index

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import geopandas as gpd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid


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
    return geocentric_points(
        facilities_gdf.geometry.x.to_numpy(dtype=np.float64),
        facilities_gdf.geometry.y.to_numpy(dtype=np.float64),
        facilities_gdf.crs,
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
    values = np.asarray(zones.to_numpy(dtype=np.int64))
    if (values <= ZONE_NODATA).any():
        msg = (
            f"{int((values <= ZONE_NODATA).sum())} facilities have a zone of {ZONE_NODATA} or "
            f"less in {column!r}; zone ids are positive and {ZONE_NODATA} means no zone"
        )
        raise ValueError(msg)
    return values


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
    tree: cKDTree, grid: BlockGrid, rows: NDArray[np.intp], cols: NDArray[np.intp]
) -> NDArray[np.float64]:
    xs, ys = grid.pixel_centers()
    chord, _ = tree.query(geocentric_points(xs[cols], ys[rows], grid.crs))
    return chord_to_arc(chord)


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
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    if not within_zone:
        rows, cols = np.nonzero(zone != ZONE_NODATA)
        if rows.size:
            out[rows, cols] = _nearest(cKDTree(facilities), grid, rows, cols)
        return out
    facility_zone = _facility_zones(facilities_gdf, str(facility_zone_col))
    present = np.unique(zone[zone != ZONE_NODATA])
    for value in present:
        chosen = facility_zone == value
        if not chosen.any():
            continue  # no facility on this zone: its pixels stay NaN by design
        rows, cols = _zone_pixels(zone, int(value), windows)
        out[rows, cols] = _nearest(cKDTree(facilities[chosen]), grid, rows, cols)
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
