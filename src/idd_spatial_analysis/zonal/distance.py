"""Nearest-facility distances on a zone mask, and the share of weight within thresholds."""

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
    ZONE_DTYPE,
    ZONE_NODATA,
)
from idd_spatial_analysis.crs import crs_equal, is_geographic

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import geopandas as gpd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid


def _facility_xy(
    facilities_gdf: gpd.GeoDataFrame, crs: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if facilities_gdf.crs is None:
        msg = "facilities carry no CRS"
        raise ValueError(msg)
    if facilities_gdf.empty:
        msg = "no facilities"
        raise ValueError(msg)
    if not (facilities_gdf.geometry.geom_type == "Point").all():
        msg = "facilities must be Point geometries"
        raise TypeError(msg)
    points = facilities_gdf.geometry
    if not crs_equal(facilities_gdf.crs, crs):
        points = points.to_crs(crs)
    return points.x.to_numpy(dtype=np.float64), points.y.to_numpy(dtype=np.float64)


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


def nearest_facility_distance(
    mask: NDArray[Any],
    facilities_gdf: gpd.GeoDataFrame,
    grid: BlockGrid,
    *,
    within_zone: bool,
    windows: Mapping[int, Window] | None = None,
) -> NDArray[np.float32]:
    """Distance from each zoned pixel's centre to the nearest facility, in the grid's units.

    The grid must be projected (metres); facilities are reprojected to it. With
    ``within_zone=True`` only facilities on the same zone of ``mask`` count (a facility's zone
    is the mask value under it), and a zone without a facility is NaN throughout. Pixels with
    zone ``0`` are NaN. ``windows`` (from ``rasterize_zones``) bounds each zone's search.
    """
    if is_geographic(grid.crs):
        msg = f"distances need a projected grid, got {grid.crs}"
        raise ValueError(msg)
    zone = np.asarray(mask)
    if zone.shape != grid.shape:
        msg = f"mask shape {zone.shape} does not match the grid {grid.shape}"
        raise ValueError(msg)
    fx, fy = _facility_xy(facilities_gdf, grid.crs)
    xs, ys = grid.pixel_centers()
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    if not within_zone:
        rows, cols = np.nonzero(zone != ZONE_NODATA)
        tree = cKDTree(np.column_stack([fx, fy]))
        distance, _ = tree.query(np.column_stack([xs[cols], ys[rows]]))
        out[rows, cols] = distance
        return out
    frow, fcol = grid.rowcol(fx, fy)
    inside = grid.contains(frow, fcol)
    facility_zone = np.full(fx.size, ZONE_NODATA, dtype=ZONE_DTYPE)
    facility_zone[inside] = zone[frow[inside], fcol[inside]]
    present = np.unique(zone[zone != ZONE_NODATA])
    for value in present:
        chosen = facility_zone == value
        if not chosen.any():
            continue  # no facility on this zone: its pixels stay NaN by design
        rows, cols = _zone_pixels(zone, int(value), windows)
        tree = cKDTree(np.column_stack([fx[chosen], fy[chosen]]))
        distance, _ = tree.query(np.column_stack([xs[cols], ys[rows]]))
        out[rows, cols] = distance
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
    if not d.shape == w.shape == zone.shape:
        msg = f"distance {d.shape}, weight {w.shape} and mask {zone.shape} must share a shape"
        raise ValueError(msg)
    cutoffs = sorted({float(t) for t in thresholds})
    if not cutoffs:
        msg = "thresholds is empty"
        raise ValueError(msg)
    present = np.unique(zone[zone != ZONE_NODATA]).astype(ZONE_DTYPE)
    n_bins = int(zone.max()) + 1 if present.size else 1
    valid = (zone != ZONE_NODATA) & np.isfinite(w)
    total = np.bincount(zone[valid], weights=w[valid], minlength=n_bins)
    frames = []
    for cutoff in cutoffs:
        within = valid & (d <= cutoff)
        reached = np.bincount(zone[within], weights=w[within], minlength=n_bins)
        frames.append(
            pd.DataFrame(
                {
                    ZONE_COLUMN: present,
                    THRESHOLD_COLUMN: np.full(present.size, cutoff, dtype=np.float64),
                    WEIGHT_WITHIN_COLUMN: reached[present],
                    WEIGHT_TOTAL_COLUMN: total[present],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out[SHARE_COLUMN] = out[WEIGHT_WITHIN_COLUMN] / out[WEIGHT_TOTAL_COLUMN].where(
        out[WEIGHT_TOTAL_COLUMN] > 0, np.nan
    )
    return out
