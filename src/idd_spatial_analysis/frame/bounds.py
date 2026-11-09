"""Bounds across coordinate reference systems."""

from __future__ import annotations

import math

from rasterio.warp import transform_bounds

from idd_spatial_analysis.crs import crs_equal, is_geographic

Bounds = tuple[float, float, float, float]
"""(left, bottom, right, top) in one CRS's units."""

_HALF_WORLD_DEGREES = 180.0


class AntimeridianError(ValueError):
    """A reprojected box would wrap the antimeridian; split the geometry first."""


def pad_bounds(bounds: Bounds, pad: float) -> Bounds:
    """``bounds`` grown by ``pad`` on every side (CRS units)."""
    left, bottom, right, top = bounds
    return (left - pad, bottom - pad, right + pad, top + pad)


def bbox_in_crs(bounds: Bounds, src_crs: str, dst_crs: str, *, densify_points: int = 21) -> Bounds:
    """The box in ``dst_crs`` that covers ``bounds`` given in ``src_crs``.

    Edges are densified before reprojection. A result that is not finite means the bounds fall
    outside the valid area of ``dst_crs``. A geographic result whose left edge is not west of
    its right edge (how GDAL reports a crossing) or that is wider than 180 degrees means the box
    wraps the antimeridian, and ``AntimeridianError`` is raised instead of returning a box that
    is inside out or spans the world.
    """
    left, bottom, right, top = bounds
    if left >= right or bottom >= top:
        msg = f"degenerate bounds {bounds}"
        raise ValueError(msg)
    if crs_equal(src_crs, dst_crs):
        return bounds
    out = transform_bounds(src_crs, dst_crs, left, bottom, right, top, densify_pts=densify_points)
    result = (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    if not all(math.isfinite(value) for value in result):
        msg = f"bounds {bounds} fall outside the valid area of {dst_crs}"
        raise ValueError(msg)
    if is_geographic(dst_crs) and (
        result[0] >= result[2] or result[2] - result[0] > _HALF_WORLD_DEGREES
    ):
        msg = f"bounds {bounds} wrap the antimeridian in {dst_crs}: {result}"
        raise AntimeridianError(msg)
    return result
