"""Cell areas of regular grids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from idd_spatial_analysis.constants import EARTH_RADIUS_KM
from idd_spatial_analysis.crs import is_geographic, is_projected_metres

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame import BlockGrid

_M2_PER_KM2 = 1e6
_MIN_EDGES = 2
_POLE_DEGREES = 90.0


def cell_area_km2(
    lat_edges: NDArray[Any], dlon_degrees: float, *, radius_km: float = EARTH_RADIUS_KM
) -> NDArray[np.float64]:
    """Exact spherical area of each row of a lat/lon grid, in km2.

    ``lat_edges`` are the row edges from top to bottom (length rows + 1, degrees); each row's
    area is ``R^2 * dlon * |sin(top) - sin(bottom)|`` and is the same for every column. Edges
    beyond the poles (a grid whose rows are centred on them) are clipped to +/-90, so such a
    row gets the area of the polar cap it covers.
    """
    latitudes = np.clip(np.asarray(lat_edges, dtype=np.float64), -_POLE_DEGREES, _POLE_DEGREES)
    edges = np.radians(latitudes)
    if edges.ndim != 1 or edges.size < _MIN_EDGES:
        msg = "lat_edges must be a 1-D array of at least two edges"
        raise ValueError(msg)
    if dlon_degrees <= 0:
        msg = f"dlon_degrees must be positive, got {dlon_degrees}"
        raise ValueError(msg)
    band = np.abs(np.sin(edges[:-1]) - np.sin(edges[1:]))
    areas = radius_km**2 * np.radians(dlon_degrees) * band
    return np.asarray(areas, dtype=np.float64)


def grid_cell_areas_km2(
    grid: BlockGrid, *, radius_km: float = EARTH_RADIUS_KM
) -> NDArray[np.float64]:
    """Area of each row's pixels in km2: exact spherical for lat/lon grids, constant for metric ones."""
    if is_geographic(grid.crs):
        rows = np.arange(grid.height + 1, dtype=np.float64)
        lat_edges = grid.transform.f + rows * grid.transform.e
        return cell_area_km2(lat_edges, grid.transform.a, radius_km=radius_km)
    if is_projected_metres(grid.crs):
        xres, yres = grid.resolution
        return np.full(grid.height, xres * yres / _M2_PER_KM2, dtype=np.float64)
    msg = f"cannot compute areas for CRS {grid.crs}: neither lat/lon nor metric"
    raise ValueError(msg)
