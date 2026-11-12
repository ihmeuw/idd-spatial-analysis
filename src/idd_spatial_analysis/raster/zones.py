"""Zone polygons to an integer id mask on a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box

from idd_spatial_analysis.constants import ZONE_DTYPE, ZONE_NODATA
from idd_spatial_analysis.crs import crs_equal, crs_string
from idd_spatial_analysis.frame.bounds import bbox_in_crs
from idd_spatial_analysis.geometry.antimeridian import check_lon_180

if TYPE_CHECKING:
    import geopandas as gpd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid

_PREFILTER_PAD = 0.01
"""Share of the grid box's size added on every side before the prefilter of ``_near_grid``."""


def _near_grid(zones: gpd.GeoDataFrame, grid: BlockGrid) -> gpd.GeoDataFrame:
    """The rows whose geometry may reach ``grid``, chosen in the zones' own CRS, order kept.

    A superset, so that only nearby polygons are reprojected; the exact test follows in the
    grid's CRS. The grid's box is drawn in the zones' CRS with densified edges and padded by
    ``_PREFILTER_PAD`` of its size, far more than the curvature a reprojected edge can hide
    between densification points. When the box cannot be drawn there (it wraps the
    antimeridian or leaves the CRS's valid area) every row is kept, which costs time only.
    """
    try:
        left, bottom, right, top = bbox_in_crs(grid.bounds, grid.crs, crs_string(zones.crs))
    except ValueError:
        return zones
    pad_x, pad_y = _PREFILTER_PAD * (right - left), _PREFILTER_PAD * (top - bottom)
    reach = box(left - pad_x, bottom - pad_y, right + pad_x, top + pad_y)
    return zones.iloc[np.sort(zones.sindex.query(reach, predicate="intersects"))]


def rasterize_zones(
    zones_gdf: gpd.GeoDataFrame,
    zone_col: str,
    grid: BlockGrid,
    *,
    all_touched: bool = False,
) -> tuple[NDArray[np.int32], dict[int, Window]]:
    """Zone id per pixel (``0`` where no zone) and, per zone that got pixels, its window.

    Ids must be positive, unique integers. Zones in another CRS are reprojected to the grid's;
    lon/lat zones that need reprojecting must be in the -180..180 convention, while zones
    already in the grid's CRS (a 0..360 table on a 0..360 grid, say) are burned as they are. A
    pixel belongs to the zone containing its centre (``all_touched=True``: any zone touching
    it); where zones overlap, the later row in ``zones_gdf`` wins, the same rule as
    ``geometry.points_to_zones``. A grid no zone reaches yields an all-zero mask and no
    windows; an empty ``zones_gdf`` is an error.

    Only the zones near the grid are reprojected, so calling this once per block with a table
    of the whole world costs what the block's own zones cost.
    """
    if zone_col not in zones_gdf.columns:
        msg = f"zones lack column {zone_col!r}"
        raise KeyError(msg)
    if zones_gdf.crs is None:
        msg = "zones carry no CRS"
        raise ValueError(msg)
    if zones_gdf.empty:
        msg = "zones is empty"
        raise ValueError(msg)
    ids = zones_gdf[zone_col]
    if not pd.api.types.is_integer_dtype(ids):
        msg = f"zone ids must be integers, got {ids.dtype}"
        raise TypeError(msg)
    if ids.duplicated().any():
        msg = f"duplicate zone ids: {sorted(ids[ids.duplicated()].unique().tolist())[:10]}"
        raise ValueError(msg)
    values = ids.to_numpy().astype(np.int64)
    if (values <= ZONE_NODATA).any() or values.max() > np.iinfo(ZONE_DTYPE).max:
        msg = f"zone ids must lie in 1..{np.iinfo(ZONE_DTYPE).max} (0 is nodata)"
        raise ValueError(msg)

    zones = zones_gdf[[zone_col, "geometry"]]
    if not crs_equal(zones.crs, grid.crs):
        check_lon_180(zones)  # reprojection is what wraps a 0..360 polygon around the world
        zones = _near_grid(zones, grid).to_crs(grid.crs)
    hit = zones.loc[zones.intersects(box(*grid.bounds)) & ~zones.geometry.is_empty]
    mask = np.full(grid.shape, ZONE_NODATA, dtype=ZONE_DTYPE)
    windows: dict[int, Window] = {}
    if hit.empty:
        return mask, windows
    shapes = [(geom, int(zone)) for geom, zone in zip(hit.geometry, hit[zone_col], strict=True)]
    rasterize(shapes, out=mask, transform=grid.transform, all_touched=all_touched)
    present = np.unique(mask)
    bounds_by_zone = dict(
        zip(
            (int(zone) for zone in hit[zone_col]),
            hit.geometry.bounds.itertuples(index=False, name=None),
            strict=True,
        )
    )
    for zone in present[present != ZONE_NODATA]:
        windows[int(zone)] = grid.window(bounds_by_zone[int(zone)])
    return mask, windows
