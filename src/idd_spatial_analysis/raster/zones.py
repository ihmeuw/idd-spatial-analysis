"""Zone polygons to an integer id mask on a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box

from idd_spatial_analysis.constants import ZONE_DTYPE, ZONE_NODATA
from idd_spatial_analysis.crs import crs_equal
from idd_spatial_analysis.geometry.antimeridian import check_lon_180

if TYPE_CHECKING:
    import geopandas as gpd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid


def rasterize_zones(
    zones_gdf: gpd.GeoDataFrame,
    zone_col: str,
    grid: BlockGrid,
    *,
    all_touched: bool = False,
) -> tuple[NDArray[np.int32], dict[int, Window]]:
    """Zone id per pixel (``0`` where no zone) and, per zone that got pixels, its window.

    Ids must be positive, unique integers. Zones are reprojected to the grid's CRS (lon/lat
    zones must be in the -180..180 convention). A pixel belongs to the zone containing its
    centre (``all_touched=True``: any zone touching it); where zones overlap, the later row in
    ``zones_gdf`` wins. A grid no zone reaches yields an all-zero mask and no windows; an empty
    ``zones_gdf`` is an error.
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
    check_lon_180(zones_gdf)
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
        zones = zones.to_crs(grid.crs)
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
