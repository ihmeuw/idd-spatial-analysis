"""Points to the zones that contain them."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from idd_spatial_analysis.crs import crs_equal


def points_to_zones(
    points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    zone_col: str,
    *,
    predicate: str = "within",
) -> pd.Series:
    """The zone id of each point, aligned to ``points.index``; ``<NA>`` where no zone contains it.

    Points are reprojected to the zones' CRS. A point matching several zones (on a shared
    boundary, or overlapping zones) keeps the first match in ``zones`` order.
    """
    if zone_col not in zones.columns:
        msg = f"zones lack column {zone_col!r}"
        raise KeyError(msg)
    if points.crs is None or zones.crs is None:
        msg = "points and zones must both carry a CRS"
        raise ValueError(msg)
    if points.index.has_duplicates:
        msg = "points index must be unique"
        raise ValueError(msg)
    located = points[["geometry"]]
    if not crs_equal(points.crs, zones.crs):
        located = located.to_crs(zones.crs)
    joined = gpd.sjoin(located, zones[[zone_col, "geometry"]], how="left", predicate=predicate)
    first = joined[~joined.index.duplicated(keep="first")]
    result = first[zone_col].reindex(points.index)
    return pd.Series(result, name=zone_col).astype("Int64")
