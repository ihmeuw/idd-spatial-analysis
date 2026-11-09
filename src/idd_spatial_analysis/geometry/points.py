"""Points to the zones that contain them."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from idd_spatial_analysis.crs import crs_equal

_ZONE_ID_DTYPE = "Int32"


def points_to_zones(
    points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    zone_col: str,
    *,
    predicate: str = "within",
) -> pd.Series:
    """The zone id of each point as nullable ``Int32``, aligned to ``points.index``.

    Points are reprojected to the zones' CRS. ``<NA>`` where no zone matches: with the default
    ``predicate="within"`` that includes points on a zone boundary, since a boundary is not the
    interior; ``predicate="intersects"`` counts them. A point matching several zones keeps the
    zone listed last in ``zones``, the rule ``raster.rasterize_zones`` applies to overlapping
    polygons (later rows win), so a facility's zone and the mask under it agree.
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
    if not pd.api.types.is_integer_dtype(zones[zone_col]):
        msg = f"zone ids must be integers, got {zones[zone_col].dtype}"
        raise TypeError(msg)
    located = points[["geometry"]]
    if not crs_equal(points.crs, zones.crs):
        located = located.to_crs(zones.crs)
    candidates = zones[[zone_col, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(located, candidates, how="left", predicate=predicate)
    # index_right is the zone's position in ``zones``; after a stable ascending sort, keeping
    # the last row per point makes the last-listed zone win, as rasterize_zones' burn order does.
    ordered = joined.sort_values("index_right", kind="stable")
    last = ordered[~ordered.index.duplicated(keep="last")]
    result = last[zone_col].reindex(points.index)
    return pd.Series(result, name=zone_col).astype(_ZONE_ID_DTYPE)
