"""
spatial_utils.py
================
Spatial data utilities for project notebooks.

Sections
--------
1. GeoDataFrame clipping / filtering  — supersedes map_layout.clip_to_bbox
                                         for non-figure use
2. Water layer loading                — load Natural Earth water features
                                         for figure functions
3. CRS / reprojection helpers
4. Point creation
5. Area calculation
6. Backwards-compatible wrappers (to be deleted once callers are updated)
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Tuple
from shapely.geometry import Point, box


# =============================================================================
# 1. GeoDataFrame clipping / filtering
# =============================================================================

def clip_gdf(gdf: gpd.GeoDataFrame,
             bbox_geom) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to a shapely geometry (typically a box).
    This is the canonical clip function used across the whole project.

    Parameters
    ----------
    gdf       : GeoDataFrame to clip
    bbox_geom : shapely geometry (box, polygon, etc.)

    Returns
    -------
    GeoDataFrame clipped to bbox_geom
    """
    return gpd.clip(gdf, bbox_geom)


def clip_gdf_to_bounds(gdf: gpd.GeoDataFrame,
                       bounds: Tuple[float, float, float, float]
                       ) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to a bounding box given as (minx, miny, maxx, maxy).
    Convenience wrapper around clip_gdf for when you have a bounds tuple
    rather than a shapely geometry.

    Parameters
    ----------
    gdf    : GeoDataFrame to clip
    bounds : (minx, miny, maxx, maxy) in the same CRS as gdf

    Returns
    -------
    GeoDataFrame clipped to bounds
    """
    return gpd.clip(gdf, box(*bounds))


def bbox_from_gdf(gdf: gpd.GeoDataFrame,
                  buffer: float = 0):
    """
    Return a shapely box around gdf.total_bounds with optional buffer.

    Parameters
    ----------
    gdf    : GeoDataFrame
    buffer : buffer in CRS units (metres for projected; degrees for latlon)

    Returns
    -------
    shapely.geometry.box
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    return box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)


# =============================================================================
# 2. Water layer loading
# =============================================================================

def load_water_gdfs(crs,
                    resolution: str = '10m',
                    include_ocean: bool = True,
                    include_lakes: bool = True,
                    include_rivers: bool = True
                    ) -> dict:
    """
    Load Natural Earth water features via cartopy.io.shapereader and
    reproject to the given CRS.

    Call this once at the top of your notebook and pass the results
    to make_schematic_figure / make_raster_map / make_choropleth_map
    as ocean_gdf, lakes_gdf, rivers_gdf.

    Parameters
    ----------
    crs        : target CRS — anything accepted by GeoDataFrame.to_crs()
                 e.g. a pyproj.CRS, EPSG string, or WKT string.
                 Typically projection_dict['ea']['crs'].
    resolution : '10m' (most detailed), '50m', or '110m'
    include_ocean   : load ocean polygons
    include_lakes   : load lake polygons
    include_rivers  : load river + lake-centerline lines

    Returns
    -------
    dict with keys 'ocean', 'lakes', 'rivers' (any not requested are None)

    Example
    -------
    water = load_water_gdfs(projection_dict['ea']['crs'])
    fig, ax = make_raster_map(...,
        ocean_gdf  = water['ocean'],
        lakes_gdf  = water['lakes'],
        rivers_gdf = water['rivers'],
    )
    """
    import cartopy.io.shapereader as shpreader

    result = {'ocean': None, 'lakes': None, 'rivers': None}

    if include_ocean:
        path = shpreader.natural_earth(resolution=resolution,
                                       category='physical', name='ocean')
        result['ocean'] = gpd.read_file(path).to_crs(crs)

    if include_lakes:
        path = shpreader.natural_earth(resolution=resolution,
                                       category='physical', name='lakes')
        result['lakes'] = gpd.read_file(path).to_crs(crs)

    if include_rivers:
        path = shpreader.natural_earth(resolution=resolution,
                                       category='physical',
                                       name='rivers_lake_centerlines')
        result['rivers'] = gpd.read_file(path).to_crs(crs)

    return result


# =============================================================================
# 3. CRS / reprojection helpers
# =============================================================================

def reproject_gdf(gdf: gpd.GeoDataFrame,
                  target_crs) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to target_crs.
    Thin wrapper kept for readability in notebooks.
    """
    return gdf.to_crs(target_crs)


# =============================================================================
# 4. Point creation
# =============================================================================

def points_from_xy(df: pd.DataFrame,
                   x_col: str = 'longitude',
                   y_col: str = 'latitude',
                   crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame with x/y coordinate columns.

    Parameters
    ----------
    df    : DataFrame with coordinate columns
    x_col : column name for x / longitude
    y_col : column name for y / latitude
    crs   : CRS of the coordinates

    Returns
    -------
    GeoDataFrame with Point geometry
    """
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


# =============================================================================
# 5. Area calculation
# =============================================================================

def calculate_area(gdf: gpd.GeoDataFrame,
                   units: str = 'km2') -> pd.Series:
    """
    Calculate area of geometries, auto-projecting to a UTM CRS if needed.

    Parameters
    ----------
    gdf   : GeoDataFrame
    units : 'm2' | 'km2' | 'ha'

    Returns
    -------
    pd.Series of areas in requested units
    """
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs()) if not gdf.crs.is_projected else gdf
    areas = gdf_proj.geometry.area
    if units == 'km2':
        return areas / 1_000_000
    elif units == 'ha':
        return areas / 10_000
    return areas   # m2


# =============================================================================
# 6. Backwards-compatible wrappers
# -- These are thin aliases kept so existing callers don't break.
# -- Delete once all callers have been updated.
# =============================================================================

def load_shapefile(filepath: Union[str, Path]) -> gpd.GeoDataFrame:
    """DEPRECATED: use gpd.read_file() directly."""
    return gpd.read_file(filepath)


def spatial_join(left_gdf: gpd.GeoDataFrame,
                 right_gdf: gpd.GeoDataFrame,
                 how: str = 'inner',
                 predicate: str = 'intersects') -> gpd.GeoDataFrame:
    """DEPRECATED: use gpd.sjoin() directly."""
    return gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)


def create_buffer(gdf: gpd.GeoDataFrame,
                  distance: float,
                  units: str = 'meters') -> gpd.GeoDataFrame:
    """DEPRECATED: buffer directly on gdf.geometry or use a projected CRS."""
    if units == 'meters' and not gdf.crs.is_projected:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        out = gdf_proj.copy()
        out.geometry = gdf_proj.buffer(distance)
        return out.to_crs(gdf.crs)
    out = gdf.copy()
    out.geometry = gdf.buffer(distance)
    return out


def clip_by_bounds(gdf: gpd.GeoDataFrame,
                   bounds: Tuple[float, float, float, float]
                   ) -> gpd.GeoDataFrame:
    """DEPRECATED: use clip_gdf_to_bounds()."""
    return clip_gdf_to_bounds(gdf, bounds)


def load_raster_as_xarray(filepath: Union[str, Path],
                           band: Optional[int] = None):
    """DEPRECATED: use rioxarray directly."""
    import rioxarray as rxr
    da = rxr.open_rasterio(filepath)
    if band is not None:
        da = da.sel(band=band)
    return da


__all__ = [
    # Current API
    'clip_gdf',
    'clip_gdf_to_bounds',
    'bbox_from_gdf',
    'load_water_gdfs',
    'reproject_gdf',
    'points_from_xy',
    'calculate_area',
    # Deprecated
    'load_shapefile',
    'spatial_join',
    'create_buffer',
    'clip_by_bounds',
    'load_raster_as_xarray',
]
