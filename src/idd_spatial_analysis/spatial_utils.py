"""
Core spatial utilities for geospatial analysis
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Union, Optional, Tuple
import xarray as xr
from shapely.geometry import Point, Polygon, box


def load_shapefile(filepath: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load a shapefile into a GeoDataFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the shapefile
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the shapefile data
    """
    return gpd.read_file(filepath)


def reproject_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: Union[str, int] = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to a target CRS.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    target_crs : str or int
        Target coordinate reference system (default: EPSG:4326)
        
    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame
    """
    return gdf.to_crs(target_crs)


def create_buffer(
    gdf: gpd.GeoDataFrame,
    distance: float,
    units: str = "meters"
) -> gpd.GeoDataFrame:
    """
    Create a buffer around geometries in a GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    distance : float
        Buffer distance
    units : str
        Units for buffer distance ('meters' or 'degrees')
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with buffered geometries
    """
    if units == "meters" and not gdf.crs.is_projected:
        # Convert to projected CRS for accurate meter-based buffer
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        buffered = gdf_proj.copy()
        buffered.geometry = gdf_proj.buffer(distance)
        return buffered.to_crs(gdf.crs)
    else:
        buffered = gdf.copy()
        buffered.geometry = gdf.buffer(distance)
        return buffered


def spatial_join(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    how: str = "inner",
    predicate: str = "intersects"
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between two GeoDataFrames.
    
    Parameters
    ----------
    left_gdf : gpd.GeoDataFrame
        Left GeoDataFrame
    right_gdf : gpd.GeoDataFrame
        Right GeoDataFrame
    how : str
        Type of join ('inner', 'left', 'right')
    predicate : str
        Spatial predicate ('intersects', 'contains', 'within', etc.)
        
    Returns
    -------
    gpd.GeoDataFrame
        Joined GeoDataFrame
    """
    return gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)


def clip_by_bounds(
    gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame by bounding box coordinates.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    bounds : tuple
        Bounding box (minx, miny, maxx, maxy)
        
    Returns
    -------
    gpd.GeoDataFrame
        Clipped GeoDataFrame
    """
    bbox = box(*bounds)
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs=gdf.crs)
    return gpd.clip(gdf, bbox_gdf)


def load_raster_as_xarray(
    filepath: Union[str, Path],
    band: Optional[int] = None
) -> xr.DataArray:
    """
    Load a raster file as an xarray DataArray.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the raster file
    band : int, optional
        Band number to load (1-indexed). If None, loads all bands.
        
    Returns
    -------
    xr.DataArray
        Raster data as xarray DataArray
    """
    import rioxarray as rxr
    da = rxr.open_rasterio(filepath)
    
    if band is not None:
        da = da.sel(band=band)
        
    return da


def calculate_area(
    gdf: gpd.GeoDataFrame,
    units: str = "km2"
) -> pd.Series:
    """
    Calculate area of geometries in a GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    units : str
        Output units ('m2', 'km2', or 'ha')
        
    Returns
    -------
    pd.Series
        Areas in specified units
    """
    # Convert to projected CRS for accurate area calculation
    if not gdf.crs.is_projected:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf
    
    areas = gdf_proj.geometry.area
    
    # Convert units
    if units == "km2":
        areas = areas / 1_000_000
    elif units == "ha":
        areas = areas / 10_000
    # m2 is default, no conversion needed
    
    return areas


def points_from_xy(
    df: pd.DataFrame,
    x_col: str = "longitude",
    y_col: str = "latitude",
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame with x, y coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Column name for x coordinates (longitude)
    y_col : str
        Column name for y coordinates (latitude)
    crs : str
        Coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries
    """
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


__all__ = [
    "load_shapefile",
    "reproject_gdf",
    "create_buffer",
    "spatial_join",
    "clip_by_bounds",
    "load_raster_as_xarray",
    "calculate_area",
    "points_from_xy",
]
