"""
Clinic proximity analysis tools for calculating population coverage.

This module provides functions to:
1. Calculate how many people live within X km of each clinic
2. Calculate how many people live within X km of at least Y clinics
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds, Resampling
from rasterio.features import rasterize
from shapely.geometry import Point, box
from pyproj import CRS, Transformer
import warnings


def load_clinic_data(filepath, crs="EPSG:4326"):
    """
    Load clinic GPS coordinates from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to file containing clinic coordinates
    crs : str, optional
        Coordinate reference system, default is WGS84 (EPSG:4326)
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with clinic locations
    """
    # Convert Path to string if needed
    filepath = str(filepath)
    
    # Detect file type and load accordingly
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        # Assume columns are 'longitude', 'latitude' or 'lon', 'lat' or 'x', 'y'
        lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon', 'x']), None)
        lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat', 'y']), None)
        
        if lon_col is None or lat_col is None:
            raise ValueError("Could not find longitude/latitude columns in CSV")
        
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    else:
        # Try to load as shapefile or other geodata format
        gdf = gpd.read_file(filepath)
        if gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
    
    return gdf


def load_population_raster(filepath):
    """
    Load population raster data.
    
    Parameters
    ----------
    filepath : str
        Path to population raster file
    
    Returns
    -------
    tuple
        (raster_data, transform, crs)
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        # Handle nodata values
        if nodata is not None:
            data = np.where(data == nodata, 0, data)
        
    return data, transform, crs


def calculate_distance_raster(clinic_point, raster_shape, transform, raster_crs):
    """
    Calculate distance from each raster cell to a clinic point.
    
    Parameters
    ----------
    clinic_point : shapely.geometry.Point
        Clinic location (must be in raster_crs)
    raster_shape : tuple
        Shape of the raster (rows, cols)
    transform : affine.Affine
        Raster transform
    raster_crs : CRS
        Coordinate reference system of raster
    
    Returns
    -------
    np.ndarray
        Distance raster in meters
    """
    rows, cols = raster_shape
    
    # Create arrays of pixel coordinates
    col_coords, row_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert pixel coordinates to map coordinates
    xs, ys = rasterio.transform.xy(transform, row_coords, col_coords)
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Calculate distance in meters (assuming equal area projection)
    # For equal area projections, distances are in the projection units (meters)
    dx = xs - clinic_point.x
    dy = ys - clinic_point.y
    distances = np.sqrt(dx**2 + dy**2)
    
    return distances


def population_within_distance(
    clinics_gdf,
    population_raster_path,
    distance_km,
    admin_boundaries=None,
    clinic_id_col='clinic_id'
):
    """
    Calculate population within X km of each clinic.
    
    Parameters
    ----------
    clinics_gdf : gpd.GeoDataFrame
        GeoDataFrame with clinic locations
    population_raster_path : str
        Path to population raster (should be in equal area projection)
    distance_km : float
        Distance threshold in kilometers
    admin_boundaries : gpd.GeoDataFrame, optional
        Administrative boundaries for clipping/summarizing
    clinic_id_col : str, optional
        Column name for clinic identifier
    
    Returns
    -------
    pd.DataFrame
        DataFrame with clinic_id and population_within_Xkm columns
    """
    # Load population raster
    pop_data, transform, raster_crs = load_population_raster(population_raster_path)
    
    # Convert clinics to raster CRS if needed
    if clinics_gdf.crs != raster_crs:
        clinics_proj = clinics_gdf.to_crs(raster_crs)
    else:
        clinics_proj = clinics_gdf.copy()
    
    # Convert distance to meters (assuming raster is in meters)
    distance_m = distance_km * 1000
    
    results = []
    
    for idx, clinic in clinics_proj.iterrows():
        # Calculate distance raster from this clinic
        dist_raster = calculate_distance_raster(
            clinic.geometry, 
            pop_data.shape, 
            transform, 
            raster_crs
        )
        
        # Create mask for pixels within distance
        within_distance = dist_raster <= distance_m
        
        # Sum population within distance
        pop_within = np.sum(pop_data[within_distance])
        
        clinic_id = clinic[clinic_id_col] if clinic_id_col in clinic.index else idx
        
        results.append({
            clinic_id_col: clinic_id,
            f'population_within_{distance_km}km': pop_within
        })
    
    return pd.DataFrame(results)


def population_coverage_multiple_clinics(
    clinics_gdf,
    population_raster_path,
    distance_km,
    min_clinics=1,
    admin_boundaries=None
):
    """
    Calculate population with access to at least Y clinics within X km.
    
    Parameters
    ----------
    clinics_gdf : gpd.GeoDataFrame
        GeoDataFrame with clinic locations
    population_raster_path : str
        Path to population raster (should be in equal area projection)
    distance_km : float
        Distance threshold in kilometers
    min_clinics : int, optional
        Minimum number of clinics within distance (default 1)
    admin_boundaries : gpd.GeoDataFrame, optional
        Administrative boundaries for summarizing results
    
    Returns
    -------
    dict
        Dictionary with results including:
        - total_population: total population in study area
        - population_with_access: population within X km of at least Y clinics
        - coverage_percentage: percentage with access
        - coverage_raster: numpy array showing number of clinics within distance
    """
    # Load population raster
    pop_data, transform, raster_crs = load_population_raster(population_raster_path)
    
    # Convert clinics to raster CRS if needed
    if clinics_gdf.crs != raster_crs:
        clinics_proj = clinics_gdf.to_crs(raster_crs)
    else:
        clinics_proj = clinics_gdf.copy()
    
    # Convert distance to meters
    distance_m = distance_km * 1000
    
    # Initialize coverage raster (counts number of clinics within distance)
    coverage_count = np.zeros_like(pop_data, dtype=np.int32)
    
    print(f"Calculating coverage for {len(clinics_proj)} clinics...")
    
    for idx, clinic in clinics_proj.iterrows():
        if idx % 100 == 0:
            print(f"Processing clinic {idx + 1}/{len(clinics_proj)}")
        
        # Calculate distance raster from this clinic
        dist_raster = calculate_distance_raster(
            clinic.geometry, 
            pop_data.shape, 
            transform, 
            raster_crs
        )
        
        # Increment coverage count for pixels within distance
        within_distance = dist_raster <= distance_m
        coverage_count[within_distance] += 1
    
    # Calculate populations
    total_population = np.sum(pop_data)
    
    # Population with access to at least min_clinics
    has_access = coverage_count >= min_clinics
    population_with_access = np.sum(pop_data[has_access])
    
    coverage_percentage = (population_with_access / total_population * 100) if total_population > 0 else 0
    
    results = {
        'total_population': float(total_population),
        'population_with_access': float(population_with_access),
        'coverage_percentage': float(coverage_percentage),
        'coverage_raster': coverage_count,
        'distance_km': distance_km,
        'min_clinics': min_clinics
    }
    
    # Add admin-level summaries if boundaries provided
    if admin_boundaries is not None:
        admin_results = summarize_by_admin(
            pop_data, 
            coverage_count, 
            transform, 
            raster_crs,
            admin_boundaries, 
            min_clinics
        )
        results['admin_summary'] = admin_results
    
    return results


def summarize_by_admin(pop_data, coverage_count, transform, raster_crs, 
                       admin_boundaries, min_clinics):
    """
    Summarize population coverage by administrative boundaries.
    
    Parameters
    ----------
    pop_data : np.ndarray
        Population raster
    coverage_count : np.ndarray
        Raster counting number of clinics within distance
    transform : affine.Affine
        Raster transform
    raster_crs : CRS
        Raster coordinate system
    admin_boundaries : gpd.GeoDataFrame
        Administrative boundaries
    min_clinics : int
        Minimum number of clinics threshold
    
    Returns
    -------
    pd.DataFrame
        Summary by administrative unit
    """
    # Reproject admin boundaries to raster CRS if needed
    if admin_boundaries.crs != raster_crs:
        admin_proj = admin_boundaries.to_crs(raster_crs)
    else:
        admin_proj = admin_boundaries.copy()
    
    results = []
    
    for idx, admin in admin_proj.iterrows():
        # Rasterize this admin boundary
        mask = rasterize(
            [(admin.geometry, 1)],
            out_shape=pop_data.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)
        
        # Calculate statistics for this admin area
        admin_pop = pop_data[mask]
        admin_coverage = coverage_count[mask]
        
        total_pop = np.sum(admin_pop)
        pop_with_access = np.sum(admin_pop[admin_coverage >= min_clinics])
        coverage_pct = (pop_with_access / total_pop * 100) if total_pop > 0 else 0
        
        # Get admin name (try common column names)
        name_col = next((c for c in admin.index if c.lower() in ['name', 'admin_name', 'adm_name']), None)
        admin_name = admin[name_col] if name_col else f"Admin_{idx}"
        
        results.append({
            'admin_name': admin_name,
            'total_population': total_pop,
            'population_with_access': pop_with_access,
            'coverage_percentage': coverage_pct
        })
    
    return pd.DataFrame(results)


def export_coverage_raster(coverage_results, output_path, reference_raster_path):
    """
    Export coverage raster to file.
    
    Parameters
    ----------
    coverage_results : dict
        Results from population_coverage_multiple_clinics
    output_path : str
        Path for output raster
    reference_raster_path : str
        Path to reference raster (for metadata)
    """
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.int32,
            count=1,
            compress='lzw'
        )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(coverage_results['coverage_raster'], 1)
    
    print(f"Coverage raster exported to {output_path}")
