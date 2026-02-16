"""
Pakistan Dengue Clinic Distance Analysis

This script analyzes the distance from population centers to healthcare clinics
in Pakistan using geospatial raster and vector data. The analysis:
1. Loads clinic locations and administrative boundaries
2. Processes population raster data
3. Calculates distances to nearest clinic within each admin region
4. Saves results in multiple map projections (equal-area and lat/lon)
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import sys
import os
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

from idd_spatial_analysis import constants as rfc
from idd_spatial_analysis.shapefile_functions import load_shapefiles
from idd_spatial_analysis.raster_functions import calculate_distance_within_admin, reproject_raster, save_raster

# ============================================================================
# DEFINE PATHS AND CONSTANTS
# ============================================================================

# Define base data paths
ANALYSIS_DIR = rfc.MODEL_ROOT / "analyses" / "pakistan_denv"
DATA_DIR = ANALYSIS_DIR / "data"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Population data configuration
MOST_RECENT_QUARTER = "2023q1"
POP_RASTER_DIR = rfc.POPULATION_DATA_ROOT / MOST_RECENT_QUARTER / "sex_id_3_age_group_id_22"

# Load global configuration data
tile_index = gpd.read_parquet(rfc.TILE_INDEX_PATH)
block_path_template = rfc.BLOCK_PATH_TEMPLATE

admin_shp_filename_template = rfc.ADMIN_SHP_FILENAME_TEMPLATE
projection_dict = rfc.ALL_PROJECTION_DICT

# Data files
CLINIC_FILE = DATA_DIR / 'ALL_CPs_547.csv'

# Pakistan Country Code (ISO 3166-1 numeric)
ADM0_CODE = 173

# ============================================================================
# HELPER FUNCTION FOR VALIDATION
# ============================================================================

def validate_nodata(raster_data, raster_name, expected_nodata=np.nan):
    """
    Check and report on nodata values in a raster.
    
    Parameters
    ----------
    raster_data : ndarray
        The raster data to check (can be 2D or 3D)
    raster_name : str
        Name of the raster for reporting
    expected_nodata : value
        Expected nodata value (default: np.nan)
    """
    # Handle both 2D and 3D arrays
    if raster_data.ndim == 3:
        data_2d = raster_data[0]
    else:
        data_2d = raster_data
    
    total_pixels = data_2d.size
    
    if np.isnan(expected_nodata):
        nodata_pixels = np.isnan(data_2d).sum()
        nodata_pct = (nodata_pixels / total_pixels) * 100
        print(f"  {raster_name}:")
        print(f"    Total pixels: {total_pixels:,}")
        print(f"    NaN pixels: {nodata_pixels:,} ({nodata_pct:.2f}%)")
        print(f"    Valid pixels: {(total_pixels - nodata_pixels):,}")
        
        if nodata_pixels == 0:
            print(f"    ⚠️  WARNING: No NaN values found!")
        return nodata_pixels > 0
    else:
        nodata_pixels = (data_2d == expected_nodata).sum()
        nodata_pct = (nodata_pixels / total_pixels) * 100
        print(f"  {raster_name}:")
        print(f"    Total pixels: {total_pixels:,}")
        print(f"    Nodata pixels ({expected_nodata}): {nodata_pixels:,} ({nodata_pct:.2f}%)")
        print(f"    Valid pixels: {(total_pixels - nodata_pixels):,}")
        
        if nodata_pixels == 0:
            print(f"    ⚠️  WARNING: No nodata values found!")
        return nodata_pixels > 0

# ============================================================================
# LOAD AND PROCESS CLINIC DATA
# ============================================================================

# Load clinic CSV as a GeoDataFrame
# The CSV contains latitude/longitude coordinates
clinics_gdf = gpd.read_file(
    CLINIC_FILE,
    x_possible_names="Longitude",
    y_possible_names="Latitude",
).set_crs(projection_dict['latlon']['crs'])

# Standardize column names
clinics_gdf = clinics_gdf.rename(columns={
    'Longitude': 'longitude',
    'Latitude': 'latitude',
    'NAME_V1': 'clinic_name',
})

# Reproject clinics to equal-area projection for accurate distance calculations
clinics_ea_gdf = clinics_gdf.to_crs(projection_dict['ea']['crs'])

# Store clinic data in dictionary with both projections
clinics = {
    'latlon': {
        'gdf': clinics_gdf,
        'meta': {
            **projection_dict['latlon'],
        }
    },
    'ea': {
        'gdf': clinics_ea_gdf,
        'meta': {
            **projection_dict['ea'],
        }   
    }
}

# ============================================================================
# LOAD ADMINISTRATIVE BOUNDARIES
# ============================================================================

# Load global admin0 boundaries (simplified version)
global_simple_admin = load_shapefiles(depth=0, simple=True, projection_dict=projection_dict)

# Load Pakistan admin boundaries (all levels, simplified)
admin_simple_gdfs = load_shapefiles(ADM0_CODE_LIST=[ADM0_CODE], depth=2, simple=True, projection_dict=projection_dict)

# Load Pakistan admin boundaries (all levels, full detail)
# These will be used for spatial joins and rasterization
admin_gdfs = load_shapefiles(ADM0_CODE_LIST=[ADM0_CODE], depth=2, simple=False, projection_dict=projection_dict)

# ============================================================================
# SPATIAL JOIN: ASSIGN ADMIN CODES TO CLINICS
# ============================================================================

# For each projection, assign all admin region codes to clinics in one operation
for proj_name, proj_info in projection_dict.items():
    # Start with the clinic GeoDataFrame
    clinics_with_admin = clinics[proj_name]['gdf'].copy()
    
    # Join all admin levels sequentially
    for admin_level in [0, 1, 2]:
        # Select and rename columns to match expected naming convention
        admin_gdf = admin_gdfs[(f'admin{admin_level}', proj_name)]['gdf']
        admin_subset = admin_gdf[[f'ADM{admin_level}_CODE', f'ADM{admin_level}_NAME', 'geometry']].rename(columns={
            f'ADM{admin_level}_CODE': f'ADM{admin_level}_CODE_gps_adm{admin_level}',
            f'ADM{admin_level}_NAME': f'ADM{admin_level}_NAME_gps_adm{admin_level}',
        })
        
        # Spatial join to assign admin codes
        clinics_with_admin = gpd.sjoin(
            clinics_with_admin,
            admin_subset,
            how='left',
            predicate='within',
        ).drop(columns=['index_right'], errors='ignore')
    
    # Update the clinics dictionary with enriched GeoDataFrame
    clinics[proj_name]['gdf'] = clinics_with_admin

# ============================================================================
# IDENTIFY RELEVANT POPULATION RASTER TILES
# ============================================================================

# Find all tiles that intersect with Pakistan's boundaries
country_tiles = gpd.sjoin(
    tile_index,
    admin_gdfs[('admin0', 'ea')]['gdf'],
    how='inner',
    predicate='intersects'
)

# Extract unique block keys for loading population data
country_blocks = country_tiles['block_key'].unique().tolist()

# ============================================================================
# LOAD AND MERGE POPULATION RASTER BLOCKS
# ============================================================================

# Load all block rasters that cover Pakistan
block_rasters = []
for block_id in country_blocks:
    block_path = block_path_template.format(POP_RASTER_DIR=POP_RASTER_DIR, block_id=block_id)
    if os.path.exists(block_path):
        src = rasterio.open(block_path)
        block_rasters.append(src)
        print(f"Loaded: {block_path}")
    else:
        print(f"Warning: File not found - {block_path}")

print(f"\nTotal blocks loaded: {len(block_rasters)}")

# Merge all rasters into a single mosaic for Pakistan
if block_rasters:
    population_ea_raster, population_ea_raster_transform = merge(block_rasters)
    
    # Get metadata from the first raster
    population_ea_raster_meta = block_rasters[0].meta.copy()
    population_ea_raster_meta.update({
        "driver": "GTiff",
        "height": population_ea_raster.shape[1],
        "width": population_ea_raster.shape[2],
        "transform": population_ea_raster_transform,
        "compress": "lzw"
    })
    
    print(f"\nPopulation raster shape: {population_ea_raster.shape}")
    print(f"Population raster bounds: {rasterio.transform.array_bounds(population_ea_raster.shape[1], population_ea_raster.shape[2], population_ea_raster_transform)}")
else:
    print("No blocks were loaded successfully.")

# After the merge operation, add:
for src in block_rasters:
    src.close()

# ============================================================================
# BUFFER COUNTRY BOUNDARIES
# ============================================================================

# Add 500m buffer around Pakistan to avoid edge effects in distance calculations
buffer_distance = 500  # meters (in ea_crs units)

# Buffer in equal-area projection for accurate metric distances
admin0_gdf_buffered_ea_geom = admin_gdfs[('admin0', 'ea')]['gdf'].buffer(buffer_distance)

# Create dictionary with buffered boundaries in all projections
admin0_gdf_buffered = {}
admin0_gdf_buffered['ea'] = {
    'gdf': gpd.GeoDataFrame(geometry=admin0_gdf_buffered_ea_geom),
    'meta': {
        **projection_dict['ea'],
    }
}
for proj_name, proj_info in projection_dict.items():
    if proj_name != 'ea':  # We already have the 'ea' version
        admin0_gdf_buffered[proj_name] = {
            'gdf': admin0_gdf_buffered['ea']['gdf'].to_crs(proj_info['crs']),
            'meta': {
                **proj_info,
            }
        }

# ============================================================================
# CLIP POPULATION RASTER TO COUNTRY BOUNDARIES
# ============================================================================

# Clip population raster to buffered bounding box and mask to polygon shape
# This reduces memory usage by focusing only on Pakistan
with MemoryFile() as memfile:
    # Write population raster to in-memory file
    with memfile.open(**population_ea_raster_meta) as dataset:
        dataset.write(population_ea_raster)  # Write all bands
    
    # Reopen to read and mask
    with memfile.open() as dataset:
        # mask() clips to bounding box and sets outside pixels to nodata
        clipped_population_ea_raster, clipped_population_ea_raster_transform = mask(
            dataset, 
            admin0_gdf_buffered['ea']['gdf'].geometry,  # Use equal-area projection
            crop=True,  # Crop to bounding box
            nodata=np.nan,  # Set outside pixels to NaN
            all_touched=False,  # Only include pixels whose center is within the polygon
            filled=True  # Return masked array filled with nodata
        )

# IMPORTANT FIX: Ensure pixels outside Pakistan are truly NaN
# The mask() function may not always set them properly, especially if source nodata != NaN
print("\n" + "="*80)
print("Checking clipped population raster mask...")
print("="*80)

# Get the actual mask from the original operation
with MemoryFile() as memfile:
    with memfile.open(**population_ea_raster_meta) as dataset:
        dataset.write(population_ea_raster)
    with memfile.open() as dataset:
        # Get both the data and the mask
        clipped_data, clipped_transform = mask(
            dataset, 
            admin0_gdf_buffered['ea']['gdf'].geometry,
            crop=True,
            nodata=np.nan,
            all_touched=False,
            filled=False  # Keep as masked array to identify masked pixels
        )
        # Now explicitly set masked pixels to NaN
        clipped_population_ea_raster = clipped_data.filled(np.nan)

# Display memory reduction statistics
print(f"\nOriginal population raster shape: {population_ea_raster.shape}")
print(f"Clipped population raster shape: {clipped_population_ea_raster.shape}")
print(f"Memory reduction: {(1 - clipped_population_ea_raster.nbytes / population_ea_raster.nbytes) * 100:.1f}%")
print(f"Clipped memory: {clipped_population_ea_raster.nbytes / (1024**2):.2f} MB")

# Validate the clipping worked
print("\nValidating clipped population raster (equal-area):")
validate_nodata(clipped_population_ea_raster, "Clipped EA Population", expected_nodata=np.nan)

# Store clipped population raster in equal-area projection
population_rasters = {}
population_rasters['ea'] = {
    'data': clipped_population_ea_raster,
    'meta': {
        **population_ea_raster_meta,
    }
}
population_rasters['ea']['meta'].update({
    "height": clipped_population_ea_raster.shape[1],
    "width": clipped_population_ea_raster.shape[2],
    "transform": clipped_population_ea_raster_transform,
    'bounds': rasterio.transform.array_bounds(clipped_population_ea_raster.shape[1], clipped_population_ea_raster.shape[2], clipped_population_ea_raster_transform),
    'nodata': np.nan  # Explicitly set nodata in metadata
})

# ============================================================================
# REPROJECT POPULATION RASTER TO LAT/LON
# ============================================================================

# Reproject clipped population raster to lat/lon for geographic visualization
print("\n" + "="*80)
print("Reprojecting clipped population raster to lat/lon...")
print("="*80)
dst_raster = reproject_raster(
    src_raster = population_rasters['ea'],
    data_key='data',
    src_raster_key = 'ea',
    dst_raster_key = 'latlon',
    projection_dict=projection_dict,
    nodata=np.nan
)
population_rasters['latlon'] = dst_raster
population_rasters['latlon']['meta']['nodata'] = np.nan  # Ensure nodata is set

# Validate reprojected population
print("\nValidating reprojected population raster (lat/lon):")
validate_nodata(population_rasters['latlon']['data'], "Reprojected LatLon Population", expected_nodata=np.nan)

# ============================================================================
# CREATE ADMIN REGION RASTERS
# ============================================================================

# Rasterize admin boundaries to create integer rasters matching population data dimensions
# Each pixel will contain the admin code for that location (0 = outside Pakistan)
admin_rasters = {}

print("\n" + "="*80)
print("Creating admin region rasters...")
print("="*80)

for admin_level in [0, 1, 2]:
    admin_key = f'admin{admin_level}'
    for proj_name, proj_info in projection_dict.items():
        # Get admin boundaries for this level and projection
        gdf = admin_gdfs[(admin_key, proj_name)]['gdf']
        
        # Create (geometry, code) pairs for rasterization
        shapes = [(geom, code) for geom, code in 
                zip(gdf.geometry, gdf[f'ADM{admin_level}_CODE'])]
        
        # Use population raster dimensions and transform
        meta = population_rasters[proj_name]['meta']
        height, width = meta['height'], meta['width']
        transform = meta['transform']
        
        # Rasterize admin boundaries (temp fill value will be replaced)
        admin_raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Temporary fill - will be masked to match population
            dtype=np.int32
        )
        
        # Apply population mask: set pixels outside Pakistan to 0
        # This ensures admin rasters align perfectly with population/distance rasters
        pop_mask_outside = np.isnan(population_rasters[proj_name]['data'][0])
        admin_raster[pop_mask_outside] = 0
        
        # Store rasterized admin boundaries
        admin_rasters[(admin_key, proj_name)] = {
            'data': admin_raster[np.newaxis, :, :],  # Add band dimension
            'meta': {
                **proj_info,
                'transform': transform,
                'height': height,
                'width': width,
                'nodata': 0  # 0 = outside Pakistan
            }
        }

# ============================================================================
# CALCULATE PIXEL CENTER COORDINATES
# ============================================================================

# Create arrays of pixel center coordinates for distance calculations
# These are in the equal-area projection for accurate distance metrics
ea_meta = population_rasters['ea']['meta']
rows_ea, cols_ea = np.indices((ea_meta['height'], ea_meta['width']))

# Convert pixel coordinates to world coordinates (equal-area CRS)
# Using affine transform: x = c + col * a, y = f + row * e
# Add 0.5 to get pixel centers instead of upper-left corners
xs_ea = ea_meta['transform'].c + (cols_ea + 0.5) * ea_meta['transform'].a
ys_ea = ea_meta['transform'].f + (rows_ea + 0.5) * ea_meta['transform'].e

print(f"\nPixel center coordinate arrays shape: {xs_ea.shape}")
print(f"X range: {xs_ea.min():.2f} to {xs_ea.max():.2f}")
print(f"Y range: {ys_ea.min():.2f} to {ys_ea.max():.2f}")

# ============================================================================
# CALCULATE DISTANCE TO NEAREST CLINIC BY ADMIN LEVEL
# ============================================================================

# Initialize distance rasters dictionary
distance_rasters = {}
ea_meta = population_rasters['ea']['meta']

# Calculate distances for all admin levels
# Distance is calculated within each admin region separately
for admin_level in [0, 1, 2]:
    print("\n" + "="*80)
    print(f"Calculating distances for Admin {admin_level} under equal area...")
    print("="*80)
    admin_key = f'admin{admin_level}'
    
    # Calculate distance to nearest clinic within each admin region
    dist_admin_ea, admin_without_clinics = calculate_distance_within_admin(
        admin_level=admin_level,
        admin_raster=admin_rasters[(admin_key, 'ea')]['data'],
        admin_gdf=admin_gdfs[(admin_key, 'ea')]['gdf'],
        clinics_gdf=clinics['ea']['gdf'],
        xs=xs_ea,
        ys=ys_ea,
        height=ea_meta['height'],
        width=ea_meta['width'],
        population_raster=population_rasters['ea']['data'],
        verbose=True
    )

    # Validate distance raster has NaN values for outside Pakistan
    print(f"\nValidating {admin_key} distance raster (equal-area):")
    validate_nodata(dist_admin_ea, f"{admin_key.upper()} EA Distance", expected_nodata=np.nan)

    # Store distance raster in equal-area projection
    distance_rasters[(admin_key, 'ea')] = {
        'data': dist_admin_ea,
        'meta': {
            **projection_dict['ea'],
            'transform': ea_meta['transform'],
            'height': ea_meta['height'],
            'width': ea_meta['width'],
            'nodata': np.nan
        },
    }

    # Reproject distance raster to lat/lon
    print(f"\n  Reprojecting {admin_key} to lat/lon...")
    dst_raster = reproject_raster(
        src_raster = distance_rasters[(admin_key, 'ea')],
        data_key='data',
        src_raster_key = 'ea',
        dst_raster_key = 'latlon',
        projection_dict=projection_dict,
        resampling_method='nearest',
        nodata=np.nan
    )

    # Store reprojected distance raster
    distance_rasters[(admin_key, 'latlon')] = dst_raster
    distance_rasters[(admin_key, 'latlon')]['meta']['nodata'] = np.nan
    
    # Validate reprojected distance raster
    print(f"\nValidating {admin_key} distance raster (lat/lon):")
    validate_nodata(distance_rasters[(admin_key, 'latlon')]['data'], 
                   f"{admin_key.upper()} LatLon Distance", 
                   expected_nodata=np.nan)

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

# Save clinic data as GeoPackage files (better format than shapefiles)
print("\n" + "="*80)
print("Saving clinic GeoPackage files...")
print("="*80)
clinics['latlon']['gdf'].to_file(
    OUTPUT_DIR / 'clinics_latlon.gpkg',
    driver='GPKG'
)

clinics['ea']['gdf'].to_file(
    OUTPUT_DIR / 'clinics_ea.gpkg', 
    driver='GPKG'
)

print("\n" + "="*80)
print("Saving admin rasters...")
print("="*80)

for admin_level in [0, 1, 2]:
    admin_key = f'admin{admin_level}'
    for proj_name in projection_dict.keys():
        print(f"\nSaving {admin_key} raster for: {proj_name}")
        raster = admin_rasters[(admin_key, proj_name)]
        
        # Validate before saving
        print(f"Pre-save validation for {admin_key}_{proj_name}:")
        validate_nodata(raster['data'], f"{admin_key}_{proj_name}", expected_nodata=0)
        
        save_raster(
            raster_data=raster['data'],
            transform=raster['meta']['transform'],
            crs=raster['meta']['crs'],
            filepath=OUTPUT_DIR / f'{admin_key}_raster_{proj_name}.tif',
            dtype='int32',
            nodata=0  # 0 = outside all admin regions
        )

print("\n" + "="*80)      
print("Saving distance rasters...")
print("="*80)
n_rasters_saved = 0

# Save Distance Rasters (6 total: 3 admin levels × 2 projections)
for admin_level in [0, 1, 2]:
    print(f"\nSaving rasters for Admin {admin_level}...")
    admin_key = f'admin{admin_level}'
    for proj_name in projection_dict.keys():
        print(f"  Projection: {proj_name}")
        key = (admin_key, proj_name)
        raster = distance_rasters[key]
        
        # Validate before saving
        print(f"  Pre-save validation for {admin_key}_{proj_name}:")
        validate_nodata(raster['data'], f"  {admin_key}_{proj_name}_distance", expected_nodata=np.nan)
        
        save_raster(
            raster_data=raster['data'],
            transform=raster['meta']['transform'],
            crs=raster['meta']['crs'],
            filepath=OUTPUT_DIR / f'dist_nearest_clinic_{admin_key}_{proj_name}_km.tif',
            dtype='float32',
            nodata=np.nan
        )
        n_rasters_saved += 1

# Save Population Rasters (2 total: 2 projections)
print("\n" + "="*80)
print("Saving population rasters...")
print("="*80)

for proj_name in projection_dict.keys():
    print(f"\nSaving population raster for: {proj_name}")
    raster = population_rasters[proj_name]
    
    # Validate before saving
    print(f"Pre-save validation for population_{proj_name}:")
    validate_nodata(raster['data'], f"population_{proj_name}", expected_nodata=np.nan)
    
    save_raster(
        raster_data=raster['data'],
        transform=raster['meta']['transform'],
        crs=raster['meta']['crs'],
        filepath=OUTPUT_DIR / f'population_clipped_{proj_name}.tif',
        dtype='float32',
        nodata=np.nan
    )
    n_rasters_saved += 1

print("\n" + "="*80)
print(f"✓ All {n_rasters_saved} rasters saved successfully!")
print("="*80)