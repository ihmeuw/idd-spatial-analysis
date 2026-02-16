from pathlib import Path

# Project configuration
project_name = 'spatial analysis'
repo_name = f"idd-{project_name.replace(' ', '-')}"      # 'idd-spatial-analysis'
package_name = f"idd_{project_name.replace(' ', '_')}"   # 'idd_spatial_analysis'

# Repository paths
REPO_ROOT = Path("/ihme/homes/bcreiner/repos") / repo_name
MODEL_ROOT = Path("/mnt/team/idd/pub") / project_name.replace(' ', '_')
SCRIPT_ROOT = REPO_ROOT / "src" / package_name


## Data paths
# Population data
POPULATION_DATA_ROOT = Path('/mnt/share/geospatial/ihmepop/gbd_release_id_16/lsae_1285/sex_age_splits.20Nov2025')
TILE_INDEX_PATH = "/mnt/team/rapidresponse/pub/building-density/tiles/100m/tile_index.parquet"
BLOCK_PATH_TEMPLATE = "{POP_RASTER_DIR}/{block_id}.tif"

# Shapefiles
SHAPEFILE_ROOT = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')
ADMIN_SHP_FILENAME_TEMPLATE = f"{SHAPEFILE_ROOT}/lbd_standard_admin_{{admin_level}}{{simple_suffix}}.shp"
DISPUTED_SHP_FILENAME_TEMPLATE = f"{SHAPEFILE_ROOT}/lbd_disputed_mask{{simple_suffix}}.shp"

## Coordinate Reference Systems
# WGS 84 (lat/lon)
LL_CRS = "EPSG:4326"

# Equal area projection for global use (Mollweide)
EA_CRS = "ESRI:54034"

ALL_PROJECTION_DICT = {
    'latlon': {
        'name': 'WGS 84 (lat/lon)',
        'crs': LL_CRS
    },
    'ea': {
        'name': 'Equal-area (Mollweide)',
        'crs': EA_CRS
    }
}