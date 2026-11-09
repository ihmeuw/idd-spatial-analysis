"""Conventions shared by every subpackage.

Only relative fragments, column names and CRS strings live here; absolute roots come from the
gitignored ``paths.yaml`` (STANDARDS §Path safety).
"""

from __future__ import annotations

from typing import Final

# Coordinate reference systems
WGS84: Final = "EPSG:4326"
"""Geographic lat/lon; what shapefiles, climate grids and GPS points arrive in."""

WORLD_CYLINDRICAL_EQUAL_AREA: Final = "ESRI:54034"
"""The population model's block frame: metres, equal area, the CRS of every 100 m block."""

# LBD standard admin shapefiles: <admin_shapefile_root>/<release>/<filename>
LBD_ADMIN_FILENAME: Final = "lbd_standard_admin_{level}{suffix}.shp"
LBD_DISPUTED_FILENAME: Final = "lbd_disputed_mask{suffix}.shp"
LBD_SIMPLIFIED_SUFFIX: Final = "_simplified"
LBD_MAX_LEVEL: Final = 2

# Column conventions
ADM_CODE_COLUMN: Final = "ADM{level}_CODE"
ADM_NAME_COLUMN: Final = "ADM{level}_NAME"
LOC_ID_COLUMN: Final = "loc_id"
"""LBD shapefiles: the GBD location id of the most detailed unit in the file."""
LOCATION_ID_COLUMN: Final = "location_id"
"""Population-model raking shapes and hierarchy tables."""
PARENT_ID_COLUMN: Final = "parent_id"
LEVEL_COLUMN: Final = "level"
BLOCK_KEY_COLUMN: Final = "block_key"
TILE_KEY_COLUMN: Final = "tile_key"
ZONE_COLUMN: Final = "zone"

# Tile indexes: <tile_index_root>/<resolution>m/tile_index.parquet
TILE_INDEX_FILENAME: Final = "tile_index.parquet"
TILE_INDEX_RESOLUTION_DIR: Final = "{resolution}m"

# Raster conventions
RASTER_FLOAT_DTYPE: Final = "float32"
"""Value rasters are float32 with NaN as nodata."""
ZONE_DTYPE: Final = "int32"
"""Zone masks are int32; ``ZONE_NODATA`` marks pixels outside every zone."""
ZONE_NODATA: Final = 0
GEOTIFF_BLOCKSIZE: Final = 512
GEOTIFF_COMPRESSION: Final = "LZW"
PIXEL_EDGE_DECIMALS: Final = 6
"""Pixel-index quotients are rounded to this many decimals before floor or ceil, so a coordinate
within a millionth of a pixel of an edge counts as on it. Grids with non-binary steps such as
1/120 degree otherwise land one pixel off."""
GEOCENTRIC_CRS: Final = "EPSG:4978"
"""Earth-centred Cartesian metres; distances between points are measured here."""

# Earth, for spherical cell areas (IUGG mean radius, km)
EARTH_RADIUS_KM: Final = 6371.0088

# Zonal statistics tables (zonal/): one row per zone, and per slice of a 3-D value stack
WEIGHTED_SUM_COLUMN: Final = "weighted_sum"
WEIGHT_SUM_COLUMN: Final = "weight_sum"
MEAN_COLUMN: Final = "mean"
SLICE_COLUMN: Final = "slice"
WEIGHT_COLUMN: Final = "weight"
COARSE_ROW_COLUMN: Final = "coarse_row"
COARSE_COL_COLUMN: Final = "coarse_col"
THRESHOLD_COLUMN: Final = "threshold"
WEIGHT_WITHIN_COLUMN: Final = "weight_within"
WEIGHT_TOTAL_COLUMN: Final = "weight_total"
SHARE_COLUMN: Final = "share"

# Where the CLI materialises an LBD release under a project's output node
ADMIN_SHAPEFILES_DIR: Final = "admin_shapefiles"
