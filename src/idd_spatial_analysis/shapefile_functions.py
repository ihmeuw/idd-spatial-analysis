"""
shapefile_functions.py
======================
Shapefile loading utilities for project notebooks.

Main entry points
-----------------
load_shapefiles()      Load admin shapefiles for one or more countries,
                       optionally reprojecting to all projections in
                       projection_dict.

load_global_admin0()   Load the full global admin0 shapefile (no country
                       filter). Used to supply neighbour countries to figure
                       functions via config.neighbor_admin0_path, or when you
                       want to inspect the full shapefile programmatically.

rasterize_admin()      Burn admin codes from a GDF into a pixel grid matching
                       a reference raster. Needed before calling
                       calculate_distance_within_admin().
"""

import geopandas as gpd
import numpy as np

from idd_spatial_analysis import constants as rfc

_admin_shp_template    = rfc.ADMIN_SHP_FILENAME_TEMPLATE
_all_projection_dict   = rfc.ALL_PROJECTION_DICT


# =============================================================================
# Core loaders
# =============================================================================

def load_shapefiles(
    ADM0_CODE_LIST=None,
    depth=1,
    simple=False,
    projection_dict=None,
    admin_shp_filename_template=_admin_shp_template,
):
    """
    Load admin-boundary shapefiles for one or more countries at all admin
    levels from 0 up to `depth`.

    Parameters
    ----------
    ADM0_CODE_LIST : list of int or None
        Country codes to load. None = load all countries (global).
    depth : int
        Maximum admin level to load (0 = country only, 1 = admin1, 2 = admin2).
    simple : bool
        If True, load the '_simplified' variants.
    projection_dict : dict or None
        If provided, each GDF is reprojected into every CRS in the dict.
        Keys should be projection names (e.g. 'ea', 'latlon'); values should
        be dicts with at least a 'crs' key (latlon projection is kept as-is).
        If None, shapefiles are stored in latlon only.

    Returns
    -------
    dict keyed by (admin_level_str, projection_name), e.g.
        shapefiles[('admin2', 'ea')]['gdf']
        shapefiles[('admin2', 'ea')]['meta']
    """
    simple_suffix = '_simplified' if simple else ''
    shapefiles = {}

    for admin_level in range(0, depth + 1):
        filename = admin_shp_filename_template.format(
            admin_level=admin_level, simple_suffix=simple_suffix)

        if ADM0_CODE_LIST:
            gdf = _load_subset(
                'ADM0_CODE', ADM0_CODE_LIST, filename)
        else:
            gdf = gpd.read_file(filename)

        if projection_dict is not None:
            for proj_name, proj_info in projection_dict.items():
                if proj_name == 'latlon':
                    gdf_proj = gdf
                else:
                    crs = proj_info['crs'] if isinstance(proj_info, dict) else proj_info
                    gdf_proj = gdf.to_crs(crs)
                shapefiles[(f'admin{admin_level}', proj_name)] = {
                    'gdf':  gdf_proj,
                    'meta': {**proj_info},
                }
        else:
            proj_info = _all_projection_dict['latlon']
            shapefiles[(f'admin{admin_level}', 'latlon')] = {
                'gdf':  gdf,
                'meta': {**proj_info},
            }

    return shapefiles


def load_global_admin0(
    projection_dict=None,
    simple=False,
    admin_shp_filename_template=_admin_shp_template,
):
    """
    Load the full global admin0 shapefile with no country filter.

    This is the path to pass as config.neighbor_admin0_path in figure configs.
    The figure functions lazy-load only the features that appear within each
    panel's bbox, so passing the full path is efficient.

    Parameters
    ----------
    projection_dict : dict or None
        If provided, returns the GDF reprojected into every CRS in the dict,
        same structure as load_shapefiles().
        If None, returns a plain GeoDataFrame in latlon.
    simple : bool
        Load the simplified variant.

    Returns
    -------
    GeoDataFrame (if projection_dict is None) or dict same structure
    as load_shapefiles() restricted to admin0.

    Examples
    --------
    # Get the file path to pass to figure configs (most common use)
    from idd_spatial_analysis import constants as rfc
    neighbor_path = rfc.ADMIN_SHP_FILENAME_TEMPLATE.format(
        admin_level=0, simple_suffix='')

    cfg.neighbor_admin0_path = neighbor_path
    cfg.focus_adm0_code      = 165   # Pakistan ADM0_CODE
    cfg.adm0_code_field      = 'ADM0_CODE'

    # Or load the GDF directly (e.g. to inspect it)
    admin0_global = load_global_admin0()
    """
    return load_shapefiles(
        ADM0_CODE_LIST=None,
        depth=0,
        simple=simple,
        projection_dict=projection_dict,
        admin_shp_filename_template=admin_shp_filename_template,
    )


def rasterize_admin(
    admin_gdf,
    code_col,
    reference_meta,
):
    """
    Burn admin codes from a GeoDataFrame into a raster grid.

    The output raster matches the extent and resolution of `reference_meta`
    (e.g. the population raster meta). Pixels outside all features are 0.

    Parameters
    ----------
    admin_gdf      : GeoDataFrame
        Admin boundaries in the same CRS as reference_meta['crs'].
        Must have a column `code_col` with integer admin codes.
    code_col       : str
        Column in admin_gdf containing integer codes to burn
        (e.g. 'ADM1_CODE' or 'ADM2_CODE').
    reference_meta : dict
        Raster metadata dict with keys: 'transform', 'crs', 'height', 'width'.
        Typically distance_raster['meta'] or population_raster['meta'].

    Returns
    -------
    np.ndarray, shape (1, height, width), dtype int32
        Rasterized admin codes. 0 = outside all features.

    Example
    -------
    admin2_raster = rasterize_admin(
        admin_gdf    = shapefiles[('admin2', 'ea')]['gdf'],
        code_col     = 'ADM2_CODE',
        reference_meta = population_raster['meta'],
    )
    # Then pass to calculate_distance_within_admin:
    dist_raster, _ = calculate_distance_within_admin(
        admin_level      = 2,
        admin_raster     = admin2_raster,
        ...
    )
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    h = reference_meta['height']
    w = reference_meta['width']
    t = reference_meta['transform']

    shapes = (
        (geom, int(code))
        for geom, code in zip(admin_gdf.geometry, admin_gdf[code_col])
        if geom is not None and not geom.is_empty
    )

    burned = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=t,
        fill=0,
        dtype=np.int32,
    )
    return burned[np.newaxis, :, :]   # (1, H, W)


# =============================================================================
# Private helpers
# =============================================================================

def _load_subset(code_name, code_list, shp_filename):
    """Load a shapefile filtered to a list of codes."""
    if len(code_list) == 1:
        where = f"{code_name} = {code_list[0]}"
    else:
        where = f"{code_name} IN {tuple(code_list)}"
    return gpd.read_file(shp_filename, where=where)


# =============================================================================
# Backwards-compatible aliases
# -- Keep so existing callers don't break; delete once updated.
# =============================================================================

def load_subset_of_gdf(CODE_NAME, CODE_LIST, shp_filename):
    """DEPRECATED: use _load_subset or load_shapefiles() directly."""
    return _load_subset(CODE_NAME, CODE_LIST, shp_filename)


__all__ = [
    'load_shapefiles',
    'load_global_admin0',
    'rasterize_admin',
    # Deprecated
    'load_subset_of_gdf',
]
