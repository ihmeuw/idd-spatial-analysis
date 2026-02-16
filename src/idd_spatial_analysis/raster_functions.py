"""
raster_functions.py
===================
Raster analysis functions for clinic distance and population modelling.

Main entry points
-----------------
reproject_raster()                Reproject a raster dict to a new CRS.
calculate_distance_within_admin() Distance-to-nearest-clinic per pixel,
                                   within the same admin region.
calc_pct_within_threshold()       Per-admin % of population within X km
                                   of a clinic. Produces the GDF column
                                   needed by make_choropleth_map().
save_raster()                     Write a raster dict to a GeoTIFF.
"""

from curses import meta
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.warp import calculate_default_transform, reproject, Resampling


# =============================================================================
# Reprojection
# =============================================================================

def reproject_raster(
    src_raster,
    data_key,
    src_raster_key,
    dst_raster_key,
    projection_dict,
    dst_resolution=None,
    resampling_method='bilinear',
    nodata=None,
):
    """
    Reproject a raster dict from one CRS to another.

    Parameters
    ----------
    src_raster      : dict with 'data' (3D ndarray) and 'meta' keys
    data_key        : unused — kept for backwards compatibility
    src_raster_key  : key into projection_dict for the source CRS
    dst_raster_key  : key into projection_dict for the destination CRS
    projection_dict : dict mapping projection names to dicts with 'crs' key
    dst_resolution  : (x_res, y_res) in destination CRS units, or None
                      to use the default transform
    resampling_method : 'nearest' | 'bilinear' | 'cubic' | 'average' |
                        'mode' | 'max' | 'min'
    nodata          : nodata value; None = use whatever is in the source

    Returns
    -------
    dict with 'data' (3D ndarray) and 'meta' keys in destination CRS
    """
    src_crs  = projection_dict[src_raster_key]['crs']
    dst_crs  = projection_dict[dst_raster_key]['crs']
    src_meta = src_raster['meta']

    if len(src_raster['data'].shape) != 3:
        raise ValueError("Input raster data must be 3D (bands, height, width)")

    bounds = rasterio.transform.array_bounds(
        src_meta['height'], src_meta['width'], src_meta['transform'])

    if dst_resolution is None:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs,
            src_meta['width'], src_meta['height'],
            *bounds,
        )
    else:
        from rasterio.transform import from_bounds
        dst_width  = int((bounds[2] - bounds[0]) / dst_resolution[0])
        dst_height = int((bounds[3] - bounds[1]) / dst_resolution[1])
        dst_transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            dst_width, dst_height)

    _resampling = {
        'nearest':  Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic':    Resampling.cubic,
        'average':  Resampling.average,
        'mode':     Resampling.mode,
        'max':      Resampling.max,
        'min':      Resampling.min,
    }.get(resampling_method.lower(), Resampling.bilinear)

    dst_data = np.empty(
        (src_raster['data'].shape[0], dst_height, dst_width),
        dtype=src_raster['data'].dtype)

    reproject(
        source=src_raster['data'],
        destination=dst_data,
        src_transform=src_meta['transform'],
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=_resampling,
        src_nodata=nodata,
        dst_nodata=nodata,
    )

    return {
        'data': dst_data,
        'meta': {
            'transform': dst_transform,
            'crs':       dst_crs,
            'width':     dst_width,
            'height':    dst_height,
            'bounds':    rasterio.transform.array_bounds(
                             dst_height, dst_width, dst_transform),
        },
    }


# =============================================================================
# Distance calculation
# =============================================================================

def calculate_distance_within_admin(
    admin_level,
    admin_raster,
    admin_gdf,
    clinics_gdf,
    xs, ys,
    height, width,
    population_raster,
    set_nopopulation_to_nan=False,
    verbose=True,
):
    """
    Calculate distance to nearest clinic within the same admin region
    for every pixel in the raster.

    Parameters
    ----------
    admin_level            : int — 0, 1, or 2
    admin_raster           : (1,H,W) or (H,W) int array of admin codes
                             produced by shapefile_functions.rasterize_admin()
    admin_gdf              : GeoDataFrame with ADMn_CODE and ADMn_NAME columns
    clinics_gdf            : GeoDataFrame of clinic points (equal-area CRS)
                             with column ADMn_CODE_gps_admn
    xs, ys                 : (H,W) arrays of pixel-centre coordinates
    height, width          : int — raster dimensions
    population_raster      : (1,H,W) or (H,W) population array
    set_nopopulation_to_nan: if True, zero-pop pixels → NaN in output
    verbose                : print per-region statistics

    Returns
    -------
    distance_raster : (1,H,W) float32 ndarray, distances in km
                      NaN for pixels outside country boundaries;
                      np.inf for pixels in regions with no clinics
    regions_without_clinics : list of admin codes with no clinics
    """
    import time
    from scipy.spatial import cKDTree

    start = time.time()

    admin_2d = admin_raster[0] if admin_raster.ndim == 3 else admin_raster
    pop_2d   = population_raster[0] if population_raster.ndim == 3 else population_raster

    distance_raster = np.full((height, width), np.inf, dtype=np.float32)
    unique_codes    = np.unique(admin_2d[admin_2d > 0])

    print(f"Processing {len(unique_codes)} Admin {admin_level} regions...")
    regions_without_clinics = []

    for code in unique_codes:
        name = admin_gdf.loc[
            admin_gdf[f'ADM{admin_level}_CODE'] == code,
            f'ADM{admin_level}_NAME'
        ].iloc[0]

        clinics_here = clinics_gdf[
            clinics_gdf[f'ADM{admin_level}_CODE_gps_adm{admin_level}'] == code]

        if len(clinics_here) == 0:
            if verbose:
                print(f"\nAdmin {admin_level}: {name} (Code: {code})")
                print(f"  WARNING: No clinics in this region")
            regions_without_clinics.append(code)
            continue

        if verbose:
            print(f"\nAdmin {admin_level}: {name} (Code: {code})")

        clinic_coords = np.column_stack(
            [clinics_here.geometry.x, clinics_here.geometry.y])
        tree = cKDTree(clinic_coords)

        mask         = admin_2d == code
        pixel_coords = np.column_stack([xs[mask], ys[mask]])
        dists, _     = tree.query(pixel_coords, workers=-1)
        dists_km     = (dists / 1000).astype(np.float32)

        distance_raster[mask] = dists_km

        if verbose:
            print(f"  Pixels:  {mask.sum():,}")
            print(f"  Clinics: {len(clinics_here)}")
            print(f"  Range:   {dists_km.min():.2f} – {dists_km.max():.2f} km")
            pop_here  = pop_2d[mask]
            valid_pop = pop_here > 0
            if valid_pop.sum() > 0:
                pw_mean = np.average(dists_km[valid_pop], weights=pop_here[valid_pop])
                print(f"  Pop-weighted mean: {pw_mean:.2f} km")
            else:
                print(f"  Pop-weighted mean: N/A (no population)")

    if set_nopopulation_to_nan:
        distance_raster[pop_2d == 0] = np.nan

    distance_raster[admin_2d == 0] = np.nan   # outside country

    elapsed = time.time() - start
    n_with  = len(unique_codes) - len(regions_without_clinics)

    print(f"\n{'='*60}")
    print(f"Admin {admin_level} distance calculation: {elapsed:.1f}s")
    if n_with > 0:
        print(f"  Avg per region (with clinics): {elapsed/n_with:.2f}s")

    valid = (admin_2d > 0) & ~np.isinf(distance_raster) & ~np.isnan(distance_raster)
    if valid.sum() > 0:
        print(f"  Distance range (valid): "
              f"{distance_raster[valid].min():.2f} – {distance_raster[valid].max():.2f} km")
        print(f"  Mean distance (valid):  {distance_raster[valid].mean():.2f} km")

    print(f"  NaN pixels (outside country or zero pop): {np.isnan(distance_raster).sum():,}")
    print(f"  Pixels w/o clinics (np.inf):     {(np.isinf(distance_raster)).sum():.0f}")

    if regions_without_clinics:
        print(f"\n  Admin {admin_level} regions without clinics "
              f"({len(regions_without_clinics)}): {regions_without_clinics}")

    return distance_raster[np.newaxis, :, :], regions_without_clinics


# =============================================================================
# Choropleth statistics
# =============================================================================

def calc_pct_within_multiple_thresholds(
    distance_raster,
    population_raster,
    admin_raster,
    admin_gdf,
    admin_code_col,
    thresholds_km,
    clinic_counts=None,
    clinic_count_col='n_clinics',
    n_jobs=1,
):
    """
    Efficiently compute % population within MULTIPLE distance thresholds
    for each admin unit. Much faster than calling calc_pct_within_threshold
    repeatedly.

    Parameters
    ----------
    distance_raster   : dict {'data': (1,H,W) float km, 'meta': {...}}
                        or (1,H,W) ndarray
    population_raster : dict {'data': (1,H,W), 'meta': {...}}
                        or (1,H,W) ndarray
    admin_raster      : (1,H,W) or (H,W) int array of admin codes
    admin_gdf         : GeoDataFrame with admin_code_col and geometry
    admin_code_col    : str — e.g. 'ADM2_CODE'
    thresholds_km     : list of floats — e.g. [1, 2, 5, 10, 20, 50]
    clinic_counts     : pd.Series or dict mapping admin code → clinic count
    clinic_count_col  : str — column name for clinic counts in output
    n_jobs            : int — number of parallel jobs (1 = no parallelization,
                        -1 = use all CPUs)

    Returns
    -------
    GeoDataFrame with columns:
        - All original columns from admin_gdf
        - pop_total : total population in admin unit
        - pct_within_{n}km : for each threshold n in thresholds_km
        - clinic_count_col : if clinic_counts provided
    
    Example
    -------
    admin2_curves = calc_pct_within_multiple_thresholds(
        distance_raster   = dist_raster_adm2,
        population_raster = pop_raster,
        admin_raster      = admin2_raster['data'],
        admin_gdf         = admin2,
        admin_code_col    = 'ADM2_CODE',
        thresholds_km     = [1, 2, 5, 10, 20, 50, 100],
        clinic_counts     = n_clinics_per_adm2,
        n_jobs            = -1,  # Use all CPUs
    )
    # Now has columns: pct_within_1km, pct_within_2km, ..., pct_within_100km
    """
    
    
    # Normalize inputs
    dist_2d = (distance_raster['data'][0]
               if isinstance(distance_raster, dict)
               else (distance_raster[0] if distance_raster.ndim == 3
                     else distance_raster))
    pop_2d  = (population_raster['data'][0]
               if isinstance(population_raster, dict)
               else (population_raster[0] if population_raster.ndim == 3
                     else population_raster))
    adm_2d  = admin_raster[0] if admin_raster.ndim == 3 else admin_raster

    result = admin_gdf.copy()
    result['pop_total'] = np.nan
    
    # Create columns for each threshold
    for thresh in thresholds_km:
        result[f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = np.nan

    if n_jobs == 1:
        # Serial version (original code)
        for idx, row in result.iterrows():
            _process_single_admin(idx, row, admin_code_col, adm_2d, dist_2d, 
                                 pop_2d, thresholds_km, result)
    else:
        # Parallel version
        from joblib import Parallel, delayed
        
        # Process all admin regions in parallel
        results_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_single_admin_parallel)(
                row[admin_code_col], adm_2d, dist_2d, pop_2d, thresholds_km
            )
            for _, row in result.iterrows()
        )
        
        # Merge results back into dataframe
        for idx, admin_results in enumerate(results_list):
            if admin_results is not None:
                for col, val in admin_results.items():
                    result.at[idx, col] = val

    if clinic_counts is not None:
        result[clinic_count_col] = result[admin_code_col].map(clinic_counts).fillna(0).astype(int)

    return result


def _process_single_admin(idx, row, admin_code_col, adm_2d, dist_2d, 
                          pop_2d, thresholds_km, result):
    """Helper function for serial processing of a single admin region."""
    code = row[admin_code_col]
    mask = adm_2d == code

    if not mask.any():
        return

    # Get all population and distances for this admin region
    pop = pop_2d[mask].astype(float)
    pop = np.where(np.isfinite(pop), pop, 0.0)
    pop_total = pop.sum()
    result.at[idx, 'pop_total'] = pop_total
    
    if pop_total == 0:
        # No population → all percentages are NaN
        return

    dist = dist_2d[mask]
    
    # Filter to valid distances (not NaN, not np.inf)
    valid_mask = np.isfinite(dist) & ~np.isinf(dist)
    
    if not valid_mask.any():
        # No valid distances → 0% for all thresholds
        for thresh in thresholds_km:
            result.at[idx, f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = 0.0
        return
    
    # For each threshold, calculate % within
    pop_valid = pop[valid_mask]
    dist_valid = dist[valid_mask]
    
    for thresh in thresholds_km:
        within_mask = dist_valid <= thresh
        pop_within = pop_valid[within_mask].sum()
        result.at[idx, f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = (pop_within / pop_total * 100)


def _process_single_admin_parallel(code, adm_2d, dist_2d, pop_2d, thresholds_km):
    """
    Helper function for parallel processing of a single admin region.
    Returns a dict of results instead of modifying result in-place.
    """
    mask = adm_2d == code

    if not mask.any():
        return None

    # Get all population and distances for this admin region
    pop = pop_2d[mask].astype(float)
    pop = np.where(np.isfinite(pop), pop, 0.0)
    pop_total = pop.sum()
    
    results = {'pop_total': pop_total}
    
    if pop_total == 0:
        # No population → all percentages are NaN
        for thresh in thresholds_km:
            results[f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = np.nan
        return results

    dist = dist_2d[mask]
    
    # Filter to valid distances (not NaN, not np.inf)
    valid_mask = np.isfinite(dist) & ~np.isinf(dist)
    
    if not valid_mask.any():
        # No valid distances → 0% for all thresholds
        for thresh in thresholds_km:
            results[f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = 0.0
        return results
    
    # For each threshold, calculate % within
    pop_valid = pop[valid_mask]
    dist_valid = dist[valid_mask]
    
    for thresh in thresholds_km:
        within_mask = dist_valid <= thresh
        pop_within = pop_valid[within_mask].sum()
        results[f'pct_within_{thresh:.1f}km'.replace('.', 'p')] = (pop_within / pop_total * 100)
    
    return results


def calc_pct_within_threshold(
    distance_raster,
    population_raster,
    admin_raster,
    admin_gdf,
    admin_code_col,
    threshold_km,
    clinic_counts=None,
    clinic_count_col='n_clinics',
):
    # Normalise inputs
    dist_2d = (distance_raster['data'][0]
               if isinstance(distance_raster, dict)
               else (distance_raster[0] if distance_raster.ndim == 3
                     else distance_raster))
    pop_2d  = (population_raster['data'][0]
               if isinstance(population_raster, dict)
               else (population_raster[0] if population_raster.ndim == 3
                     else population_raster))
    adm_2d  = admin_raster[0] if admin_raster.ndim == 3 else admin_raster

    pct_col = f'pct_within_{threshold_km:.1f}km'.replace('.', 'p')

    # ── Single pass over all pixels ───────────────────────────────────────────
    # Flatten everything to 1D
    codes = adm_2d.ravel()
    dist  = dist_2d.ravel().astype(float)
    pop   = pop_2d.ravel().astype(float)

    # Valid pixels: inside an admin unit (code > 0), finite pop, valid distance
    valid = (codes > 0) & np.isfinite(pop) & np.isfinite(dist) & ~np.isinf(dist)

    codes_v = codes[valid]
    dist_v  = dist[valid]
    pop_v   = pop[valid]

    # Build a DataFrame and groupby admin code — one pass, all units at once
    df = pd.DataFrame({
        'code':   codes_v,
        'pop':    pop_v,
        'within': pop_v * (dist_v <= threshold_km),
    })
    agg = df.groupby('code')[['pop', 'within']].sum()
    agg.columns = ['pop_total', 'pop_within']
    agg[pct_col] = np.where(
        agg['pop_total'] > 0,
        agg['pop_within'] / agg['pop_total'] * 100,
        np.nan
    )

    # ── Merge back onto admin_gdf ─────────────────────────────────────────────
    result = admin_gdf.copy()
    result = result.merge(
        agg[[pct_col, 'pop_total', 'pop_within']],
        left_on  = admin_code_col,
        right_index = True,
        how      = 'left',
    )

    if clinic_counts is not None:
        result[clinic_count_col] = (result[admin_code_col]
                                    .map(clinic_counts)
                                    .fillna(0)
                                    .astype(int))
    return result


# =============================================================================
# I/O
# =============================================================================

def save_raster(raster_data, transform, crs, filepath,
                dtype='float32', nodata=np.nan):
    """
    Write a (bands, H, W) raster to a compressed GeoTIFF.

    Parameters
    ----------
    raster_data : (bands, H, W) ndarray
    transform   : rasterio Affine transform
    crs         : CRS (anything rasterio accepts)
    filepath    : Path or str
    dtype       : output dtype string (default 'float32')
    nodata      : nodata value
    """
    if raster_data.ndim != 3:
        raise ValueError("raster_data must be 3D (bands, height, width)")

    meta = {
        'driver':    'GTiff',
        'height':    raster_data.shape[1],
        'width':     raster_data.shape[2],
        'count':     raster_data.shape[0],
        'dtype':     dtype,
        'crs':       crs,
        'transform': transform,
        'compress':  'lzw',
        'nodata':    nodata,
    }
    with rasterio.open(filepath, 'w', **meta) as dst:
        dst.write(raster_data)
    print(f"  Saved: {Path(filepath).name}")
