#!/usr/bin/env python3
"""
Create comprehensive distance-to-clinic coverage table for Pakistan.

Calculates:
- % population within various distance thresholds
- Population-weighted mean distance
- Population-weighted median distance

Runs for both 'latlon' and 'ea' projections.
"""

import sys
sys.path.insert(0, '../../src')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from joblib import Parallel, delayed

from idd_spatial_analysis import constants as rfc
from idd_spatial_analysis.shapefile_functions import load_shapefiles
from idd_spatial_analysis.raster_functions import calc_pct_within_multiple_thresholds


def load_raster(path):
    """Load a GeoTIFF into {'data': (bands,H,W), 'meta': {...}}."""
    with rasterio.open(path) as src:
        return {
            'data': src.read(),
            'meta': {
                'transform': src.transform,
                'crs': src.crs,
                'height': src.height,
                'width': src.width,
                'nodata': src.nodata,
            }
        }


def calc_weighted_distance_stats(admin_code, distance_data, population_data, 
                                 admin_raster_data, distance_nodata, pop_nodata):
    """
    Calculate population-weighted mean and median distance for a single admin unit.
    
    Returns:
        tuple: (admin_code, weighted_mean, weighted_median)
    """
    # Get mask for this admin unit
    mask = admin_raster_data == admin_code
    
    # Get distance and population values
    dist_vals = distance_data[mask]
    pop_vals = population_data[mask]
    
    # Filter out nodata values
    valid = np.ones(len(dist_vals), dtype=bool)
    if distance_nodata is not None:
        valid &= (dist_vals != distance_nodata)
    if pop_nodata is not None:
        valid &= (pop_vals != pop_nodata)
    valid &= (pop_vals > 0)  # Only consider pixels with population
    
    if not valid.any():
        return (admin_code, np.nan, np.nan)
    
    dist_vals = dist_vals[valid]
    pop_vals = pop_vals[valid]
    
    # Calculate weighted mean
    weighted_mean = np.sum(dist_vals * pop_vals) / np.sum(pop_vals)
    
    # Calculate weighted median
    # Sort by distance
    sort_idx = np.argsort(dist_vals)
    sorted_dist = dist_vals[sort_idx]
    sorted_pop = pop_vals[sort_idx]
    
    # Calculate cumulative population
    cumsum_pop = np.cumsum(sorted_pop)
    total_pop = cumsum_pop[-1]
    
    # Find median (50th percentile)
    median_idx = np.searchsorted(cumsum_pop, total_pop / 2.0)
    weighted_median = sorted_dist[median_idx]
    
    return (admin_code, weighted_mean, weighted_median)


def calc_all_weighted_stats(distance_raster, population_raster, admin_raster,
                            admin_codes, n_jobs=-1):
    """
    Calculate weighted statistics for all admin units in parallel.
    
    Returns:
        DataFrame with columns: admin_code, pop_weighted_mean_dist_km, pop_weighted_median_dist_km
    """
    # Extract 2D arrays (assuming single band rasters)
    dist_data = distance_raster['data'][0]
    pop_data = population_raster['data'][0]
    admin_data = admin_raster['data'][0]
    
    dist_nodata = distance_raster['meta']['nodata']
    pop_nodata = population_raster['meta']['nodata']
    
    print(f"  Calculating weighted statistics for {len(admin_codes)} admin units...")
    
    # Parallel calculation
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(calc_weighted_distance_stats)(
            code, dist_data, pop_data, admin_data, dist_nodata, pop_nodata
        )
        for code in admin_codes
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['admin_code', 'pop_weighted_mean_dist_km', 'pop_weighted_median_dist_km'])
    
    return df


def process_projection(proj_name):
    """Process a single projection and create coverage tables."""
    
    print("\n" + "="*80)
    print(f"Processing projection: {proj_name}")
    print("="*80)
    
    # Configuration
    projection_dict = rfc.ALL_PROJECTION_DICT
    CRS = projection_dict[proj_name]['crs']
    ADM0_CODE = 173  # Pakistan
    
    ANALYSIS_DIR = rfc.MODEL_ROOT / 'analyses' / 'pakistan_denv'
    OUTPUT_DIR = ANALYSIS_DIR / 'outputs'
    
    # ── Load Admin Shapefiles ─────────────────────────────────────────────
    print('\n1. Loading admin shapefiles...')
    admin_gdfs = load_shapefiles(
        ADM0_CODE_LIST=[ADM0_CODE],
        depth=2,
        simple=False,
        projection_dict={proj_name: projection_dict[proj_name]},
    )
    
    admin0 = admin_gdfs[('admin0', proj_name)]['gdf']
    admin1 = admin_gdfs[('admin1', proj_name)]['gdf']
    admin2 = admin_gdfs[('admin2', proj_name)]['gdf']
    print(f'  admin0: {len(admin0)} | admin1: {len(admin1)} | admin2: {len(admin2)}')
    
    # ── Load Clinics ──────────────────────────────────────────────────────
    print('\n2. Loading clinics...')
    clinics = gpd.read_file(OUTPUT_DIR / 'clinics_latlon.gpkg')
    if proj_name != 'latlon':
        clinics = clinics.to_crs(CRS)
    print(f'  {len(clinics)} clinics loaded')
    
    # Count clinics per admin2
    n_clinics_per_adm2 = (
        clinics
        .groupby('ADM2_CODE_gps_adm2')
        .size()
        .rename('n_clinics')
    )
    
    # ── Load Rasters ──────────────────────────────────────────────────────
    print('\n3. Loading rasters...')
    pop_raster = load_raster(OUTPUT_DIR / f'population_clipped_{proj_name}.tif')
    dist_raster_adm2 = load_raster(OUTPUT_DIR / f'dist_nearest_clinic_admin2_{proj_name}_km.tif')
    admin2_raster = load_raster(OUTPUT_DIR / f'admin2_raster_{proj_name}.tif')
    print(f'  pop_raster       : {pop_raster["data"].shape}')
    print(f'  dist_raster_adm2 : {dist_raster_adm2["data"].shape}')
    print(f'  admin2_raster    : {admin2_raster["data"].shape}')
    
    # ── Define Distance Thresholds ────────────────────────────────────────
    print('\n4. Defining distance thresholds...')
    thresholds_km = (
        # 0.5 to 20 by 0.5
        list(np.arange(0.5, 20.5, 0.5)) +
        # 30, 40, 50, 60, 70, 80, 90, 100
        [30, 40, 50, 60, 70, 80, 90, 100]
    )
    print(f'  {len(thresholds_km)} thresholds: {thresholds_km[:5]} ... {thresholds_km[-5:]}')
    
    # ── Calculate Coverage Percentages ────────────────────────────────────
    print('\n5. Calculating coverage percentages...')
    admin2_coverage = calc_pct_within_multiple_thresholds(
        distance_raster=dist_raster_adm2,
        population_raster=pop_raster,
        admin_raster=admin2_raster['data'],
        admin_gdf=admin2,
        admin_code_col='ADM2_CODE',
        thresholds_km=thresholds_km,
        clinic_counts=n_clinics_per_adm2,
        clinic_count_col='n_clinics',
        n_jobs=-1,  # Use all CPUs
    )
    print(f'  Complete! {len(admin2_coverage)} admin2 regions processed')
    
    # ── Calculate Weighted Distance Statistics ───────────────────────────
    print('\n6. Calculating population-weighted distance statistics...')
    admin_codes = admin2['ADM2_CODE'].values
    weighted_stats = calc_all_weighted_stats(
        distance_raster=dist_raster_adm2,
        population_raster=pop_raster,
        admin_raster=admin2_raster,
        admin_codes=admin_codes,
        n_jobs=-1,
    )
    
    # Merge weighted stats into main dataframe
    admin2_coverage = admin2_coverage.merge(
        weighted_stats, 
        left_on='ADM2_CODE', 
        right_on='admin_code',
        how='left'
    ).drop(columns=['admin_code'])
    
    print(f'  Weighted statistics calculated for {(~admin2_coverage["pop_weighted_mean_dist_km"].isna()).sum()} admin2 regions')
    
    # ── Create Output DataFrame ───────────────────────────────────────────
    print('\n7. Creating output dataframe...')
    
    # Get all percentage columns
    pct_cols = sorted([col for col in admin2_coverage.columns if col.startswith('pct_within_')])
    
    output_df = pd.DataFrame()
    
    # Add administrative information
    output_df['ADM1_NAME'] = admin2_coverage['ADM1_NAME']
    output_df['ADM2_CODE'] = admin2_coverage['ADM2_CODE']
    output_df['ADM2_NAME'] = admin2_coverage['ADM2_NAME']
    output_df['population'] = admin2_coverage['pop_total']
    output_df['n_clinics'] = admin2_coverage['n_clinics']
    
    # Add weighted distance statistics
    output_df['pop_weighted_mean_dist_km'] = admin2_coverage['pop_weighted_mean_dist_km']
    output_df['pop_weighted_median_dist_km'] = admin2_coverage['pop_weighted_median_dist_km']
    
    # Add all the percentage columns
    for col in pct_cols:
        output_df[col] = admin2_coverage[col]
    
    # ── Display Summary Statistics ────────────────────────────────────────
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Total population: {output_df['population'].sum():,.0f}")
    print(f"Regions with clinics: {(output_df['n_clinics'] > 0).sum()} / {len(output_df)}")
    print(f"Regions with 0 clinics: {(output_df['n_clinics'] == 0).sum()}")
    print(f"Regions with <5 clinics: {(output_df['n_clinics'] < 5).sum()}")
    print(f"Regions with ≥5 clinics: {(output_df['n_clinics'] >= 5).sum()}")
    
    print("\nPopulation-weighted distance statistics (median across regions):")
    print(f"  Mean distance:   {output_df['pop_weighted_mean_dist_km'].median():.2f} km")
    print(f"  Median distance: {output_df['pop_weighted_median_dist_km'].median():.2f} km")
    
    print("\nCoverage at selected thresholds (median across regions):")
    for thresh in [1, 2, 5, 10, 20, 50, 100]:
        col_name = f'pct_within_{thresh:.1f}km'.replace('.', 'p')
        if col_name in output_df.columns:
            median_pct = output_df[col_name].median()
            print(f"  {thresh:3.0f} km: {median_pct:5.1f}%")
    
    # ── Save Admin2 Output ────────────────────────────────────────────────
    print('\n8. Saving outputs...')
    output_path = OUTPUT_DIR / f'admin2_clinic_coverage_by_distance_{proj_name}.csv'
    output_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f'  ✓ Saved Admin2 table: {output_path}')
    
    # ── Create and Save Admin1 Summary ────────────────────────────────────
    admin1_summary = output_df.groupby('ADM1_NAME').agg({
        'ADM2_CODE': 'count',
        'population': 'sum',
        'n_clinics': 'sum',
        'pop_weighted_mean_dist_km': 'mean',
        'pop_weighted_median_dist_km': 'mean',
        **{col: 'mean' for col in pct_cols}
    }).rename(columns={'ADM2_CODE': 'n_admin2_regions'})
    
    admin1_output_path = OUTPUT_DIR / f'admin1_clinic_coverage_summary_{proj_name}.csv'
    admin1_summary.to_csv(admin1_output_path, float_format='%.2f')
    print(f'  ✓ Saved Admin1 summary: {admin1_output_path}')
    
    print("\n" + "="*80)
    print(f"Processing complete for {proj_name}!")
    print("="*80)
    
    return output_df, admin1_summary


def main():
    """Main entry point."""
    
    print("="*80)
    print("Pakistan Clinic Coverage Table Generator")
    print("="*80)
    
    # Process both projections
    projections = ['latlon', 'ea']
    results = {}
    
    for proj_name in projections:
        try:
            admin2_df, admin1_df = process_projection(proj_name)
            results[proj_name] = {
                'admin2': admin2_df,
                'admin1': admin1_df,
            }
        except Exception as e:
            print(f"\n❌ Error processing {proj_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("All Projections Complete!")
    print("="*80)
    
    for proj_name in projections:
        if proj_name in results:
            print(f"\n{proj_name.upper()}:")
            print(f"  Admin2 regions: {len(results[proj_name]['admin2'])}")
            print(f"  Admin1 regions: {len(results[proj_name]['admin1'])}")
    
    print("\n✓ All coverage tables generated successfully!")


if __name__ == '__main__':
    main()
