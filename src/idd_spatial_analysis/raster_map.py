"""
raster_map.py — Single-panel raster map
Replaces the old plot_country_raster() function.

Key differences from the old code:
  - No reprojection inside the function; all data must already be in the same CRS
  - Explicit inch-based layout; never uses bbox_inches='tight'
  - Neighbor countries loaded from a global admin0 shapefile (not passed as GDFs)
  - Water drawn from pre-loaded GDFs (ocean_gdf, lakes_gdf, rivers_gdf)
  - Population masking (zero-pop → grey) is an explicit config toggle
  - Admin1 spatial masking is an explicit config toggle
"""

import numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.colors as mcolors

from idd_spatial_analysis.base_config import SharedConfig
from idd_spatial_analysis.map_layout import (single_panel_layout, bbox_from_gdf,
                                 clip_to_bbox, load_neighbors, clip_raster)
from idd_spatial_analysis.map_layers import (
    draw_raster, draw_population_raster, draw_binary_raster,
    draw_neighbors, draw_ocean, draw_lakes, draw_rivers,
    draw_admin_boundaries, draw_focus_boundary, draw_clinics,
    draw_figure_legend, build_bin_legend_handles,
    set_extent, style_ax, draw_latlon_grid,
)


# ── Config ────────────────────────────────────────────────────────────────────

class RasterMapConfig(SharedConfig):
    """Configuration for the single-panel raster map."""

    # ── Render mode ───────────────────────────────────────────────────────────
    # 'full'           everything (raster, shapefiles, clinics, legend)
    # 'shapefiles'     just shapefiles (no raster, no clinics) 
    # 'windows_legend' just legend
    render_mode: str = 'full'

    # ── Raster type ───────────────────────────────────────────────────────────
    # 'distance'    standard distance raster (nodata → transparent)
    # 'population'  population density (zero-pop → grey, nodata → transparent)
    # 'binary'      Y/N within threshold_km of a clinic
    raster_type: str = 'distance'

    # For raster_type='binary'
    binary_threshold_km:   float = 5.0
    binary_within_color:   str   = '#1a9641'
    binary_within_alpha:   float = 0.7
    binary_outside_color:  str   = '#d73027'
    binary_outside_alpha:  float = 0.5

    # For handling np.inf "no clinics" pixels
    show_no_clinic_bin:    bool  = True   # Add separate bin for np.inf pixels

    # For raster_type='population'
    zero_pop_color:       str   = '#cccccc'
    zero_pop_alpha:       float = 0.35
    zero_pop_label:       str   = 'Unpopulated'
    show_zero_pop_legend: bool  = False   # Add legend entry for zero-pop pixels
    raster_alpha:         float = 0.85    # overall raster transparency

    # ── Masking ───────────────────────────────────────────────────────────────
    # If True, pixels where population == 0 are masked (set to nodata)
    # before binning/display. Applies to distance and binary rasters.
    mask_zero_population:   bool = False
    # If True, pixels outside focal_admin1 (identified by admin1_code_col /
    # focal_admin1_code) are masked. Requires admin1_raster to be passed in.
    mask_outside_admin1:    bool = False
    admin1_code_col:        str  = 'ADM1_CODE'
    focal_admin1_code              = None   # int/str code matching admin1_raster values
    # If True, pixels outside focal_admin2 (identified by admin2_code_col /
    # focal_admin2_code) are masked. Requires admin2_raster to be passed in.
    mask_outside_admin2:    bool = False
    admin2_code_col:        str  = 'ADM2_CODE'
    focal_admin2_code              = None   # int/str code matching admin2_raster values

    # ── Extent ────────────────────────────────────────────────────────────────
    # 'admin0'  full country
    # 'admin1'  single admin1 region  (requires focal_admin1_name)
    # 'admin2'  single admin2 region  (requires focal_admin2_name)
    extent_type:  str = 'admin0'
    # Name fields
    focal_admin1_name: str = ''; admin1_name_field: str = 'ADM1_NAME'
    focal_admin2_name: str = ''; admin2_name_field: str = 'ADM2_NAME'
    # Padding around extent in CRS units
    extent_padding: float = 10_000   # metres (or degrees for latlon)

    # ── Figure geometry (inches) ──────────────────────────────────────────────
    figure_width_in:  float = 14.0
    figure_height_in: float = 10.0
    top_margin_in:    float = 0.45
    legend_height_in: float = 0.90
    left_margin_in:   float = 0.10
    right_margin_in:  float = 0.10


def _make_cmap(edges, colors):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([e if np.isfinite(e) else 1e9 for e in edges],
                                 ncolors=len(colors))
    return cmap, norm


def _get_extent_bbox(c, admin0_gdf, admin1_gdf, admin2_gdf):
    """Return the geographic bbox for the requested extent type."""
    if c.extent_type == 'admin0':
        gdf = admin0_gdf
    elif c.extent_type == 'admin1':
        gdf = admin1_gdf[admin1_gdf[c.admin1_name_field] == c.focal_admin1_name]
        if gdf.empty:
            raise ValueError(f"Admin1 '{c.focal_admin1_name}' not found.")
    elif c.extent_type == 'admin2':
        gdf = admin2_gdf[admin2_gdf[c.admin2_name_field] == c.focal_admin2_name]
        if gdf.empty:
            raise ValueError(f"Admin2 '{c.focal_admin2_name}' not found.")
    else:
        raise ValueError(f"extent_type must be 'admin0','admin1','admin2'")
    return bbox_from_gdf(gdf, buffer=c.extent_padding), gdf


# ── Main function ─────────────────────────────────────────────────────────────

def make_raster_map(
    distance_raster,        # {'data':(1,H,W), 'meta':{transform,crs,...}}
    admin0_gdf,
    admin1_gdf,
    admin2_gdf,
    clinics_gdf,
    population_raster=None, # {'data':(1,H,W), 'meta':{...}}  for masking or pop display
    admin1_raster=None,     # (H,W) int array of admin1 codes  for spatial masking
    admin2_raster=None,     # (H,W) int array of admin2 codes  for spatial masking
    ocean_gdf=None,
    lakes_gdf=None,
    rivers_gdf=None,
    config=None,
):
    """
    Single-panel raster map.

    All GeoDataFrames, rasters, and water GDFs must be in the SAME CRS.
    Reprojection must be done before calling this function.

    Parameters
    ----------
    distance_raster   : dict  {'data': (1,H,W) ndarray km values, 'meta': {...}}
    admin0/1/2_gdf    : GeoDataFrame
    clinics_gdf       : GeoDataFrame
    population_raster : dict or None — required for mask_zero_population=True
                        or raster_type='population'
    admin1_raster     : dict or (H,W) ndarray or None — required for mask_outside_admin1=True
                        If dict, expects {'data': (1,H,W), 'meta': {...}}
    admin2_raster     : dict or (H,W) ndarray or None — required for mask_outside_admin2=True
                        If dict, expects {'data': (1,H,W), 'meta': {...}}
    ocean/lakes/rivers_gdf : GeoDataFrame or None — pre-loaded natural earth water

    Returns
    -------
    fig, ax
    """
    c = config or RasterMapConfig()
    matplotlib.rcParams.update({'font.size': c.base_fontsize})

    _draw_r  = c.render_mode == 'full'
    _draw_sf = c.render_mode in {'full', 'shapefiles'}
    _draw_cl = c.render_mode == 'full'
    _draw_lg = c.render_mode in {'full', 'windows_legend'}

    cmap, norm = _make_cmap(c.bin_edges, c.bin_colors)

    # Extent bbox
    extent_bbox, extent_gdf = _get_extent_bbox(c, admin0_gdf, admin1_gdf, admin2_gdf)
    crs = admin0_gdf.crs

    # Layout
    L = single_panel_layout(c)
    print(f"Figure: {L['figsize'][0]:.2f}×{L['figsize'][1]:.2f} in")

    # Prepare raster data
    raster_data = None
    raster_transform = None
    if _draw_r:
        src = distance_raster['data'].copy()
        
        # Convert nodata values to NaN (but preserve np.inf for "no clinics")
        nodata = distance_raster['meta'].get('nodata')
        if nodata is not None and np.isfinite(nodata) and nodata != np.inf:
            src[0][src[0] == nodata] = np.nan

        # Population masking
        if c.mask_zero_population and population_raster is not None:
            pop = population_raster['data'][0]
            src[0][pop == 0] = np.nan

        # Admin1 spatial masking
        if c.mask_outside_admin1 and admin1_raster is not None and c.focal_admin1_code is not None:
            admin1_data = admin1_raster['data'][0] if isinstance(admin1_raster, dict) else admin1_raster
            src[0][admin1_data != c.focal_admin1_code] = np.nan

        # Admin2 spatial masking
        if c.mask_outside_admin2 and admin2_raster is not None and c.focal_admin2_code is not None:
            admin2_data = admin2_raster['data'][0] if isinstance(admin2_raster, dict) else admin2_raster
            src[0][admin2_data != c.focal_admin2_code] = np.nan

        raster_data, raster_transform = clip_raster(
            src, distance_raster['meta'], extent_bbox)

    # Clip shapefiles to extent
    a0 = clip_to_bbox(admin0_gdf, extent_bbox) if _draw_sf else None
    a1 = clip_to_bbox(admin1_gdf, extent_bbox) if _draw_sf else None
    a2 = clip_to_bbox(admin2_gdf, extent_bbox) if _draw_sf else None
    cl = clip_to_bbox(clinics_gdf, extent_bbox)

    # Neighbors from global shapefile
    nb = load_neighbors(c.neighbor_admin0_path, extent_bbox,
                        c.focus_adm0_code, c.adm0_code_field, crs) if _draw_sf else None

    # Build figure
    fig = plt.figure(figsize=L['figsize'], facecolor='white')
    ax  = fig.add_axes(L['main_panel_rect'])
    ax.set_facecolor(c.axes_facecolor)

    if _draw_sf:
        draw_neighbors(ax, nb, c)
        draw_ocean(ax, ocean_gdf, extent_bbox, c)
        draw_lakes(ax, lakes_gdf, extent_bbox, c)
        draw_rivers(ax, rivers_gdf, extent_bbox, c)

    # Extract admin0 mask from population raster (NaN = outside country)
    admin0_mask = None
    if population_raster is not None:
        pop_clipped, _ = clip_raster(
            population_raster['data'], population_raster['meta'], extent_bbox)
        # Create mask: True = inside country (finite values), False = outside (NaN)
        admin0_mask = np.isfinite(pop_clipped[0])

    if _draw_r and raster_data is not None:
        if c.raster_type == 'population' and population_raster is not None:
            pop_clipped, pop_transform = clip_raster(
                population_raster['data'], population_raster['meta'], extent_bbox)
            draw_population_raster(ax, pop_clipped, pop_transform, cmap, norm,
                                    alpha=c.raster_alpha,
                                    zero_pop_color=c.zero_pop_color,
                                    zero_pop_alpha=c.zero_pop_alpha,
                                    admin0_mask=admin0_mask)
        elif c.raster_type == 'binary':
            draw_binary_raster(ax, raster_data, raster_transform,
                                threshold_km=c.binary_threshold_km,
                                within_color=c.binary_within_color,
                                within_alpha=c.binary_within_alpha,
                                outside_color=c.binary_outside_color,
                                outside_alpha=c.binary_outside_alpha)
        else:
            draw_raster(ax, raster_data, raster_transform, cmap, norm,
                        alpha=c.raster_alpha,
                        no_clinic_color=c.no_clinic_color,
                        no_clinic_alpha=c.no_clinic_alpha,
                        show_no_clinic_bin=c.show_no_clinic_bin,
                        admin0_mask=admin0_mask)

    if _draw_sf:
        if c.show_admin0_boundaries:
            draw_admin_boundaries(ax, a0, c.admin0_lw, c.admin0_color, 4)
        if c.show_admin1_boundaries:
            draw_admin_boundaries(ax, a1, c.admin1_lw, c.admin1_color, 5)
        if c.show_admin2_boundaries:
            draw_admin_boundaries(ax, a2, c.admin2_lw, c.admin2_color, 6)

    # Focus boundary: only drawn for admin1/admin2 extent
    if _draw_sf and c.extent_type in {'admin1', 'admin2'}:
        draw_focus_boundary(ax, clip_to_bbox(extent_gdf, extent_bbox), c, zorder=7)

    if _draw_cl:
        draw_clinics(ax, cl, c)

    set_extent(ax, extent_bbox, enforce_aspect=True)
    style_ax(ax, '',  # Title drawn separately in top margin
             c.title_fontsize, border_lw=0.8,
             hide_ticks=not c.show_latlon_grid)

    # Title lives in the top margin
    fig.text(0.5, 1.0 - (c.top_margin_in / 2) / L['figsize'][1],
             getattr(c, 'map_title', ''),
             ha='center', va='center',
             fontsize=c.title_fontsize, fontweight='bold')

    draw_latlon_grid(ax, c)

    if _draw_lg:
        extra = []
        if c.raster_type == 'binary':
            import matplotlib.patches as mpatches
            extra = [
                mpatches.Patch(facecolor=c.binary_within_color,
                               alpha=c.binary_within_alpha,
                               label=f'Within {c.binary_threshold_km} km'),
                mpatches.Patch(facecolor=c.binary_outside_color,
                               alpha=c.binary_outside_alpha,
                               label=f'Beyond {c.binary_threshold_km} km'),
            ]
            draw_figure_legend(fig, [], c, L['legend_y_frac'], extra_handles=extra)
        else:
            handles = build_bin_legend_handles(c)
            # Add "no clinics" bin if enabled
            if c.show_no_clinic_bin and c.raster_type == 'distance':
                import matplotlib.patches as mpatches
                label = c.no_clinic_label if c.no_clinic_label else 'No clinics in district'
                handles.append(
                    mpatches.Patch(facecolor=c.no_clinic_color,
                                   alpha=c.no_clinic_alpha,
                                   label=label))
            draw_figure_legend(fig, handles, c, L['legend_y_frac'])

    if c.save_figure:
        fig.savefig(c.output_path, dpi=c.dpi, facecolor='white')
        print(f"Saved: {c.output_path}")
    if c.show_figure: plt.show()
    else: plt.close(fig)
    return fig, ax


# ── Pre-built configs for your two existing maps ──────────────────────────────

def population_country_config(output_path, pop_bins, pop_colors, pop_labels):
    """
    Config equivalent to:
        plot_country_raster(raster_type='population', extent_type='admin0',
                            show_admin2_boundaries=True, ...)

    Usage
    -----
        cfg = population_country_config(OUTPUT_DIR / 'population_country.png',
                                        pop_bins, pop_colors, pop_labels)
        fig, ax = make_raster_map(
            distance_raster    = {'data': clipped_data_2d[np.newaxis],
                                  'meta': {'transform': clipped_transform,
                                           'crs': your_crs,
                                           'height': clipped_data_2d.shape[0],
                                           'width':  clipped_data_2d.shape[1]}},
            admin0_gdf         = admin0_gdf,
            admin1_gdf         = admin1_gdf,
            admin2_gdf         = admin2_gdf,
            clinics_gdf        = clinics_ea_gdf,
            population_raster  = {'data': clipped_data_2d[np.newaxis], 'meta': ...},
            config             = cfg,
        )
    """
    c = RasterMapConfig()
    c.raster_type          = 'population'
    c.extent_type          = 'admin0'
    c.extent_padding       = 50_000     # ~50 km padding around country

    c.bin_edges            = pop_bins
    c.bin_colors           = pop_colors
    c.bin_labels           = pop_labels
    c.legend_title         = 'Population'
    c.legend_ncol          = 8

    c.show_admin0_boundaries = True
    c.show_admin1_boundaries = False    # not shown for country-level pop map
    c.show_admin2_boundaries = True
    c.admin2_lw              = 0.25
    c.show_focus_boundary    = False    # no focal region at admin0 level

    c.focus_boundary_color = 'black'   # matches old focus_color='black'
    c.show_clinics         = True
    c.clinic_size          = 5

    c.figure_width_in  = 15.0
    c.figure_height_in = 12.0

    c.map_title    = 'Pakistan Population Density (100m² resolution)'
    c.output_path  = str(output_path)
    c.save_figure  = True
    c.show_figure  = False
    return c


def distance_admin1_country_config(output_path, dist_bins, dist_colors, dist_labels):
    """
    Config equivalent to:
        plot_country_raster(raster_type='distance', extent_type='admin0',
                            mask_outside_admin1=True, show_admin1=True, ...)

    Usage
    -----
        cfg = distance_admin1_country_config(OUTPUT_DIR / 'map_country_dist_adm1.png',
                                             dist_bins, dist_colors, dist_labels)
        # Set the admin1 code to mask against:
        cfg.mask_outside_admin1 = True
        cfg.focal_admin1_code   = YOUR_ADMIN1_CODE   # int matching admin1_raster values

        fig, ax = make_raster_map(
            distance_raster  = {'data': dist_raster[np.newaxis], 'meta': ...},
            admin0_gdf       = admin0_gdf,
            admin1_gdf       = admin1_gdf,
            admin2_gdf       = admin2_gdf,
            clinics_gdf      = clinics_ea_gdf,
            population_raster= {'data': population_2d[np.newaxis], 'meta': ...},
            admin1_raster    = admin1_raster,   # (H,W) array of admin1 codes
            config           = cfg,
        )
    """
    c = RasterMapConfig()
    c.raster_type          = 'distance'
    c.extent_type          = 'admin0'
    c.extent_padding       = 50_000

    c.bin_edges            = dist_bins
    c.bin_colors           = dist_colors
    c.bin_labels           = dist_labels
    c.legend_title         = 'Distance (km)'

    c.mask_zero_population = True       # grey out zero-pop pixels
    c.mask_outside_admin1  = False      # set True + focal_admin1_code to activate
    c.raster_alpha         = 0.85

    c.show_admin0_boundaries = True
    c.show_admin1_boundaries = True
    c.show_admin2_boundaries = False
    c.show_focus_boundary    = False

    c.show_clinics  = True
    c.clinic_color  = '#1f1f8a'
    c.clinic_size   = 5

    c.figure_width_in  = 15.0
    c.figure_height_in = 12.0

    c.map_title    = 'Distance to Nearest Clinic (Same Admin 1) - Pakistan'
    c.output_path  = str(output_path)
    c.save_figure  = True
    c.show_figure  = False
    return c
