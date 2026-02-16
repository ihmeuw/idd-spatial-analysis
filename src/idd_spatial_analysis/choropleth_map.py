"""
choropleth_map.py — Single-panel choropleth map

Fills admin polygons by a numeric value column. Designed for:
  - % of population within X km of a clinic (by admin2)
  - Any other per-admin aggregated metric

Clinic count threshold outlines
--------------------------------
Set clinic_count_col on your GDF and configure:
    c.clinic_count_col       = 'n_clinics'
    c.clinic_threshold       = 5
    c.few_clinics_lw         = 2.0   # bold outline for < threshold
    c.many_clinics_lw        = 0.4   # thin outline for >= threshold
"""

import numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from idd_spatial_analysis.base_config import SharedConfig
from idd_spatial_analysis.map_layout import (single_panel_layout, bbox_from_gdf,
                                 clip_to_bbox, load_neighbors)
from idd_spatial_analysis.map_layers import (
    draw_choropleth, draw_clinic_count_outlines,
    draw_neighbors, draw_ocean, draw_lakes, draw_rivers,
    draw_admin_boundaries, draw_focus_boundary, draw_clinics,
    draw_figure_legend, set_extent, style_ax, draw_latlon_grid,
)


# ── Config ────────────────────────────────────────────────────────────────────

class ChoroplethConfig(SharedConfig):

    # ── Render mode ───────────────────────────────────────────────────────────
    # 'full'           everything (choropleth, shapefiles, clinics, legend)
    # 'shapefiles'     just shapefiles (no choropleth, no clinics) 
    # 'windows_legend' just legend
    render_mode: str = 'full'

    # ── Choropleth data ───────────────────────────────────────────────────────
    value_col:   str = 'pct_within_5km'   # column in admin_gdf to colour by
    admin_level: int = 2                   # 1 or 2

    # Bins / colors / labels for the choropleth fill
    # Inherit from SharedConfig — override here as needed
    bin_edges  = [0, 20, 40, 60, 80, 100]
    bin_colors = ['#a50026','#f46d43','#ffffbf','#74add1','#313695']
    bin_labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    legend_title: str = '% population within 5 km of a clinic (same admin 2)'

    missing_color: str = '#e0e0e0'   # fill for admin units with no data

    # ── Clinic count visualization ────────────────────────────────────────────
    # Choose between outline-based or hatch-based visualization
    use_clinic_count_outlines: bool = False  # Use bold outlines (legacy)
    use_clinic_count_hatching: bool = True   # Use diagonal hatching (new default)
    
    clinic_count_col:     str  = 'n_clinics'
    clinic_threshold:     int  = 5
    
    # Outline settings (if use_clinic_count_outlines=True)
    few_clinics_lw:       float = 2.0
    few_clinics_color:    str   = '#111111'
    many_clinics_lw:      float = 0.35
    many_clinics_color:   str   = '#555555'
    
    # Hatching settings (if use_clinic_count_hatching=True)
    hatch_pattern:        str   = '///'     # '///', '\\\\\\', '|||', '---', 'xxx', etc.
    hatch_color:          str   = '#666666'
    hatch_lw:             float = 1.0
    
    # Color for regions with zero clinics
    show_no_clinic_legend: bool = True
    no_clinic_color: str = '#d9d9d9'
    missing_color: str = '#f0f0f0'   # fill for admin units with no data (different from no clinics)

    # ── Extent ────────────────────────────────────────────────────────────────
    extent_type:       str   = 'admin0'
    focal_admin1_name: str   = ''
    admin1_name_field: str   = 'ADM1_NAME'
    focal_admin2_name: str   = ''
    admin2_name_field: str   = 'ADM2_NAME'
    extent_padding:    float = 10_000

    # ── Sub-level boundary lines on top of fill ───────────────────────────────
    # (Separate from the clinic-count outlines)
    show_sublevel_lines: bool  = False   # e.g. admin2 lines inside admin1 choropleth
    sublevel_lw:         float = 0.2
    sublevel_color:      str   = '#888888'

    # ── Figure geometry (inches) ──────────────────────────────────────────────
    figure_width_in:  float = 14.0
    figure_height_in: float = 10.0
    top_margin_in:    float = 0.45
    legend_height_in: float = 0.90
    left_margin_in:   float = 0.10
    right_margin_in:  float = 0.10

    map_title: str = ''


def _make_cmap(edges, colors):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([e if np.isfinite(e) else 1e9 for e in edges],
                                 ncolors=len(colors))
    return cmap, norm


# ── Main function ─────────────────────────────────────────────────────────────
def make_choropleth_map(
    admin_gdf,          # GeoDataFrame with value_col (and optionally clinic_count_col)
    admin0_gdf,         # country boundary
    admin1_gdf,         # admin1 boundaries (for lines / extent lookup)
    admin2_gdf,         # admin2 boundaries (for lines / extent lookup)
    clinics_gdf=None,
    ocean_gdf=None,
    lakes_gdf=None,
    rivers_gdf=None,
    config=None,
):
    """
    Single-panel choropleth map.

    Parameters
    ----------
    admin_gdf    : GeoDataFrame
        The admin polygons to fill. Must contain config.value_col.
        Optionally contains config.clinic_count_col for threshold outlines/hatching.
    admin0/1/2_gdf : GeoDataFrame  boundary lines
    clinics_gdf  : GeoDataFrame or None
    ocean/lakes/rivers_gdf : GeoDataFrame or None — pre-loaded natural earth water

    Returns
    -------
    fig, ax
    """
    c = config or ChoroplethConfig()
    matplotlib.rcParams.update({'font.size': c.base_fontsize})

    _draw_sf = c.render_mode in {'full', 'shapefiles'}
    _draw_cl = c.render_mode == 'full'
    _draw_lg = c.render_mode in {'full', 'windows_legend'}
    _draw_ch = c.render_mode == 'full'

    cmap, norm = _make_cmap(c.bin_edges, c.bin_colors)

    # Extent bbox
    if c.extent_type == 'admin0':
        extent_gdf = admin0_gdf
    elif c.extent_type == 'admin1':
        extent_gdf = admin1_gdf[admin1_gdf[c.admin1_name_field] == c.focal_admin1_name]
        if extent_gdf.empty:
            raise ValueError(f"Admin1 '{c.focal_admin1_name}' not found.")
    elif c.extent_type == 'admin2':
        extent_gdf = admin2_gdf[admin2_gdf[c.admin2_name_field] == c.focal_admin2_name]
        if extent_gdf.empty:
            raise ValueError(f"Admin2 '{c.focal_admin2_name}' not found.")
    else:
        raise ValueError(f"Unknown extent_type '{c.extent_type}'")

    extent_bbox = bbox_from_gdf(extent_gdf, buffer=c.extent_padding)
    crs = admin0_gdf.crs

    L = single_panel_layout(c)
    print(f"Figure: {L['figsize'][0]:.2f}×{L['figsize'][1]:.2f} in")

    # Clip data to extent
    admin_clipped = clip_to_bbox(admin_gdf, extent_bbox)
    a0 = clip_to_bbox(admin0_gdf, extent_bbox) if _draw_sf else None
    a1 = clip_to_bbox(admin1_gdf, extent_bbox) if _draw_sf else None
    a2 = clip_to_bbox(admin2_gdf, extent_bbox) if _draw_sf else None
    cl = clip_to_bbox(clinics_gdf, extent_bbox) if clinics_gdf is not None else None
    nb = load_neighbors(c.neighbor_admin0_path, extent_bbox,
                        c.focus_adm0_code, c.adm0_code_field, crs) if _draw_sf else None

    # Build figure
    fig = plt.figure(figsize=L['figsize'], facecolor='white')
    ax  = fig.add_axes(L['main_panel_rect'])

    if _draw_sf:
        draw_neighbors(ax, nb, c)
        draw_ocean(ax, ocean_gdf, extent_bbox, c)
        draw_lakes(ax, lakes_gdf, extent_bbox, c)
        draw_rivers(ax, rivers_gdf, extent_bbox, c)

    if _draw_ch and not admin_clipped.empty:
        # Draw choropleth with optional hatching for few clinics
        draw_choropleth(
            ax, admin_clipped, c.value_col, cmap, norm,
            edgecolor='none', lw=0, zorder=3,
            missing_color=c.missing_color,
            clinic_count_col=c.clinic_count_col if c.use_clinic_count_hatching else None,
            clinic_threshold=c.clinic_threshold,
            hatch_few_clinics=c.use_clinic_count_hatching,
            hatch_pattern=c.hatch_pattern,
            hatch_color=c.hatch_color,
            hatch_lw=c.hatch_lw,
            no_clinic_color=c.no_clinic_color
        )

    # Clinic count threshold outlines (legacy mode - on top of fill, below admin lines)
    if (_draw_ch and c.use_clinic_count_outlines
            and c.clinic_count_col in admin_clipped.columns):
        draw_clinic_count_outlines(
            ax, admin_clipped, c.clinic_count_col,
            threshold=c.clinic_threshold,
            low_color=c.few_clinics_color,   low_lw=c.few_clinics_lw,
            high_color=c.many_clinics_color, high_lw=c.many_clinics_lw,
            zorder=4,
        )
    elif _draw_sf:
        # Draw admin boundaries for the choropleth level if not using clinic count outlines
        if c.admin_level == 2 and c.show_admin2_boundaries:
            draw_admin_boundaries(ax, admin_clipped, c.admin2_lw, c.admin2_color, 4)
        elif c.admin_level == 1:
            draw_admin_boundaries(ax, admin_clipped, c.admin1_lw, c.admin1_color, 4)

    if _draw_sf:
        if c.show_admin0_boundaries:
            draw_admin_boundaries(ax, a0, c.admin0_lw, c.admin0_color, 5)
        if c.show_admin1_boundaries and c.admin_level == 2:
            draw_admin_boundaries(ax, a1, c.admin1_lw, c.admin1_color, 6)
        if c.show_sublevel_lines and c.admin_level == 1:
            draw_admin_boundaries(ax, a2, c.sublevel_lw, c.sublevel_color, 6)

    if _draw_sf and c.extent_type in {'admin1', 'admin2'}:
        draw_focus_boundary(ax, clip_to_bbox(extent_gdf, extent_bbox), c, zorder=7)

    if _draw_cl and cl is not None:
        draw_clinics(ax, cl, c)

    set_extent(ax, extent_bbox)
    style_ax(ax, '', c.title_fontsize, border_lw=0.8,
             hide_ticks=not c.show_latlon_grid)
    draw_latlon_grid(ax, c)

    # Title in top margin
    fig.text(0.5, 1.0 - (c.top_margin_in / 2) / L['figsize'][1],
             c.map_title, ha='center', va='center',
             fontsize=c.title_fontsize, fontweight='bold')

    # Legend
    if _draw_lg:
        handles = [
            mpatches.Patch(facecolor=c.bin_colors[i], edgecolor='#555555',
                           linewidth=0.3, label=c.bin_labels[i])
            for i in range(len(c.bin_colors))
        ]
        
        extra = []
        
        # Add "no clinics" legend entry FIRST
        if c.show_no_clinic_legend and c.clinic_count_col in admin_clipped.columns:
            extra.append(
                mpatches.Patch(facecolor=c.no_clinic_color, edgecolor='#555555',
                              linewidth=0.3, label='No clinics in admin 2')
            )
        
        # Add hatching legend entry
        if c.use_clinic_count_hatching and c.clinic_count_col in admin_clipped.columns:
            # Create a patch with hatching
            hatch_patch = mpatches.Patch(facecolor='white', edgecolor='#555555',
                                        hatch=c.hatch_pattern, linewidth=0.3,
                                        label=f'<{c.clinic_threshold} clinics')
            extra.append(hatch_patch)
        
        # Add clinic count outline legend entries (legacy mode)
        if c.use_clinic_count_outlines:
            from matplotlib.lines import Line2D
            extra.extend([
                Line2D([0], [0], color=c.many_clinics_color,
                       lw=c.many_clinics_lw,
                       label=f'≥{c.clinic_threshold} clinics'),
                Line2D([0], [0], color=c.few_clinics_color,
                       lw=c.few_clinics_lw,
                       label=f'<{c.clinic_threshold} clinics (bold)'),
            ])
        
        # Add clinic marker
        if c.show_clinics and cl is not None and not cl.empty:
            from matplotlib.lines import Line2D
            extra.append(
                Line2D([0], [0], marker=c.clinic_marker, color='w',
                       markerfacecolor=c.clinic_color,
                       markeredgecolor=c.clinic_edgecolor,
                       markersize=c.base_fontsize * 0.7,
                       markeredgewidth=0.5, label='Clinic')
            )
        
        draw_figure_legend(fig, handles, c, L['legend_y_frac'],
                           extra_handles=extra)

    if c.save_figure:
        fig.savefig(c.output_path, dpi=c.dpi, facecolor='white')
        print(f"Saved: {c.output_path}")
    if c.show_figure: plt.show()
    else: plt.close(fig)
    return fig, ax


# ── Pre-built config for % population within 5km choropleth ──────────────────

def pct_within_5km_admin2_config(output_path):
    """
    Config for: % of population within 5 km of a clinic in the same admin2.
    Bold outlines for admin2s with < 5 clinics; thin for >= 5.

    Usage
    -----
        # Prepare your GDF first:
        # admin2_stats must have columns:
        #   'geometry', 'pct_within_5km' (0-100), 'n_clinics' (int)

        cfg = pct_within_5km_admin2_config(OUTPUT_DIR / 'pct_within_5km.png')
        fig, ax = make_choropleth_map(
            admin_gdf   = admin2_stats,    # admin2 with pct_within_5km + n_clinics
            admin0_gdf  = admin0_gdf,
            admin1_gdf  = admin1_gdf,
            admin2_gdf  = admin2_gdf,      # for boundary lines
            clinics_gdf = clinics_ea_gdf,
            ocean_gdf   = ocean_gdf,       # optional
            lakes_gdf   = lakes_gdf,       # optional
            config      = cfg,
        )
    """
    c = ChoroplethConfig()

    c.value_col   = 'pct_within_5km'
    c.admin_level = 2

    # Diverging color scale: red (low coverage) → yellow → blue (high coverage)
    c.bin_edges  = [0, 20, 40, 60, 80, 100.001]
    c.bin_colors = ['#a50026', '#f46d43', '#ffffbf', '#74add1', '#313695']
    c.bin_labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    c.legend_title = '% population within 5 km of a clinic (same admin 2)'
    c.legend_ncol  = 5
    c.missing_color = '#e0e0e0'

    # Clinic count threshold outlines
    c.use_clinic_count_outlines = True
    c.clinic_count_col          = 'n_clinics'
    c.clinic_threshold          = 5          # bold outline if < 5 clinics
    c.few_clinics_lw            = 2.0
    c.few_clinics_color         = '#111111'
    c.many_clinics_lw           = 0.35
    c.many_clinics_color        = '#555555'

    # Admin boundaries
    c.extent_type             = 'admin0'
    c.show_admin0_boundaries  = True
    c.show_admin1_boundaries  = True   # admin1 lines on top of admin2 fill
    c.show_admin2_boundaries  = False  # handled by clinic_count_outlines
    c.show_focus_boundary     = False
    c.admin1_lw               = 0.8
    c.admin1_color            = '#222222'

    c.show_clinics  = False   # set True to overlay clinic dots

    c.figure_width_in  = 14.0
    c.figure_height_in = 10.0

    c.map_title   = '% Population Within 5 km of Nearest Clinic (Same Admin 2)'
    c.output_path = str(output_path)
    c.save_figure = True
    c.show_figure = False
    return c
