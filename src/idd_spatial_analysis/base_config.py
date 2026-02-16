"""
base_config.py — SharedConfig base class
All settings that appear in more than one figure type live here.
Figure-specific configs subclass this and add their own attrs.
"""
import numpy as np


class SharedConfig:

    # ── Save / display ────────────────────────────────────────────────────────
    save_figure: bool = True
    show_figure: bool = False
    output_path: str  = 'figure.png'
    dpi:         int  = 200

    # ── Figure dimensions (inches) ────────────────────────────────────────────
    figure_width_in:  float = 14.0
    figure_height_in: float = 10.0
    top_margin_in:    float = 0.45
    legend_height_in: float = 0.90
    left_margin_in:   float = 0.10
    right_margin_in:  float = 0.10

    # ── Color bins ────────────────────────────────────────────────────────────
    bin_edges  = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]
    bin_colors = ['#1a9641','#a6d96a','#ffffbf','#fee08b',
                  '#fdae61','#f46d43','#d73027','#a50026']
    bin_labels = ['0-1 km','1-2 km','2-5 km','5-10 km',
                  '10-20 km','20-50 km','50-100 km','>100 km']
    legend_title:       str = 'Distance to nearest clinic (km)'
    legend_ncol:        int = 0    # 0 = auto (one column per item)
    legend_frameon:     bool = False

    # ── Admin boundary lines ──────────────────────────────────────────────────
    show_admin0_boundaries: bool  = True
    show_admin1_boundaries: bool  = True
    show_admin2_boundaries: bool  = True
    admin0_lw:    float = 0.8;  admin0_color: str = '#222222'
    admin1_lw:    float = 0.5;  admin1_color: str = '#444444'
    admin2_lw:    float = 0.3;  admin2_color: str = '#666666'

    # Focus boundary
    show_focus_boundary:  bool  = True
    focus_boundary_color: str   = '#cc0000'
    focus_boundary_lw:    float = 1.5

    # ── Neighbor countries ────────────────────────────────────────────────────
    # Point to a global admin0 shapefile; the function loads only the
    # neighbors that appear within each panel's bbox.
    neighbor_admin0_path         = None   # str / Path / None
    focus_adm0_code              = None   # your country's code to exclude
    adm0_code_field:         str = 'ADM0_CODE'
    show_neighbors:          bool  = True
    neighbor_facecolor:      str   = '#d3d3d3'
    neighbor_edgecolor:      str   = '#999999'
    neighbor_lw:             float = 0.5
    neighbor_alpha:          float = 1.0
    neighbor_zorder:         int   = 1

    # ── Water layers ──────────────────────────────────────────────────────────
    # Pass ocean_gdf, lakes_gdf, rivers_gdf to the figure function.
    # All must already be in the same CRS as admin data.
    # Load them in your notebook with cartopy.io.shapereader + .to_crs().
    axes_facecolor:     str   = '#A6B6DC'

    show_ocean:         bool  = False
    ocean_color:        str   = '#A6B6DC'
    ocean_alpha:        float = 0.7
    ocean_zorder:       int   = 0
    show_lakes:         bool  = False
    lake_color:         str   = '#A6B6DC'
    lake_alpha:         float = 0.7
    lake_zorder:        int   = 0
    show_rivers:        bool  = False
    river_color:        str   = '#6699cc'
    river_lw:           float = 0.5
    river_alpha:        float = 0.6
    river_zorder:       int   = 2

    # ── Clinics ───────────────────────────────────────────────────────────────
    show_clinics:      bool  = True
    clinic_marker:     str   = '*'
    clinic_size:       float = 80
    clinic_color:      str   = 'white'
    clinic_edgecolor:  str   = 'black'
    clinic_lw:         float = 0.4
    clinic_alpha:      float = 1.0

    no_clinic_color:       str   = '#999999'  # Grey for "no clinics"
    no_clinic_alpha:       float = 0.6
    no_clinic_label:       str   = ''    # Auto-generated as "No clinics in Admin {level}"

    # ── Lat/lon grid ──────────────────────────────────────────────────────────
    show_latlon_grid:         bool  = False
    latlon_grid_lw:           float = 0.4
    latlon_grid_color:        str   = '#888888'
    latlon_grid_alpha:        float = 0.5
    latlon_grid_linestyle:    str   = '--'
    latlon_grid_label_size:   int   = 7

    # ── Fonts ─────────────────────────────────────────────────────────────────
    base_fontsize:         int = 11
    title_fontsize:        int = 14
    legend_fontsize:       int = 10
    legend_title_fontsize: int = 12
    label_fontsize:        int = 8
