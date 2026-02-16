"""
schematic_figure.py
===================
Generic nested three-panel figure engine + schematic convenience wrapper.

Public API
----------
RasterLayer        one raster per panel (distance | population | binary)
PolygonLayer       filled polygons; choropleth if value_col set, solid fill otherwise
PointLayer         point markers
DistanceLayer      highlight pixel + lines to N nearest clinics (main panel only)
PanelSpec          complete specification for one panel
make_nested_figure     generic engine: takes 3 PanelSpecs, builds the figure
make_schematic_figure  convenience wrapper using SchematicConfig
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
import pyproj

from idd_spatial_analysis.base_config import SharedConfig
from idd_spatial_analysis.map_layout import (
    nested_panel_layout, bbox_from_gdf,
    clip_to_bbox, load_clipped, load_neighbors, clip_raster,
)
from idd_spatial_analysis.map_layers import (
    draw_raster, draw_population_raster, draw_binary_raster,
    draw_figure_legend, build_bin_legend_handles,
    set_extent, style_ax, style_inset, draw_latlon_grid,
)


# =============================================================================
# Layer dataclasses
# =============================================================================

@dataclass
class RasterLayer:
    """
    One raster per panel (only one allowed per PanelSpec).

    Parameters
    ----------
    data             : {'data': (1,H,W) ndarray, 'meta': {transform, crs, ...}}
    cmap             : matplotlib colormap
    norm             : matplotlib norm
    raster_type      : 'distance' | 'population' | 'binary'
    alpha            : overall raster transparency
    binary_threshold_km : threshold for 'binary' type
    zero_pop_color   : grey color for zero-population pixels (population type)
    zero_pop_alpha   : alpha for zero-pop pixels
    show_no_clinic_bin : show np.inf pixels in special color (distance type)
    no_clinic_color  : color for no-clinic pixels
    no_clinic_alpha  : alpha for no-clinic pixels
    """
    data:                dict
    cmap:                object
    norm:                object
    raster_type:         str   = 'distance'
    alpha:               float = 0.85
    binary_threshold_km: float = 5.0
    zero_pop_color:      str   = '#cccccc'
    zero_pop_alpha:      float = 0.35
    show_no_clinic_bin:  bool  = True
    no_clinic_color:     str   = '#999999'
    no_clinic_alpha:     float = 0.6


@dataclass
class PolygonLayer:
    """
    Filled polygon layer.

    Two modes:
      Solid fill   — leave value_col=None, set facecolor
      Choropleth   — set value_col + cmap + norm; facecolor is ignored

    draw_boundary_only=True draws only the outline (no fill at all).

    For choropleth layers with clinic_count_col set, bold/thin outline
    treatment is applied automatically on top of the fill.
    """
    gdf:                object
    value_col:          Optional[str]  = None
    cmap:               object         = None
    norm:               object         = None
    missing_color:      str            = '#e0e0e0'
    facecolor:          str            = 'white'
    edgecolor:          str            = '#444444'
    lw:                 float          = 0.5
    zorder:             int            = 3
    draw_boundary_only: bool           = False
    alpha:              float          = 1.0
    clinic_count_col:   Optional[str]  = None
    clinic_threshold:   int            = 5
    few_lw:             float          = 2.0
    few_color:          str            = '#111111'
    many_lw:            float          = 0.35
    many_color:         str            = '#555555'
    hatch:              Optional[str]  = None
    hatch_col:          Optional[str]  = None
    hatch_threshold:    int            = 5
    hatch_color:        str            = '#555555'
    hatch_lw:           float          = 0.5
    hatch_label:        Optional[str]  = None


@dataclass
class PointLayer:
    """Point markers (clinics, cities, etc.)."""
    gdf:       object
    color:     str   = '#1f1f8a'
    edgecolor: str   = 'white'
    size:      float = 8.0
    alpha:     float = 0.85
    marker:    str   = 'o'
    lw:        float = 0.4
    zorder:    int   = 6


@dataclass
class DistanceLayer:
    """
    Highlight pixel + lines to N nearest clinics.
    Valid only on the main panel (specs[2]).
    Requires a RasterLayer to also be present on that panel.
    Distances computed by reprojecting to equal-area — exact, not approximate.
    """
    highlight_pixel_mode:    str            = 'median'  # 'median' | 'center'
    highlighted_pixel_rowcol: Optional[tuple] = None    # (row,col) or None=auto
    n_nearest:               int            = 10
    line_color:              str            = '#111111'
    line_lw:                 float          = 0.8
    line_alpha:              float          = 0.85
    label_bg_alpha:          float          = 0.75
    label_fontsize:          int            = 8
    highlight_marker:        str            = 's'
    highlight_size:          float          = 80.0
    highlight_edgecolor:     str            = '#ffcc00'
    highlight_lw:            float          = 1.8


@dataclass
class PanelSpec:
    """
    Complete specification for one panel.

    Parameters
    ----------
    bbox          : shapely box — geographic extent shown in this panel
    raster        : RasterLayer or None (at most one per panel)
    polygons      : list of PolygonLayer, drawn in list order
    points        : list of PointLayer, drawn in list order
    distance      : DistanceLayer or None (main panel only)
    zoom_rect_to  : shapely box — draws a dashed rect pointing to next panel
    label         : axes title string
    """
    bbox:          object
    raster:        Optional[RasterLayer]    = None
    polygons:      List[PolygonLayer]       = field(default_factory=list)
    points:        List[PointLayer]         = field(default_factory=list)
    distance:      Optional[DistanceLayer]  = None
    zoom_rect_to:  object                   = None
    label:         str                      = ''


# =============================================================================
# Internal helpers
# =============================================================================

def _make_cmap(edges, colors):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        [e if np.isfinite(e) else 1e9 for e in edges],
        ncolors=len(colors))
    return cmap, norm


def _find_pixel(data, meta, mode):
    d = data[0]
    if mode == 'center':
        return meta['height'] // 2, meta['width'] // 2
    valid = np.isfinite(d) & (d > 0) & ~np.isinf(d)
    if not valid.any():
        return meta['height'] // 2, meta['width'] // 2
    fv  = np.where(valid.ravel())[0]
    idx = fv[np.argmin(np.abs(d.ravel()[fv] - np.median(d[valid])))]
    return tuple(int(x) for x in np.unravel_index(idx, d.shape))


def _nearest_km(xy, clinics_gdf, n, crs):
    """
    Find N nearest clinics to xy and return distances in km.
    Reprojects both query point and clinic coords to equal-area (ESRI:54034)
    for exact metric distances regardless of the working CRS.
    """
    ea_crs = "ESRI:54034"
    t      = pyproj.Transformer.from_crs(crs, ea_crs, always_xy=True)
    xy_ea  = t.transform(*xy)
    coords_ea = np.array([t.transform(g.x, g.y) for g in clinics_gdf.geometry])
    d_m, i = cKDTree(coords_ea).query(xy_ea, k=min(n, len(coords_ea)))
    return clinics_gdf.iloc[np.atleast_1d(i)].copy(), np.atleast_1d(d_m) / 1000.0


def _draw_polygon_layer(ax, layer, bbox):
    """Clip and draw one PolygonLayer onto ax."""
    gdf = clip_to_bbox(layer.gdf, bbox)
    if gdf is None or gdf.empty:
        return

    if layer.draw_boundary_only:
        if layer.clinic_count_col and layer.clinic_count_col in gdf.columns:
            low  = gdf[gdf[layer.clinic_count_col] <  layer.clinic_threshold]
            high = gdf[gdf[layer.clinic_count_col] >= layer.clinic_threshold]
            if not high.empty:
                high.boundary.plot(ax=ax, lw=layer.many_lw, color=layer.many_color,
                                   zorder=layer.zorder, aspect=None)
            if not low.empty:
                low.boundary.plot(ax=ax, lw=layer.few_lw, color=layer.few_color,
                                  zorder=layer.zorder + 1, aspect=None)
        else:
            gdf.boundary.plot(ax=ax, lw=layer.lw, color=layer.edgecolor,
                              zorder=layer.zorder, aspect=None)
        return

    if layer.value_col is not None:
        # Choropleth fill
        missing = gdf[gdf[layer.value_col].isna()]
        valued  = gdf[gdf[layer.value_col].notna()]
        if not missing.empty:
            missing.plot(ax=ax, facecolor=layer.missing_color,
                         edgecolor=layer.edgecolor, linewidth=layer.lw,
                         zorder=layer.zorder, aspect=None)
        if not valued.empty:
            valued.plot(ax=ax,
                        color=[layer.cmap(layer.norm(v))
                               for v in valued[layer.value_col]],
                        edgecolor=layer.edgecolor, linewidth=layer.lw,
                        zorder=layer.zorder, aspect=None)
        # Optional clinic-count outlines on top
        if layer.clinic_count_col and layer.clinic_count_col in gdf.columns:
            vv   = gdf[gdf[layer.value_col].notna()]
            low  = vv[vv[layer.clinic_count_col] <  layer.clinic_threshold]
            high = vv[vv[layer.clinic_count_col] >= layer.clinic_threshold]
            if not high.empty:
                high.boundary.plot(ax=ax, lw=layer.many_lw, color=layer.many_color,
                                   zorder=layer.zorder + 1, aspect=None)
            if not low.empty:
                low.boundary.plot(ax=ax, lw=layer.few_lw, color=layer.few_color,
                                  zorder=layer.zorder + 2, aspect=None)
    else:
        # Solid fill
        gdf.plot(ax=ax, facecolor=layer.facecolor, edgecolor=layer.edgecolor,
                 linewidth=layer.lw, alpha=layer.alpha,
                 zorder=layer.zorder, aspect=None)
        # ── Hatching pass ─────────────────────────────────────────────────────────
    if layer.hatch and layer.hatch_col and layer.hatch_col in gdf.columns:
        import matplotlib as _mpl
        hatched = gdf[gdf[layer.hatch_col] < layer.hatch_threshold]
        if not hatched.empty:
            old_lw = _mpl.rcParams['hatch.linewidth']
            _mpl.rcParams['hatch.linewidth'] = layer.hatch_lw
            hatched.plot(ax=ax, facecolor='none', edgecolor=layer.hatch_color,
                        hatch=layer.hatch, linewidth=0,
                        zorder=layer.zorder + 1, aspect=None)
            _mpl.rcParams['hatch.linewidth'] = old_lw


def _draw_panel(ax, spec, c, crs, ocean_gdf, lakes_gdf, rivers_gdf,
                clip_bbox=None):
    """
    Draw everything for one panel onto ax.

    Draw order
    ----------
    0  axes facecolor = ocean blue (set before this call via ax.set_facecolor)
    1  neighboring countries — filled solid (provides surrounding land color)
    2  polygon layers with zorder <= 2 — the admin0 white fill (water trick)
    3  water layers clipped to bbox (sit on top of white land, below everything else)
    4  raster
    5  polygon layers with zorder >= 3 (boundaries, choropleth, focus outlines)
    6  point layers
    7  distance lines + highlight pixel (if DistanceLayer present)
    8  zoom rect (if zoom_rect_to set)
    """
    bbox = clip_bbox if clip_bbox is not None else spec.bbox

    # ── 1: Neighbors ──────────────────────────────────────────────────────────
    neighbors = load_neighbors(
        c.neighbor_admin0_path, bbox, c.focus_adm0_code, c.adm0_code_field, crs)
    if neighbors is not None and not neighbors.empty:
        neighbors.plot(ax=ax, facecolor=c.neighbor_facecolor,
                       edgecolor=c.neighbor_edgecolor,
                       linewidth=c.neighbor_lw, alpha=c.neighbor_alpha,
                       zorder=1, aspect=None)

    # ── 2: Pre-water polygon layers (zorder <= 2, e.g. admin0 white fill) ─────
    for layer in spec.polygons:
        if (layer.value_col is None
                and not layer.draw_boundary_only
                and layer.zorder <= 2):
            _draw_polygon_layer(ax, layer, bbox)

    # ── 3: Water ──────────────────────────────────────────────────────────────
    if ocean_gdf is not None and not ocean_gdf.empty and c.show_ocean:
        clipped = clip_to_bbox(ocean_gdf, bbox)
        if not clipped.empty:
            clipped.plot(ax=ax, facecolor=c.ocean_color, edgecolor='none',
                         alpha=c.ocean_alpha, zorder=c.ocean_zorder, aspect=None)
    if lakes_gdf is not None and not lakes_gdf.empty and c.show_lakes:
        clipped = clip_to_bbox(lakes_gdf, bbox)
        if not clipped.empty:
            clipped.plot(ax=ax, facecolor=c.lake_color, edgecolor='none',
                         alpha=c.lake_alpha, zorder=c.lake_zorder, aspect=None)
    if rivers_gdf is not None and not rivers_gdf.empty and c.show_rivers:
        clipped = clip_to_bbox(rivers_gdf, bbox)
        if not clipped.empty:
            clipped.plot(ax=ax, color=c.river_color, linewidth=c.river_lw,
                         alpha=c.river_alpha, zorder=c.river_zorder, aspect=None)

    # ── 4: Raster ─────────────────────────────────────────────────────────────
    clipped_data = clipped_transform = clipped_meta = None
    if spec.raster is not None:
        rl      = spec.raster
        cd, ct  = clip_raster(rl.data['data'], rl.data['meta'], bbox)
        clipped_data      = cd
        clipped_transform = ct
        clipped_meta      = {**rl.data['meta'], 'transform': ct,
                             'height': cd.shape[1], 'width': cd.shape[2]}
        if rl.raster_type == 'population':
            draw_population_raster(ax, cd, ct, rl.cmap, rl.norm,
                                   alpha=rl.alpha,
                                   zero_pop_color=rl.zero_pop_color,
                                   zero_pop_alpha=rl.zero_pop_alpha)
        elif rl.raster_type == 'binary':
            draw_binary_raster(ax, cd, ct, rl.binary_threshold_km)
        else:
            draw_raster(ax, cd, ct, rl.cmap, rl.norm,
                        alpha=rl.alpha,
                        show_no_clinic_bin=rl.show_no_clinic_bin,
                        no_clinic_color=rl.no_clinic_color,
                        no_clinic_alpha=rl.no_clinic_alpha)

    # ── 5: Polygon layers (main pass, zorder >= 3) ────────────────────────────
    for layer in spec.polygons:
        if (layer.zorder >= 3
                or layer.value_col is not None
                or layer.draw_boundary_only):
            _draw_polygon_layer(ax, layer, bbox)

    # ── 6: Point layers ───────────────────────────────────────────────────────
    for pl in spec.points:
        pts = clip_to_bbox(pl.gdf, bbox)
        if not pts.empty:
            pts.plot(ax=ax, marker=pl.marker, markersize=pl.size,
                     color=pl.color, edgecolors=pl.edgecolor,
                     linewidths=pl.lw, alpha=pl.alpha,
                     zorder=pl.zorder, aspect=None)

    # ── 7: Distance lines ─────────────────────────────────────────────────────
    if spec.distance is not None and clipped_data is not None:
        dl = spec.distance
        rc = dl.highlighted_pixel_rowcol
        hp_row, hp_col = (rc if rc else
                          _find_pixel(clipped_data, clipped_meta,
                                      dl.highlight_pixel_mode))
        t    = clipped_transform
        px_x = t.c + (hp_col + 0.5) * t.a
        px_y = t.f + (hp_row + 0.5) * t.e

        # Use the first PointLayer's GDF as the clinic source
        clinic_src = None
        for pl in spec.points:
            clipped_pts = clip_to_bbox(pl.gdf, bbox)
            if not clipped_pts.empty:
                clinic_src = clipped_pts
                break

        if clinic_src is not None:
            nc_gdf, nc_km = _nearest_km(
                (px_x, px_y), clinic_src, dl.n_nearest, crs)

            rl = spec.raster
            for i, (_, row) in enumerate(nc_gdf.iterrows()):
                cx, cy = row.geometry.x, row.geometry.y
                ax.plot([px_x, cx], [px_y, cy],
                        color=dl.line_color, lw=dl.line_lw,
                        alpha=dl.line_alpha, zorder=7,
                        solid_capstyle='round')
                ax.text((px_x + cx) / 2, (px_y + cy) / 2,
                        f'{nc_km[i]:.1f} km',
                        fontsize=dl.label_fontsize,
                        ha='center', va='center', color='#111111',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white',
                                  alpha=dl.label_bg_alpha,
                                  edgecolor='none'),
                        zorder=9)
            ax.scatter([px_x], [px_y],
                       s=dl.highlight_size, marker=dl.highlight_marker,
                       facecolors=rl.cmap(rl.norm(
                           float(clipped_data[0, hp_row, hp_col]))),
                       edgecolors=dl.highlight_edgecolor,
                       linewidths=dl.highlight_lw, zorder=10)

    # ── 8: Zoom rect ──────────────────────────────────────────────────────────
    if spec.zoom_rect_to is not None:
        x0, y0, x1, y1 = spec.zoom_rect_to.bounds
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=c.rect_linewidth, edgecolor=c.rect_edgecolor,
            facecolor='none', linestyle=c.rect_linestyle, zorder=10))


# =============================================================================
# Generic engine
# =============================================================================

def make_nested_figure(
    specs,
    config,
    ocean_gdf=None,
    lakes_gdf=None,
    rivers_gdf=None,
    legend_handles=None,
):
    """
    Build a three-panel nested figure from PanelSpecs.

    Parameters
    ----------
    specs         : [top_inset_spec, bottom_inset_spec, main_spec]
                    specs[0].bbox = outer extent (shown in top inset)
                    specs[1].bbox = mid extent   (shown in bottom inset)
                    specs[2].bbox = inner extent (shown in main panel)
    config        : SchematicConfig or compatible object
    ocean_gdf     : GeoDataFrame of ocean polygons (pre-projected to working CRS)
    lakes_gdf     : GeoDataFrame of lake polygons (pre-projected)
    rivers_gdf    : GeoDataFrame of river lines (pre-projected)
    legend_handles: list of matplotlib handles or None (auto-built from config bins)

    Returns
    -------
    fig, (ax_top, ax_bot, ax_main)
    """
    assert len(specs) == 3, "specs must have exactly 3 PanelSpecs"
    c = config
    matplotlib.rcParams.update({'font.size': c.base_fontsize})

    top_spec, bot_spec, main_spec = specs

    # Infer CRS from the first polygon layer GDF found
    crs = None
    for spec in specs:
        for layer in spec.polygons:
            if hasattr(layer.gdf, 'crs') and layer.gdf.crs is not None:
                crs = layer.gdf.crs
                break
        if crs is not None:
            break
    if crs is None:
        raise ValueError("Could not infer CRS from any PolygonLayer GDF.")

    L       = nested_panel_layout(c, main_spec.bbox, bot_spec.bbox, top_spec.bbox)
    figsize = L['figsize']
    print(f"Figure: {figsize[0]:.2f}×{figsize[1]:.2f} in | "
          f"main rect: {[f'{v:.3f}' for v in L['main_panel_rect']]}")

    fig     = plt.figure(figsize=figsize, facecolor='white')
    ax_main = fig.add_axes(L['main_panel_rect'])
    ax_top  = fig.add_axes(L['top_inset_rect'])
    ax_bot  = fig.add_axes(L['bottom_inset_rect'])

    # Style insets and set ocean facecolor on all axes (water trick step 0)
    for ax in (ax_top, ax_bot):
        style_inset(ax, c)
    for ax in (ax_top, ax_bot, ax_main):
        ax.set_facecolor(c.axes_facecolor)

    # Draw each panel
    for ax, spec, clip_bbox in [
        (ax_top,  top_spec,  L['top_inset_bbox']),
        (ax_bot,  bot_spec,  L['bottom_inset_bbox']),
        (ax_main, main_spec, L['main_clip_bbox']),
    ]:
        _draw_panel(ax, spec, c, crs, ocean_gdf, lakes_gdf, rivers_gdf,
                clip_bbox=clip_bbox)
        set_extent(ax, clip_bbox)

    # Titles
    style_ax(ax_top,  top_spec.label,  c.title_fontsize_inset, c.inset_border_lw)
    style_ax(ax_bot,  bot_spec.label,  c.title_fontsize_inset, c.inset_border_lw)
    style_ax(ax_main, main_spec.label, c.title_fontsize,
             border_lw=0.5, hide_ticks=not c.show_latlon_grid)
    draw_latlon_grid(ax_main, c)

    # Legend
    handles = (legend_handles if legend_handles is not None
               else build_bin_legend_handles(c))
    draw_figure_legend(fig, handles, c, L['legend_y_frac'])

    if c.save_figure:
        fig.savefig(c.output_path, dpi=c.dpi, facecolor='white')
        print(f"Saved: {c.output_path}")
    if c.show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_top, ax_bot, ax_main)


# =============================================================================
# SchematicConfig
# =============================================================================

class SchematicConfig(SharedConfig):
    """Configuration for make_schematic_figure."""

    # Inset geometry (inches)
    inset_left_in:            float = 0.20
    inset_top_height_in:      float = 2.80
    inset_top_top_in:         float = 0.75   # gap from figure top to top-inset top
    inset_bottom_height_in:   float = 2.80
    inset_bottom_bottom_in:   float = 1.10   # distance from figure bottom to bottom-inset bottom
    inset_title_pad:          float = 3.0
    inset_bg_color:     str   = 'white';    inset_bg_alpha:    float = 0.88
    inset_border_color: str   = '#333333';  inset_border_lw:   float = 0.8

    # Zoom rect style
    rect_edgecolor: str   = '#cc0000'
    rect_linewidth: float = 1.5
    rect_linestyle: str   = '--'

    # Per-panel clinic marker sizes
    clinic_size_main:     float = 18
    clinic_size_inset:    float = 6
    title_fontsize_inset: int   = 9

    # Figure geometry (inches)
    figure_width_in:  float = 14.0
    figure_height_in: float = 10.0
    top_margin_in:    float = 0.45
    legend_height_in: float = 0.90
    left_margin_in:   float = 0.10
    right_margin_in:  float = 0.10

    # Focal region names (used by make_schematic_figure)
    focal_admin1_name: str = 'Sindh'
    focal_admin2_name: str = 'Karachi Central'
    admin1_name_field: str = 'ADM1_NAME'
    admin2_name_field: str = 'ADM2_NAME'

    # Bbox mode for convenience wrapper
    # 'expand_bbox' — auto-compute from focal admin GDFs
    # 'explicit'    — caller sets bbox_inner / bbox_mid / bbox_outer
    bbox_mode:              str   = 'expand_bbox'
    main_panel_buffer:      float = 0.02      # CRS units (degrees if latlon)
    admin1_inset_buffer_m:  float = 5_000     # always metres; converted internally

    # Explicit bboxes (bbox_mode='explicit')
    bbox_inner = None
    bbox_mid   = None
    bbox_outer = None

    # Detailed shapefile paths (lazy-loaded for main panel)
    detailed_admin0_path = None
    detailed_admin1_path = None
    detailed_admin2_path = None

    # Distance layer defaults
    highlighted_pixel_rowcol        = None
    highlighted_pixel_mode: str     = 'median'
    n_nearest_clinics:      int     = 10


# =============================================================================
# Convenience wrapper
# =============================================================================

def make_schematic_figure(
    distance_raster,
    admin0_gdf,
    admin1_gdf,
    admin2_gdf,
    clinics_gdf,
    ocean_gdf=None,
    lakes_gdf=None,
    rivers_gdf=None,
    config=None,
):
    """
    Convenience wrapper: builds PanelSpecs from SchematicConfig and calls
    make_nested_figure.

    For full control over each panel's content, construct PanelSpecs directly
    and call make_nested_figure instead.
    """
    c    = config or SchematicConfig()
    crs  = admin0_gdf.crs
    cmap, norm = _make_cmap(c.bin_edges, c.bin_colors)

    # ── Focal GDFs ────────────────────────────────────────────────────────────
    fa1 = admin1_gdf[admin1_gdf[c.admin1_name_field] == c.focal_admin1_name]
    fa2 = admin2_gdf[admin2_gdf[c.admin2_name_field] == c.focal_admin2_name]
    if fa1.empty: raise ValueError(f"Admin1 '{c.focal_admin1_name}' not found.")
    if fa2.empty: raise ValueError(f"Admin2 '{c.focal_admin2_name}' not found.")

    # ── Bounding boxes ────────────────────────────────────────────────────────
    if c.bbox_mode == 'explicit':
        if any(b is None for b in [c.bbox_inner, c.bbox_mid, c.bbox_outer]):
            raise ValueError(
                "bbox_mode='explicit' requires bbox_inner, bbox_mid, "
                "and bbox_outer to all be set.")
        bbox_inner = c.bbox_inner
        bbox_mid   = c.bbox_mid
        bbox_outer = c.bbox_outer
    else:  # 'expand_bbox'
        bbox_inner = bbox_from_gdf(fa2, buffer=c.main_panel_buffer)
        buf_m      = c.admin1_inset_buffer_m
        buf        = buf_m / 111_000 if pyproj.CRS(crs).is_geographic else buf_m
        bbox_mid   = bbox_from_gdf(fa1, buffer=buf)
        bbox_outer = bbox_from_gdf(admin0_gdf)

    # ── Raster layer ──────────────────────────────────────────────────────────
    raster_layer = RasterLayer(
        data=distance_raster, cmap=cmap, norm=norm,
        raster_type='distance', alpha=0.85)

    # ── Detailed admin2 for main panel ────────────────────────────────────────
    if c.detailed_admin2_path:
        det_a2 = load_clipped(c.detailed_admin2_path, bbox_inner, crs)
    else:
        det_a2 = admin2_gdf
    fa2_main = (det_a2[det_a2[c.admin2_name_field] == c.focal_admin2_name]
                if c.admin2_name_field in det_a2.columns else fa2)

    # ── Helper: commonly reused layer constructors ────────────────────────────
    def admin0_white():
        """Admin0 white fill — zorder=2 so it runs in the pre-water pass."""
        return PolygonLayer(gdf=admin0_gdf, facecolor='white',
                            edgecolor=c.admin0_color, lw=c.admin0_lw, zorder=2)

    def admin1_outline():
        return PolygonLayer(gdf=admin1_gdf, draw_boundary_only=True,
                            edgecolor=c.admin1_color, lw=c.admin1_lw, zorder=4)

    def admin2_outline(gdf=None):
        return PolygonLayer(gdf=gdf or admin2_gdf, draw_boundary_only=True,
                            edgecolor=c.admin2_color, lw=c.admin2_lw, zorder=5)

    def focus_outline():
        return PolygonLayer(gdf=fa2_main, draw_boundary_only=True,
                            edgecolor=c.focus_boundary_color,
                            lw=c.focus_boundary_lw, zorder=11)

    def clinic_pts(size):
        return PointLayer(gdf=clinics_gdf, size=size,
                          color=c.clinic_color, edgecolor=c.clinic_edgecolor,
                          lw=c.clinic_lw, alpha=c.clinic_alpha)

    # ── PanelSpecs ────────────────────────────────────────────────────────────
    top_spec = PanelSpec(
        bbox         = bbox_outer,
        polygons     = [admin0_white(), admin1_outline(), admin2_outline()],
        points       = [clinic_pts(c.clinic_size_inset)] if c.show_clinics else [],
        zoom_rect_to = bbox_mid,
        label        = 'Pakistan',
    )

    bot_spec = PanelSpec(
        bbox         = bbox_mid,
        polygons     = [admin0_white(), admin1_outline(), admin2_outline()],
        points       = [clinic_pts(c.clinic_size_inset)] if c.show_clinics else [],
        zoom_rect_to = bbox_inner,
        label        = c.focal_admin1_name,
    )

    main_spec = PanelSpec(
        bbox     = bbox_inner,
        raster   = raster_layer,
        polygons = [admin0_white(), admin1_outline(),
                    admin2_outline(det_a2), focus_outline()],
        points   = [clinic_pts(c.clinic_size_main)] if c.show_clinics else [],
        distance = DistanceLayer(
            highlight_pixel_mode     = c.highlighted_pixel_mode,
            highlighted_pixel_rowcol = c.highlighted_pixel_rowcol,
            n_nearest                = c.n_nearest_clinics,
        ),
        label    = f'{c.focal_admin2_name} — distance to nearest clinic (km)',
    )

    return make_nested_figure(
        specs     = [top_spec, bot_spec, main_spec],
        config    = c,
        ocean_gdf = ocean_gdf,
        lakes_gdf = lakes_gdf,
        rivers_gdf= rivers_gdf,
    )
