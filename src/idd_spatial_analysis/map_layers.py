"""
map_layers.py — Atomic drawing functions (Layer 2)
Each function draws exactly one thing onto a provided ax.
No figure/axes creation. No reprojection. All data pre-clipped to CRS.
"""
import numpy as np
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from idd_spatial_analysis.map_layout import clip_to_bbox


# ── Raster ────────────────────────────────────────────────────────────────────

def draw_raster(ax, data, transform, cmap, norm,
                alpha=1.0, highlight_mask=None, highlight_alpha=1.0,
                no_clinic_color='#999999', no_clinic_alpha=0.6, 
                show_no_clinic_bin=True, admin0_mask=None):
    """Standard distance/value raster. nodata and negatives → transparent.
    
    Pixels with value np.inf (no clinics in region) are optionally shown
    in a special color if show_no_clinic_bin=True.
    
    admin0_mask : (H,W) bool array or None — True = inside country, False = outside.
                  Derived from population raster NaN mask.
    """
    d = data[0].astype(float)
    h, w = d.shape
    t = transform
    rgba         = cmap(norm(d))
    rgba[..., 3] = alpha
    if highlight_mask is not None:
        rgba[highlight_mask, 3] = highlight_alpha
    

    
    # Make truly invalid pixels (NaN or negative values) transparent
    invalid = ~np.isfinite(d) | (d < 0)
    rgba[invalid, 3] = 0.0

    # Handle np.inf "no clinics" pixels specially
    if show_no_clinic_bin:
        no_clinic_mask = np.isinf(d)
        rgba[no_clinic_mask] = mcolors.to_rgba(no_clinic_color, no_clinic_alpha)
    
    # Apply additional mask if provided (e.g., from population raster)
    # This handles edge cases where NaN might not have propagated properly
    if admin0_mask is not None:
        rgba[~admin0_mask, 3] = 0.0  # False = outside country
    
    ax.imshow(rgba, extent=[t.c, t.c + w*t.a, t.f + h*t.e, t.f],
              origin="upper", interpolation="nearest", aspect="auto", zorder=3)


def draw_population_raster(ax, data, transform, cmap, norm,
                            alpha=0.85,
                            zero_pop_color='#cccccc', zero_pop_alpha=0.35,
                            admin0_mask=None):
    """Population raster: zero-pop pixels shown in grey, nodata transparent.
    
    admin0_mask : (H,W) bool array or None — True = inside country, False = outside.
                  Usually not needed since pop raster already has NaN outside.
    """
    d = data[0].astype(float)
    h, w = d.shape
    t = transform
    rgba         = cmap(norm(d))
    rgba[..., 3] = alpha
    zero = np.isfinite(d) & (d == 0)
    rgba[zero] = mcolors.to_rgba(zero_pop_color, alpha=zero_pop_alpha)
    rgba[~np.isfinite(d), 3] = 0.0
    
    # Apply additional mask if provided (usually not needed for pop raster)
    if admin0_mask is not None:
        rgba[~admin0_mask, 3] = 0.0  # False = outside country
    ax.imshow(rgba, extent=[t.c, t.c + w*t.a, t.f + h*t.e, t.f],
              origin="upper", interpolation="nearest", aspect="auto", zorder=3)


def draw_binary_raster(ax, data, transform, threshold_km,
                       within_color='#1a9641', within_alpha=0.7,
                       outside_color='#d73027', outside_alpha=0.5):
    """Y/N is nearest clinic within threshold_km."""
    d = data[0].astype(float)
    h, w = d.shape
    t = transform
    rgba    = np.zeros((h, w, 4))
    valid   = np.isfinite(d) & (d >= 0)
    rgba[valid & (d <= threshold_km)] = mcolors.to_rgba(within_color,  within_alpha)
    rgba[valid & (d >  threshold_km)] = mcolors.to_rgba(outside_color, outside_alpha)
    ax.imshow(rgba, extent=[t.c, t.c + w*t.a, t.f + h*t.e, t.f],
              origin="upper", interpolation="nearest", aspect="auto", zorder=3)


# ── Vector — background ───────────────────────────────────────────────────────

def draw_neighbors(ax, neighbors_gdf, c):
    """Draw pre-loaded neighbor polygons (already clipped to panel bbox)."""
    if neighbors_gdf is None or neighbors_gdf.empty:
        return
    neighbors_gdf.plot(ax=ax, facecolor=c.neighbor_facecolor,
                       edgecolor=c.neighbor_edgecolor, linewidth=c.neighbor_lw,
                       alpha=c.neighbor_alpha, zorder=c.neighbor_zorder, aspect=None)


def draw_ocean(ax, ocean_gdf, clip_bbox, c):
    if ocean_gdf is None or ocean_gdf.empty or not c.show_ocean:
        return
    clipped = clip_to_bbox(ocean_gdf, clip_bbox)
    if not clipped.empty:
        clipped.plot(ax=ax, facecolor=c.ocean_color, edgecolor='none',
                     alpha=c.ocean_alpha, zorder=c.ocean_zorder, aspect=None)


def draw_lakes(ax, lakes_gdf, clip_bbox, c):
    if lakes_gdf is None or lakes_gdf.empty or not c.show_lakes:
        return
    clipped = clip_to_bbox(lakes_gdf, clip_bbox)
    if not clipped.empty:
        clipped.plot(ax=ax, facecolor=c.lake_color, edgecolor='none',
                     alpha=c.lake_alpha, zorder=c.lake_zorder, aspect=None)


def draw_rivers(ax, rivers_gdf, clip_bbox, c):
    if rivers_gdf is None or rivers_gdf.empty or not c.show_rivers:
        return
    clipped = clip_to_bbox(rivers_gdf, clip_bbox)
    if not clipped.empty:
        clipped.plot(ax=ax, color=c.river_color, linewidth=c.river_lw,
                     alpha=c.river_alpha, zorder=c.river_zorder, aspect=None)


# ── Vector — admin boundaries ─────────────────────────────────────────────────

def draw_admin_boundaries(ax, gdf, lw, color, zorder=3):
    if gdf is None or gdf.empty:
        return
    gdf.boundary.plot(ax=ax, lw=lw, color=color, zorder=zorder, aspect=None)


def draw_focus_boundary(ax, gdf, c, zorder=9):
    if gdf is None or gdf.empty or not c.show_focus_boundary:
        return
    gdf.boundary.plot(ax=ax, lw=c.focus_boundary_lw,
                      color=c.focus_boundary_color, zorder=zorder, aspect=None)


# ── Choropleth ────────────────────────────────────────────────────────────────

def draw_choropleth(ax, gdf, value_col, cmap, norm,
                    edgecolor='#444444', lw=0.3, zorder=3,
                    missing_color='#e0e0e0', 
                    clinic_count_col=None, clinic_threshold=5,
                    hatch_few_clinics=False, hatch_pattern='///',
                    hatch_color='#666666', hatch_lw=1.0,
                    no_clinic_color='#e0e0e0'):
    """Fill admin polygons by value_col; NaN rows get missing_color."""
    if gdf is None or gdf.empty:
        return
    
    gdf = gdf.copy()
    
    # Separate into categories
    missing = gdf[gdf[value_col].isna()]
    
    # If clinic_count_col is provided, handle "no clinics" separately
    if clinic_count_col is not None and clinic_count_col in gdf.columns:
        no_clinics = gdf[(gdf[value_col].notna()) & (gdf[clinic_count_col] == 0)]
        valued = gdf[(gdf[value_col].notna()) & (gdf[clinic_count_col] > 0)]
        
        # Draw regions with no clinics in special color
        if not no_clinics.empty:
            print(f"DEBUG: Drawing {len(no_clinics)} regions with no clinics in color {no_clinic_color}")
            no_clinics.plot(ax=ax, facecolor=no_clinic_color, edgecolor=edgecolor,
                           linewidth=lw, zorder=zorder, aspect=None)
    else:
        no_clinics = gpd.GeoDataFrame()
        valued = gdf[gdf[value_col].notna()]
    
    # Draw missing data regions
    if not missing.empty:
        missing.plot(ax=ax, facecolor=missing_color, edgecolor=edgecolor,
                    linewidth=lw, zorder=zorder, aspect=None)
    
    # Draw regions with data
    if not valued.empty:
        valued.plot(ax=ax, color=[cmap(norm(v)) for v in valued[value_col]],
                   edgecolor=edgecolor, linewidth=lw, zorder=zorder, aspect=None)
        
        # Apply hatching to regions with few clinics
        if hatch_few_clinics and clinic_count_col is not None:
            few_clinics = valued[valued[clinic_count_col] < clinic_threshold]
            if not few_clinics.empty:
                print(f"DEBUG: Applying hatching to {len(few_clinics)} regions with <{clinic_threshold} clinics")
                print(f"DEBUG: Hatch pattern='{hatch_pattern}', color='{hatch_color}', lw={hatch_lw}")
                
                # Set hatch parameters
                from matplotlib import rcParams
                old_hatch_color = rcParams.get('hatch.color', 'k')
                old_hatch_lw = rcParams.get('hatch.linewidth', 1.0)
                rcParams['hatch.color'] = hatch_color
                rcParams['hatch.linewidth'] = hatch_lw
                
                # Draw with their existing colors but add hatching on top
                # Need actual facecolor for hatching to work
                for idx, row in few_clinics.iterrows():
                    color = cmap(norm(row[value_col]))
                    gpd.GeoSeries([row.geometry]).plot(
                        ax=ax, 
                        facecolor=color,
                        edgecolor='none',
                        hatch=hatch_pattern, 
                        zorder=zorder + 0.5, 
                        aspect=None
                    )
                
                # Restore original settings
                rcParams['hatch.color'] = old_hatch_color
                rcParams['hatch.linewidth'] = old_hatch_lw
            else:
                print(f"DEBUG: No regions found with <{clinic_threshold} clinics to hatch")


def draw_clinic_count_outlines(ax, gdf, count_col, threshold=5,
                                low_color='#111111', low_lw=2.0,
                                high_color='#555555', high_lw=0.4,
                                zorder=4):
    """
    Draw admin2 outlines with two weights based on clinic count.
    Units with count < threshold (few clinics) get bold outline.
    Units with count >= threshold get thin outline.
    
    Note: This is now deprecated in favor of hatching via draw_choropleth.
    Kept for backward compatibility.
    """
    if gdf is None or gdf.empty:
        return
    low  = gdf[gdf[count_col] <  threshold]
    high = gdf[gdf[count_col] >= threshold]
    if not high.empty:
        high.boundary.plot(ax=ax, lw=high_lw, color=high_color,
                           zorder=zorder, aspect=None)
    if not low.empty:
        low.boundary.plot(ax=ax, lw=low_lw, color=low_color,
                          zorder=zorder + 1, aspect=None)


# ── Points ────────────────────────────────────────────────────────────────────

def draw_clinics(ax, clinics_gdf, c, markersize=None, zorder=6):
    if clinics_gdf is None or clinics_gdf.empty or not c.show_clinics:
        return
    ms = markersize if markersize is not None else c.clinic_size
    clinics_gdf.plot(ax=ax, marker=c.clinic_marker, markersize=ms,
                     color=c.clinic_color, edgecolors=c.clinic_edgecolor,
                     linewidths=c.clinic_lw, alpha=c.clinic_alpha,
                     zorder=zorder, aspect=None)


# ── Schematic-specific ────────────────────────────────────────────────────────

def draw_zoom_rect(ax, bbox_geom, c):
    x0, y0, x1, y1 = bbox_geom.bounds
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        linewidth=c.rect_linewidth, edgecolor=c.rect_edgecolor,
        facecolor='none', linestyle=c.rect_linestyle, zorder=10))


def draw_distance_lines(ax, px_x, px_y, near_clinics_gdf, near_dists_km,
                         raster_data, hp_row, hp_col, cmap, norm, c):
    for i, (_, row) in enumerate(near_clinics_gdf.iterrows()):
        cx, cy = row.geometry.x, row.geometry.y
        ax.plot([px_x, cx], [px_y, cy], color=c.line_color, lw=c.line_lw,
                alpha=c.line_alpha, zorder=7, solid_capstyle='round')
        ax.text((px_x+cx)/2, (px_y+cy)/2, f'{near_dists_km[i]:.1f} km',
                fontsize=c.label_fontsize, ha='center', va='center',
                color='#111111',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=c.label_bg_alpha, edgecolor='none'), zorder=9)
    ax.scatter([px_x], [px_y], s=c.highlight_size, marker=c.highlight_marker,
               facecolors=cmap(norm(float(raster_data[0, hp_row, hp_col]))),
               edgecolors=c.highlight_edgecolor, linewidths=c.highlight_lw,
               zorder=10)


# ── Grid ──────────────────────────────────────────────────────────────────────

def draw_latlon_grid(ax, c):
    if not c.show_latlon_grid:
        return
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    for span, steps in [((xr-xl), [0.1,0.25,0.5,1,2,5]),
                         ((yt-yb), [0.1,0.25,0.5,1,2,5])]:
        for s in steps:
            if span / s < 8:
                break
    x_step = y_step = s
    xt = np.arange(np.ceil(xl/x_step)*x_step,  xr, x_step)
    yt_ = np.arange(np.ceil(yb/y_step)*y_step, yt, y_step)
    kw = dict(lw=c.latlon_grid_lw, color=c.latlon_grid_color,
              alpha=c.latlon_grid_alpha, linestyle=c.latlon_grid_linestyle, zorder=0)
    for x in xt: ax.axvline(x, **kw)
    for y in yt_: ax.axhline(y, **kw)
    ax.set_xticks(xt); ax.set_yticks(yt_)
    ax.set_xticklabels([f'{x:.2g}°' for x in xt], fontsize=c.latlon_grid_label_size)
    ax.set_yticklabels([f'{y:.2g}°' for y in yt_], fontsize=c.latlon_grid_label_size)
    ax.tick_params(length=3, width=0.5)


# ── Legends ───────────────────────────────────────────────────────────────────

def build_bin_legend_handles(c, include_clinic=True):
    handles = []
    
    # Optionally add zero-population entry first (for population rasters)
    if getattr(c, 'show_zero_pop_legend', False):
        handles.append(
            mpatches.Patch(facecolor=c.zero_pop_color, edgecolor='#555555',
                          linewidth=0.3, alpha=c.zero_pop_alpha,
                          label=c.zero_pop_label)
        )
    
    # Add regular bin entries
    handles.extend([
        mpatches.Patch(facecolor=c.bin_colors[i], edgecolor='#555555',
                       linewidth=0.3, label=c.bin_labels[i])
        for i in range(len(c.bin_colors))
    ])
    
    # Optionally add clinic marker
    if include_clinic and c.show_clinics:
        handles.append(Line2D([0], [0], marker=c.clinic_marker, color='w',
                               markerfacecolor=c.clinic_color,
                               markeredgecolor=c.clinic_edgecolor,
                               markersize=c.base_fontsize * 0.7,
                               markeredgewidth=0.5, label='Clinic'))
    return handles


def draw_figure_legend(fig, handles, c, legend_y_frac,
                       ncol=None, extra_handles=None):
    all_h = handles + (extra_handles or [])
    n     = ncol if ncol else (c.legend_ncol if c.legend_ncol else len(all_h))
    fig.legend(handles=all_h, loc='center', ncol=n,
               fontsize=c.legend_fontsize, frameon=c.legend_frameon, framealpha=0.9,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, legend_y_frac),
               title=c.legend_title, title_fontsize=c.legend_title_fontsize)


# ── Axis styling ──────────────────────────────────────────────────────────────

def set_extent(ax, bbox_geom, enforce_aspect=False):
    """Set axis limits to bbox_geom bounds.

    enforce_aspect : bool
        If True, call ax.set_aspect('equal') after setting limits.
        Set False (default) when the figure layout already guarantees the
        correct aspect ratio via nested_panel_layout or when imshow is
        drawn (imshow uses aspect='auto' and set_aspect='equal' overrides it,
        causing the raster to render outside the visible axes area).
        Set True for single-panel vector-only figures where matplotlib needs
        to enforce the aspect itself.
    """
    x0, y0, x1, y1 = bbox_geom.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if enforce_aspect:
        ax.set_aspect('equal')


def style_ax(ax, title='', title_fontsize=9, border_lw=0.5, hide_ticks=True):
    if hide_ticks:
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(border_lw)
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=4, fontweight='bold')


def style_inset(ax, c):
    ax.patch.set_facecolor(c.inset_bg_color)
    ax.patch.set_alpha(c.inset_bg_alpha)
    for sp in ax.spines.values():
        sp.set_edgecolor(c.inset_border_color)
        sp.set_linewidth(c.inset_border_lw)