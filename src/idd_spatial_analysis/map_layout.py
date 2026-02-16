"""
map_layout.py — Inch-based layout engine + figure-scoped spatial utilities
All dimensions in inches; figure fractions computed internally.
bbox_inches is NEVER used — figure size is always exact.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import box


def bbox_from_gdf(gdf, buffer=0):
    minx, miny, maxx, maxy = gdf.total_bounds
    return box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)


def clip_to_bbox(gdf, bbox_geom):
    return gpd.clip(gdf, bbox_geom)


def load_clipped(path, bbox_geom, target_crs):
    import fiona
    with fiona.open(path) as src:
        src_crs  = src.crs
        bbox_src = (gpd.GeoSeries([bbox_geom], crs=target_crs)
                    .to_crs(src_crs).iloc[0])
        features = list(src.filter(bbox=bbox_src.bounds))
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gpd.clip(gdf, bbox_geom)


def load_neighbors(path, bbox_geom, focus_code, code_field, target_crs):
    """Load neighboring countries — always called for every panel."""
    if path is None:
        return None
    gdf = load_clipped(path, bbox_geom, target_crs)
    if gdf.empty:
        return gdf
    if code_field and focus_code is not None:
        gdf = gdf[gdf[code_field] != focus_code]
    return gdf


def clip_raster(raster_data, raster_meta, bbox_geom):
    import rasterio.io
    from rasterio.mask import mask as rmask
    profile = {
        'driver': 'GTiff', 'dtype': raster_data.dtype,
        'width':  raster_meta['width'], 'height': raster_meta['height'],
        'count':  raster_data.shape[0], 'crs':    raster_meta['crs'],
        'transform': raster_meta['transform'],
        'nodata':    raster_meta.get('nodata', np.nan),
    }
    with rasterio.io.MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(raster_data)
        with mf.open() as ds:
            out, transform = rmask(
                ds, [bbox_geom.__geo_interface__],
                crop=True, nodata=profile['nodata'], all_touched=True)
    return out, transform


def single_panel_layout(c):
    fw, fh = c.figure_width_in, c.figure_height_in
    mw     = fw - c.left_margin_in - c.right_margin_in
    mh     = fh - c.top_margin_in  - c.legend_height_in
    return {
        'figsize':         (fw, fh),
        'main_panel_rect': [c.left_margin_in/fw, c.legend_height_in/fh, mw/fw, mh/fh],
        'legend_y_frac':   (c.legend_height_in / 2) / fh,
    }


def nested_panel_layout(c, bbox_inner, bbox_mid, bbox_outer):
    """
    Three-panel nested layout: two stacked insets on the left + main panel.

    Panel mapping
    -------------
    specs[0] → top inset    — shows bbox_outer, zoom rect to bbox_mid
    specs[1] → bottom inset — shows bbox_mid,   zoom rect to bbox_inner
    specs[2] → main panel   — shows bbox_inner (aspect-corrected)

    Inset widths derived from each bbox aspect ratio — frames match content exactly.
    """
    fw    = c.figure_width_in
    fh    = c.figure_height_in
    mw_in = fw - c.left_margin_in - c.right_margin_in
    mh_in = fh - c.top_margin_in  - c.legend_height_in

    # Adjust main panel bbox to match panel aspect ratio
    gx0, gy0, gx1, gy1 = bbox_inner.bounds
    gw, gh = gx1 - gx0, gy1 - gy0
    pa     = mw_in / mh_in
    geo_aspect = gw / gh
    if geo_aspect < pa:
        new_w = gh * pa; cx = (gx0 + gx1) / 2
        bbox_inner = box(cx - new_w/2, gy0, cx + new_w/2, gy1)
    elif geo_aspect > pa:
        new_h = gw / pa; cy = (gy0 + gy1) / 2
        bbox_inner = box(gx0, cy - new_h/2, gx1, cy + new_h/2)

    main_rect = [c.left_margin_in/fw, c.legend_height_in/fh, mw_in/fw, mh_in/fh]

    def _inset_rect(left_in, bottom_in, h_in, display_bbox):
        x0, y0, x1, y1 = display_bbox.bounds
        w_in = h_in * (x1 - x0) / (y1 - y0)
        return [left_in/fw, bottom_in/fh, w_in/fw, h_in/fh]

    top_bot_in = fh - c.inset_top_top_in - c.inset_top_height_in
    top_rect   = _inset_rect(c.inset_left_in, top_bot_in,
                             c.inset_top_height_in, bbox_outer)
    bot_rect   = _inset_rect(c.inset_left_in, c.inset_bottom_bottom_in,
                             c.inset_bottom_height_in, bbox_mid)

    return {
        'figsize':           (fw, fh),
        'main_panel_rect':   main_rect,
        'main_clip_bbox':    bbox_inner,
        'top_inset_rect':    top_rect,
        'top_inset_bbox':    bbox_outer,
        'bottom_inset_rect': bot_rect,
        'bottom_inset_bbox': bbox_mid,
        'legend_y_frac':     (c.legend_height_in / 2) / fh,
    }


def schematic_layout(c, focal_admin2, admin0_gdf, focal_admin1):
    """DEPRECATED: use nested_panel_layout() with explicit bboxes."""
    import pyproj
    if c.main_panel_mode == 'manual_bbox':
        if c.main_panel_bbox is None:
            raise ValueError("manual_bbox mode requires main_panel_bbox to be set.")
        bbox_inner = box(*c.main_panel_bbox)
    elif c.main_panel_mode == 'expand_bbox':
        bbox_inner = bbox_from_gdf(focal_admin2, buffer=c.main_panel_buffer)
    else:
        raise ValueError(f"Unknown main_panel_mode: '{c.main_panel_mode}'")
    buf = c.admin1_inset_buffer_m
    if pyproj.CRS(admin0_gdf.crs).is_geographic:
        buf = buf / 111_000
    bbox_mid   = bbox_from_gdf(focal_admin1, buffer=buf)
    bbox_outer = bbox_from_gdf(admin0_gdf)
    return nested_panel_layout(c, bbox_inner, bbox_mid, bbox_outer)
