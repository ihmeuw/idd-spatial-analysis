import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box












# --- Helpers --------------------------------------------------------------- #

def _get_extent_geom(extent_type, extent_code, dst_crs):
    """Return (ea_geom, lonlat_geom) for the requested extent."""
    if extent_type == 'admin0':
        ea = country_simple_ea_admin0
    elif extent_type == 'admin1':
        ea = country_simple_ea_admin1[country_simple_ea_admin1['ADM1_CODE'] == extent_code]
        if ea.empty:
            raise ValueError(f"No Admin 1 region found with code {extent_code}")
    elif extent_type == 'admin2':
        ea = country_simple_ea_admin2[country_simple_ea_admin2['ADM2_CODE'] == extent_code]
        if ea.empty:
            raise ValueError(f"No Admin 2 region found with code {extent_code}")
    else:
        raise ValueError(f"Invalid extent_type '{extent_type}'. Must be 'admin0', 'admin1', or 'admin2'.")
    return ea, ea.to_crs(dst_crs)


def _reproject_raster(raster_2d, raster_transform, src_crs, dst_crs):
    """Reproject a 2-D raster to dst_crs; return (array, extent_lonlat).

    extent_lonlat is [left, right, bottom, top] in dst_crs units,
    suitable for passing directly to ax.imshow(extent=...).
    """
    nrows, ncols = raster_2d.shape  # shape is (rows, cols) = (height, width)
    bounds = rasterio.transform.array_bounds(nrows, ncols, raster_transform)

    # calculate_default_transform expects (width, height) = (ncols, nrows)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, ncols, nrows, *bounds
    )

    dst = np.empty((dst_h, dst_w), dtype=raster_2d.dtype)
    reproject(
        source=raster_2d, destination=dst,
        src_transform=raster_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # Affine: .c = left (x origin), .f = top (y origin)
    #         .a = pixel width (+ve), .e = pixel height (-ve)
    left   = dst_transform.c
    right  = dst_transform.c + dst_transform.a * dst_w
    top    = dst_transform.f
    bottom = dst_transform.f + dst_transform.e * dst_h  # e is negative

    assert right > left,  f"Degenerate raster extent: left={left}, right={right}"
    assert top  > bottom, f"Degenerate raster extent: bottom={bottom}, top={top}"

    return dst, [left, right, bottom, top]


# --- Main function --------------------------------------------------------- #

def plot_country_raster(
    raster_data,
    raster_transform,
    bins,
    colors,
    labels,
    population_raster,
    admin1_raster,
    title='Raster Map',
    legend_title='Values',
    extent_type='country',
    extent_code=None,
    show_clinics=True,
    other_countries_color='#d3d3d3',
    other_countries_edge_color='#999999',
    other_countries_linewidth=0.5,
    clinic_color='#1f77b4',
    admin0_color='black',
    admin0_linewidth=1.0,
    admin1_color='black',
    admin1_linewidth=0.5,
    admin2_color='darkgrey',
    admin2_linewidth=0.5,
    focus_color='red',
    focus_linewidth=2.0,
    src_crs='ESRI:54034',
    dst_crs='EPSG:4326',
    figsize=(15, 12),
    padding=2,
    legend_column_max=6,
    save_path=None,
    dpi=150,
    show_admin1_boundaries=True,
    show_admin2_boundaries=True,
    show_neighboring_countries=True,
    vmin=None,
    vmax=None,
    ax=None,
    skip_raster=False,
):
    """
    Plot a raster map of Pakistan with customizable options.

    Parameters
    ----------
    raster_data : np.ndarray
        2D or 3D array (if 3D, uses first band). Raw distance raster (NaN where pop=0).
    raster_transform : Affine
        Rasterio affine transform for the raster data.
    bins : list or None
        Bin boundaries for the colormap. If None, uses continuous colormap.
    colors : list or str
        Colors for each bin, or a colormap name for continuous mode.
    labels : list or None
        Legend labels for each bin.
    population_raster : np.ndarray
        Used to mark zero-population pixels as -0.5 (rendered as grey).
    admin1_raster : np.ndarray
        Admin 1 code raster; used for masking when extent_type='admin1'.
    title, legend_title : str
    extent_type : {'admin0', 'admin1', 'admin2'}
    extent_code : int, optional
        Required when extent_type is 'admin1' or 'admin2'.
    show_clinics : bool
    other_countries_color, clinic_color, admin0_color,
    admin1_color, admin2_color : str
    src_crs, dst_crs : str
    figsize : tuple
    padding : float
        Degrees of padding around the focus region.
    legend_column_max : int
    save_path : Path or str, optional
    dpi : int
    show_admin1_boundaries, show_admin2_boundaries,
    show_neighboring_countries : bool
    vmin, vmax : float, optional
        Override bin-based colormap limits.
    ax : matplotlib Axes, optional
        Plot into an existing axes instead of creating a new figure.
    skip_raster : bool
        Skip raster reprojection/plotting (useful for layout testing).

    Returns
    -------
    fig, ax
    """
    # ------------------------------------------------------------------ #
    # 1. Prepare raster                                                    #
    # ------------------------------------------------------------------ #
    raster_2d = raster_data[0] if raster_data.ndim == 3 else raster_data.copy()

    # Zero-population pixels -> distinct grey sentinel value.
    # Must happen BEFORE the admin mask so that the -0.5 sentinel is also
    # wiped out for out-of-region pixels (order matters!).
    raster_2d[population_raster == 0] = -0.5

    # Mask out-of-region pixels to NaN AFTER applying the pop sentinel,
    # so pixels outside the region are NaN (transparent), not -0.5 (grey).
    if extent_type == 'admin1' and extent_code is not None:
        raster_2d[admin1_raster != extent_code] = np.nan

    # ------------------------------------------------------------------ #
    # 2. Resolve extent geometry & map bounds                             #
    # ------------------------------------------------------------------ #
    if extent_code is None and extent_type != 'admin0':
        raise ValueError(f"extent_code is required when extent_type='{extent_type}'")

    extent_ea, extent_lonlat_geom = _get_extent_geom(extent_type, extent_code, dst_crs)
    lon_min, lat_min, lon_max, lat_max = extent_lonlat_geom.total_bounds
    map_extent = [lon_min - padding, lon_max + padding, lat_min - padding, lat_max + padding]

    # box() takes (minx, miny, maxx, maxy) = (left, bottom, right, top)
    # map_extent is [left, right, bottom, top] so we must unpack carefully
    left, right, bottom, top = map_extent[0], map_extent[1], map_extent[2], map_extent[3]
    window_lonlat = box(left, bottom, right, top)

    # Bounding box in equal-area CRS - used to pre-filter boundary geometries
    window_box_ea = (
        gpd.GeoDataFrame([1], geometry=[window_lonlat], crs=dst_crs)
        .to_crs(src_crs)
        .geometry.iloc[0]
    )

    # ------------------------------------------------------------------ #
    # 3. Reproject raster (optional)                                      #
    # ------------------------------------------------------------------ #
    if not skip_raster:
        raster_lonlat, extent_lonlat_bounds = _reproject_raster(
            raster_2d, raster_transform, src_crs, dst_crs
        )

    # ------------------------------------------------------------------ #
    # 4. Set up figure / axes                                             #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#A6B6DC', alpha=0.5, zorder=0)

    # ------------------------------------------------------------------ #
    # 5. Neighboring countries                                            #
    # ------------------------------------------------------------------ #
    if show_neighboring_countries:
        neighbors = (
            all_simple_ea_admin0[
                (all_simple_ea_admin0['ADM0_CODE'] != ADM0_CODE) &
                all_simple_ea_admin0.intersects(window_box_ea)
            ]
            .to_crs(dst_crs)
        )
        neighbors.plot(
            ax=ax, facecolor=other_countries_color, edgecolor=other_countries_edge_color,
            linewidth=other_countries_linewidth, transform=ccrs.PlateCarree(), zorder=1, aspect=None,
        )

    # ------------------------------------------------------------------ #
    # 6. Plot raster (optional)                                           #
    # ------------------------------------------------------------------ #
    if not skip_raster:
        cmap = ListedColormap(colors) if bins is not None else (
            colors if isinstance(colors, str) else ListedColormap(colors)
        )
        norm = BoundaryNorm(bins, cmap.N, clip=True) if bins is not None else None

        masked = np.ma.masked_invalid(raster_lonlat)
        if vmin is None and vmax is None:
            masked = np.ma.masked_where(raster_lonlat == 0, masked)

        ax.imshow(
            masked, extent=extent_lonlat_bounds, origin='upper',
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), zorder=2, interpolation='nearest',
        )

    # ------------------------------------------------------------------ #
    # 7. Administrative boundaries (pre-filtered, single CRS conversion) #
    # ------------------------------------------------------------------ #
    # aspect=None prevents geopandas from calling ax.set_aspect() internally,
    # which in matplotlib 3.10 raises "aspect must be finite and positive"
    # when boundary geometries have degenerate bounds after filtering.
    def _plot_boundary(gdf_ea, color, linewidth, zorder):
        filtered = gdf_ea[gdf_ea.intersects(window_box_ea)]
        if not filtered.empty:
            filtered.to_crs(dst_crs).boundary.plot(
                ax=ax, edgecolor=color, linewidth=linewidth,
                transform=ccrs.PlateCarree(), zorder=zorder, aspect=None,
            )

    if show_admin2_boundaries:
        _plot_boundary(country_simple_ea_admin2, admin2_color, admin2_linewidth, zorder=3)
    if show_admin1_boundaries:
        _plot_boundary(country_simple_ea_admin1, admin1_color, admin1_linewidth, zorder=4)

    # Focus region / Pakistan outline
    if extent_type in ('admin1', 'admin2'):
        country_simple_ea_admin0.to_crs(dst_crs).boundary.plot(
            ax=ax, edgecolor=admin0_color, linewidth=admin0_linewidth,
            transform=ccrs.PlateCarree(), zorder=5, aspect=None,
        )
        # THEN highlight the focus region on top
        extent_ea.to_crs(dst_crs).boundary.plot(
            ax=ax, edgecolor=focus_color, linewidth=focus_linewidth,
            transform=ccrs.PlateCarree(), zorder=6, aspect=None,  # higher zorder
        )
        
    else:
        country_simple_ea_admin0.to_crs(dst_crs).boundary.plot(
            ax=ax, edgecolor=focus_color, linewidth=focus_linewidth,
            transform=ccrs.PlateCarree(), zorder=5, aspect=None,
        )

    # ------------------------------------------------------------------ #
    # 8. Clinics                                                          #
    # ------------------------------------------------------------------ #
    if show_clinics:
        clinics = clinics_ea_gdf.to_crs(dst_crs)
        if extent_type != 'admin0':
            clinics = clinics[clinics.within(window_lonlat)]
        clinics.plot(
            ax=ax, color=clinic_color, markersize=5, alpha=0.7,
            transform=ccrs.PlateCarree(), zorder=6, aspect=None,
        )

    # ------------------------------------------------------------------ #
    # 9. Gridlines, legend, title                                         #
    # ------------------------------------------------------------------ #
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if labels is not None:
        legend_elements = [
            Patch(facecolor=colors[i], edgecolor='black', linewidth=0.5, label=labels[i])
            for i in range(len(labels))
        ]
        if show_clinics:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=clinic_color, markersize=8, alpha=0.7, label='Clinics')
            )
        ax.legend(
            handles=legend_elements,
            loc='upper center', bbox_to_anchor=(0.5, -0.01),
            title=legend_title, frameon=True, fancybox=True,
            fontsize=16, title_fontsize=18, ncol=min(legend_column_max, len(legend_elements)),
            columnspacing=1.5, handletextpad=0.5,
        )

    ax.set_title(title, fontsize=18, fontweight='bold')

    if save_path is not None and created_fig:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if created_fig:
        plt.tight_layout()

    return fig, ax