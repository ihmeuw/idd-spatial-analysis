"""Rasters as ``(array, BlockGrid)`` pairs: windows, mosaics, regridding, zone masks, atomic writes.

``raster.adapters`` converts to and from ``rasterra.RasterArray`` and is imported explicitly
because it needs the optional ``rasterra`` extra.
"""

from idd_spatial_analysis.raster.io import atomic_write, mosaic, read_window, write_geotiff_atomic
from idd_spatial_analysis.raster.regrid import NearestIndexMap, reproject_to_grid, resample_masked
from idd_spatial_analysis.raster.zones import rasterize_zones

__all__ = [
    "NearestIndexMap",
    "atomic_write",
    "mosaic",
    "rasterize_zones",
    "read_window",
    "reproject_to_grid",
    "resample_masked",
    "write_geotiff_atomic",
]
