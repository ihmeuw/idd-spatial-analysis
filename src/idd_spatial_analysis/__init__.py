"""Shared geospatial tools for IDD research.

Subpackages: ``frame`` (block grids and tile indexes), ``boundaries`` (admin polygons),
``geometry`` (antimeridian-safe geometry and cell areas), ``raster`` (windows, mosaics,
regridding, zone masks, writes) and ``zonal`` (weighted sums, zone weights, hierarchy roll-up,
nearest-facility distance). Import the subpackage you need; this module imports nothing heavy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
