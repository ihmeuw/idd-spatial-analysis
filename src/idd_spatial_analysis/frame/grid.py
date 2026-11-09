"""Raster grids: CRS, affine transform and shape, with windows, bounds and pixel coordinates.

The pair ``(array, BlockGrid)`` is this package's raster representation: a 2-D numpy array and
the grid that places it. Nothing here reads pixel values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from idd_spatial_analysis.constants import (
    GEOTIFF_BLOCKSIZE,
    GEOTIFF_COMPRESSION,
    RASTER_FLOAT_DTYPE,
    WGS84,
)
from idd_spatial_analysis.frame.bounds import Bounds, bbox_in_crs, pad_bounds

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_SPAN_DECIMALS = 6
"""Rounding applied before ceil() so that a span of 3199.9999999 pixels counts as 3200."""


@dataclass(frozen=True)
class BlockGrid:
    """A north-up raster grid: CRS string, affine transform and shape.

    Pixel ``(row, col)`` covers ``x`` in ``[c + col*a, c + (col+1)*a)`` and ``y`` in
    ``(f + (row+1)*e, f + row*e]`` with ``e < 0``.
    """

    crs: str
    transform: Affine
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            msg = f"grid shape must be positive, got {(self.height, self.width)}"
            raise ValueError(msg)
        if self.transform.b != 0 or self.transform.d != 0:
            msg = "rotated or sheared grids are not supported"
            raise ValueError(msg)
        if self.transform.a <= 0 or self.transform.e >= 0:
            msg = "expected a north-up grid: positive pixel width and negative pixel height"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def resolution(self) -> tuple[float, float]:
        """(pixel width, pixel height), both positive, in CRS units."""
        return (float(self.transform.a), float(-self.transform.e))

    @property
    def bounds(self) -> Bounds:
        left, bottom, right, top = array_bounds(self.height, self.width, self.transform)
        return (float(left), float(bottom), float(right), float(top))

    def lonlat_bounds(self, pad: float = 0.0) -> Bounds:
        """The lon/lat box covering the grid plus ``pad`` CRS units on every side."""
        return bbox_in_crs(pad_bounds(self.bounds, pad), self.crs, WGS84)

    def pixel_centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """x of every column centre and y of every row centre, as two 1-D arrays."""
        cols = np.arange(self.width, dtype=np.float64) + 0.5
        rows = np.arange(self.height, dtype=np.float64) + 0.5
        xs = np.asarray(self.transform.c + cols * self.transform.a, dtype=np.float64)
        ys = np.asarray(self.transform.f + rows * self.transform.e, dtype=np.float64)
        return xs, ys

    def rowcol(
        self, xs: NDArray[Any], ys: NDArray[Any]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row and column of each ``(x, y)`` by floor, as rasterio does; no bounds check.

        A point on a shared pixel edge belongs to the pixel to its right or below it.
        """
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        cols = np.floor((x_arr - self.transform.c) / self.transform.a)
        rows = np.floor((y_arr - self.transform.f) / self.transform.e)
        return rows.astype(np.int64), cols.astype(np.int64)

    def contains(self, rows: NDArray[Any], cols: NDArray[Any]) -> NDArray[np.bool_]:
        """True where ``(row, col)`` falls inside the grid."""
        row_arr = np.asarray(rows)
        col_arr = np.asarray(cols)
        inside = (row_arr >= 0) & (row_arr < self.height) & (col_arr >= 0) & (col_arr < self.width)
        return np.asarray(inside, dtype=np.bool_)

    def window(self, bounds: Bounds, *, pad: int = 0, clip: bool = True) -> Window:
        """Integer window covering ``bounds`` plus ``pad`` pixels.

        Clipped to the grid unless ``clip=False`` (for boundless reads, where the window may
        start before the first pixel). Raises ``ValueError`` when the bounds are degenerate or,
        when clipping, do not overlap the grid.
        """
        left, bottom, right, top = bounds
        if left >= right or bottom >= top:
            msg = f"degenerate bounds {bounds}"
            raise ValueError(msg)
        col0 = math.floor((left - self.transform.c) / self.transform.a) - pad
        col1 = math.ceil((right - self.transform.c) / self.transform.a) + pad
        row0 = math.floor((top - self.transform.f) / self.transform.e) - pad
        row1 = math.ceil((bottom - self.transform.f) / self.transform.e) + pad
        if clip:
            col0, row0 = max(col0, 0), max(row0, 0)
            col1, row1 = min(col1, self.width), min(row1, self.height)
            if col1 <= col0 or row1 <= row0:
                msg = f"bounds {bounds} do not overlap the grid {self.bounds}"
                raise ValueError(msg)
        return Window(col0, row0, col1 - col0, row1 - row0)

    def sub(self, window: Window) -> BlockGrid:
        """The grid of ``window`` within this grid."""
        transform = window_transform(window, self.transform)
        return BlockGrid(
            crs=self.crs,
            transform=Affine(*transform[:6]),
            height=int(window.height),
            width=int(window.width),
        )

    def profile(
        self,
        *,
        dtype: str = RASTER_FLOAT_DTYPE,
        nodata: float | None = math.nan,
        count: int = 1,
    ) -> dict[str, Any]:
        """A rasterio GeoTIFF profile for this grid: compressed, tiled when large enough."""
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": count,
            "crs": self.crs,
            "transform": self.transform,
            "height": self.height,
            "width": self.width,
            "nodata": nodata,
            "compress": GEOTIFF_COMPRESSION,
        }
        if min(self.shape) >= GEOTIFF_BLOCKSIZE:
            profile.update(tiled=True, blockxsize=GEOTIFF_BLOCKSIZE, blockysize=GEOTIFF_BLOCKSIZE)
        return profile


def grid_from_bounds(
    bounds: Bounds, resolution: float | tuple[float, float], crs: str
) -> BlockGrid:
    """The grid anchored at ``(left, top)`` whose pixels of ``resolution`` cover ``bounds``.

    The right and bottom edges round up to whole pixels, so the grid stays on the lattice of
    its origin and may overhang the bounds by less than one pixel.
    """
    left, bottom, right, top = bounds
    xres, yres = (resolution, resolution) if isinstance(resolution, int | float) else resolution
    if xres <= 0 or yres <= 0:
        msg = f"resolution must be positive, got {resolution}"
        raise ValueError(msg)
    if left >= right or bottom >= top:
        msg = f"degenerate bounds {bounds}"
        raise ValueError(msg)
    width = math.ceil(round((right - left) / xres, _SPAN_DECIMALS))
    height = math.ceil(round((top - bottom) / yres, _SPAN_DECIMALS))
    return BlockGrid(
        crs=crs,
        transform=Affine(xres, 0.0, left, 0.0, -yres, top),
        height=height,
        width=width,
    )


def grid_from_raster(path: Path | str) -> BlockGrid:
    """The grid of a raster file (band-independent)."""
    with rasterio.open(path) as src:
        if src.crs is None:
            msg = f"{path} carries no CRS"
            raise ValueError(msg)
        return BlockGrid(
            crs=str(src.crs.to_string()),
            transform=Affine(*src.transform[:6]),
            height=int(src.height),
            width=int(src.width),
        )
