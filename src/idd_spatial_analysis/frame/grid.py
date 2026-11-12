"""Raster grids: CRS, affine transform and shape, with windows, bounds and pixel coordinates.

The pair ``(array, BlockGrid)`` is this package's raster representation: a 2-D numpy array and
the grid that places it. Nothing here reads pixel values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from idd_spatial_analysis.constants import (
    GEOTIFF_BLOCKSIZE,
    GEOTIFF_COMPRESSION,
    GRID_MATCH_TOLERANCE,
    PIXEL_EDGE_DECIMALS,
    RASTER_FLOAT_DTYPE,
    WGS84,
)
from idd_spatial_analysis.crs import crs_equal
from idd_spatial_analysis.frame.bounds import Bounds, bbox_in_crs, pad_bounds

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_INTEGER_KINDS = "iu"


def check_nodata(dtype: str | np.dtype[Any], nodata: float | None) -> np.dtype[Any]:
    """The dtype, once ``nodata`` is known to be storable in it.

    ``None`` is always allowed. An integer dtype refuses NaN, infinities, non-integral values
    and values outside its range; float dtypes take any value, NaN included.
    """
    out = np.dtype(dtype)
    if nodata is None or out.kind not in _INTEGER_KINDS:
        return out
    if isinstance(nodata, int | np.integer):
        value = int(nodata)  # exact at any width; float() would round above 2**53
    else:
        as_float = float(nodata)
        if not math.isfinite(as_float) or not as_float.is_integer():
            msg = f"nodata {nodata!r} is not representable in {out.name}"
            raise ValueError(msg)
        value = int(as_float)
    info = np.iinfo(out)
    if not info.min <= value <= info.max:
        msg = f"nodata {nodata!r} is outside the range of {out.name}"
        raise ValueError(msg)
    return out


def pixel_quotient(values: NDArray[Any] | float, origin: float, step: float) -> NDArray[np.float64]:
    """``(value - origin) / step`` rounded to ``PIXEL_EDGE_DECIMALS``.

    A coordinate a rounding error away from a pixel edge is placed on the edge, so floor and
    ceil put it in the pixel the documented edge rule says, on every lattice.
    """
    quotient = (np.asarray(values, dtype=np.float64) - origin) / step
    return np.asarray(np.round(quotient, PIXEL_EDGE_DECIMALS), dtype=np.float64)


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

        A point on a shared pixel edge (within ``PIXEL_EDGE_DECIMALS`` of it) belongs to the
        pixel to its right or below it.
        """
        cols = np.floor(pixel_quotient(xs, self.transform.c, self.transform.a))
        rows = np.floor(pixel_quotient(ys, self.transform.f, self.transform.e))
        return rows.astype(np.int64), cols.astype(np.int64)

    def contains(self, rows: NDArray[Any], cols: NDArray[Any]) -> NDArray[np.bool_]:
        """True where ``(row, col)`` falls inside the grid."""
        row_arr = np.asarray(rows)
        col_arr = np.asarray(cols)
        inside = (row_arr >= 0) & (row_arr < self.height) & (col_arr >= 0) & (col_arr < self.width)
        return np.asarray(inside, dtype=np.bool_)

    def window(self, bounds: Bounds, *, pad: int = 0, clip: bool = True) -> Window:
        """Integer window covering ``bounds`` plus ``pad`` pixels.

        Edges within ``PIXEL_EDGE_DECIMALS`` of a pixel edge snap to it, so the bounds of a
        sub-grid map back to exactly its window. Clipped to the grid unless ``clip=False`` (for
        boundless reads, where the window may start before the first pixel). Raises
        ``ValueError`` when the bounds are degenerate or, when clipping, do not overlap the grid.
        """
        left, bottom, right, top = bounds
        if left >= right or bottom >= top:
            msg = f"degenerate bounds {bounds}"
            raise ValueError(msg)
        col0 = math.floor(pixel_quotient(left, self.transform.c, self.transform.a)) - pad
        col1 = math.ceil(pixel_quotient(right, self.transform.c, self.transform.a)) + pad
        row0 = math.floor(pixel_quotient(top, self.transform.f, self.transform.e)) - pad
        row1 = math.ceil(pixel_quotient(bottom, self.transform.f, self.transform.e)) + pad
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
        """A rasterio GeoTIFF profile for this grid: compressed, tiled when large enough.

        ``nodata`` must be storable in ``dtype`` (``check_nodata``), so an integer profile needs
        an explicit integer nodata or ``None``; the NaN default only suits float dtypes.
        """
        out_dtype = check_nodata(dtype, nodata)
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "dtype": out_dtype.name,
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

    def to_dict(self) -> dict[str, Any]:
        """``crs``, ``transform`` (the six affine coefficients), ``height`` and ``width``.

        Plain strings, floats and ints, so the grid can be stored in JSON or YAML beside a
        product built for it; ``from_dict`` gives back an equal grid.
        """
        return {
            "crs": self.crs,
            "transform": [float(value) for value in self.transform[:6]],
            "height": int(self.height),
            "width": int(self.width),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockGrid:
        """The grid ``to_dict`` described; ``KeyError`` names a missing entry."""
        return cls(
            crs=str(data["crs"]),
            transform=Affine(*(float(value) for value in data["transform"])),
            height=int(data["height"]),
            width=int(data["width"]),
        )


def grids_match(a: BlockGrid, b: BlockGrid, *, tolerance: float = GRID_MATCH_TOLERANCE) -> bool:
    """True when ``a`` and ``b`` are the same grid: same CRS, same shape, same pixel edges.

    Edges are compared with a tolerance, a share of one pixel (``GRID_MATCH_TOLERANCE`` by
    default): no outer edge of ``b`` may lie further than that from the same edge of ``a``,
    which bounds the offset of every pixel between them. Exact equality would refuse the same
    grid described twice, once from float32 coordinates and once from float64; the tolerance
    is far below the half pixel that a wrong registration moves a grid by.
    """
    if a.shape != b.shape or not crs_equal(a.crs, b.crs):
        return False
    xres, yres = a.resolution
    a_left, a_bottom, a_right, a_top = a.bounds
    b_left, b_bottom, b_right, b_top = b.bounds
    off_x = max(abs(a_left - b_left), abs(a_right - b_right))
    off_y = max(abs(a_top - b_top), abs(a_bottom - b_bottom))
    return off_x <= tolerance * xres and off_y <= tolerance * yres


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
    width = math.ceil(pixel_quotient(right, left, xres))
    height = math.ceil(pixel_quotient(bottom, top, -yres))
    return BlockGrid(
        crs=crs,
        transform=Affine(xres, 0.0, left, 0.0, -yres, top),
        height=height,
        width=width,
    )


CoordinateRegistration = Literal["center", "lower_left"]
"""What one coordinate value marks on a regular grid: the middle of its cell, or the cell's
western and southern edges."""

_SPACING_TOLERANCE = 0.01
"""Largest departure of any coordinate step from the mean step, as a share of that step. Loose
enough for coordinates stored as float32, tight enough to refuse a gap or a Gaussian grid."""
_MIN_COORDS = 2


def _even_step(values: NDArray[np.float64], name: str) -> float:
    """The common step of evenly spaced coordinates; raises when they are not."""
    if values.ndim != 1 or values.size < _MIN_COORDS:
        msg = f"{name} must be a 1-D array of at least two coordinates, got shape {values.shape}"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = f"{name} holds NaN or infinite coordinates"
        raise ValueError(msg)
    step = float((values[-1] - values[0]) / (values.size - 1))
    if step == 0 or float(np.abs(np.diff(values) - step).max()) > _SPACING_TOLERANCE * abs(step):
        msg = f"{name} are not evenly spaced"
        raise ValueError(msg)
    return step


def grid_from_coords(
    xs: NDArray[Any], ys: NDArray[Any], crs: str, *, registration: CoordinateRegistration
) -> BlockGrid:
    """The north-up grid whose cells the coordinate arrays of a field label.

    ``xs`` must ascend (west to east) and ``ys`` descend (north to south), which is the order
    of the field's columns and rows; both must be evenly spaced. ``registration`` says what a
    coordinate marks and has no default, because half a cell hangs on it: ``"center"`` puts
    each coordinate in the middle of its cell, ``"lower_left"`` on the cell's western and
    southern edges. A field stored south to north must have its rows, and its ``ys``, reversed
    before it is described here; nothing is flipped silently.
    """
    if registration not in get_args(CoordinateRegistration):
        msg = (
            f"registration must be one of {get_args(CoordinateRegistration)}, got {registration!r}"
        )
        raise ValueError(msg)
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    dx = _even_step(x, "xs")
    dy = _even_step(y, "ys")
    if dx < 0:
        msg = "xs must ascend (west to east)"
        raise ValueError(msg)
    if dy > 0:
        msg = "ys must descend (north to south); reverse the field's rows and its ys first"
        raise ValueError(msg)
    if registration == "center":
        left, top = float(x[0]) - dx / 2, float(y[0]) - dy / 2
    else:
        left, top = float(x[0]), float(y[0]) - dy
    return BlockGrid(
        crs=crs,
        transform=Affine(dx, 0.0, left, 0.0, dy, top),
        height=int(y.size),
        width=int(x.size),
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
