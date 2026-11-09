"""Reading windows and mosaics of rasters, and writing GeoTIFFs atomically."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS as RasterioCRS  # noqa: N811
from rasterio.merge import merge

from idd_spatial_analysis.constants import RASTER_FLOAT_DTYPE
from idd_spatial_analysis.crs import crs_equal
from idd_spatial_analysis.frame.bounds import bbox_in_crs
from idd_spatial_analysis.frame.grid import BlockGrid, check_nodata, grid_from_raster

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from idd_spatial_analysis.frame.bounds import Bounds

Raster = tuple[NDArray[Any], BlockGrid]
"""A 2-D array and the grid that places it."""

_INTEGER_KINDS = "iu"
_IMAGE_NDIM = 2
_STACK_NDIM = 3


def _to_float(data: NDArray[Any], nodata: float | None) -> NDArray[np.float32]:
    """``data`` as float32 with the file's nodata value turned into NaN."""
    out = data.astype(RASTER_FLOAT_DTYPE)
    if nodata is not None and not math.isnan(nodata):
        out[data == nodata] = np.nan
    return np.asarray(out, dtype=np.float32)


def read_window(  # noqa: PLR0913  (keyword-only read options)
    path: Path | str,
    bounds: Bounds,
    *,
    bounds_crs: str | None = None,
    band: int = 1,
    pad: int = 0,
    boundless: bool = False,
    as_float: bool = True,
) -> Raster:
    """The pixels of ``path`` covering ``bounds``, with the grid of exactly those pixels.

    ``bounds`` are in the raster's CRS unless ``bounds_crs`` says otherwise. The window is
    clipped to the raster, so a block at the raster's edge comes back smaller than its grid;
    ``boundless=True`` keeps the full window instead and fills the outside with nodata (NaN for
    float rasters without a nodata value; integer rasters must declare one). ``as_float``
    returns float32 with nodata as NaN; ``False`` returns the file's dtype and values untouched.
    """
    grid = grid_from_raster(path)
    if bounds_crs is not None:
        bounds = bbox_in_crs(bounds, bounds_crs, grid.crs)
    window = grid.window(bounds, pad=pad, clip=not boundless)
    with rasterio.open(path) as src:
        nodata = src.nodata
        if boundless:
            fill = nodata
            if fill is None:
                if np.dtype(src.dtypes[band - 1]).kind in _INTEGER_KINDS:
                    msg = f"boundless read of integer raster {path} needs a nodata value"
                    raise ValueError(msg)
                fill = math.nan
            data = src.read(band, window=window, boundless=True, fill_value=fill)
        else:
            data = src.read(band, window=window)
    out = _to_float(data, nodata) if as_float else data
    return out, grid.sub(window)


def mosaic(  # noqa: PLR0913  (keyword-only read options)
    paths: Iterable[Path | str],
    bounds: Bounds,
    resolution: float | tuple[float, float] | None = None,
    *,
    bounds_crs: str | None = None,
    band: int = 1,
    as_float: bool = True,
) -> Raster:
    """One array covering ``bounds`` merged from several rasters in the same CRS.

    Tiles are merged first-wins in the order given. Without ``resolution`` the output lies on
    the first raster's lattice and covers ``bounds`` entirely, the grid ``read_window`` would
    give for the same bounds; with one, the output grid starts at the top-left of ``bounds``
    and the merge is a nearest-neighbour resample. Pixels no raster covers hold the first
    raster's nodata value, or NaN when a float raster declares none; integer rasters must
    declare one.
    """
    files = [Path(path) for path in paths]
    if not files:
        msg = "mosaic needs at least one raster"
        raise ValueError(msg)
    grids = [grid_from_raster(path) for path in files]
    first_grid = grids[0]
    for path, grid in zip(files, grids, strict=True):
        if not crs_equal(grid.crs, first_grid.crs):
            msg = f"{path} is in {grid.crs}, expected {first_grid.crs}"
            raise ValueError(msg)
    if bounds_crs is not None:
        bounds = bbox_in_crs(bounds, bounds_crs, first_grid.crs)
    res: float | tuple[float, float]
    if resolution is None:
        target = first_grid.sub(first_grid.window(bounds, clip=False))
        bounds, res = target.bounds, first_grid.resolution
    else:
        res = resolution
    with rasterio.open(files[0]) as first:
        nodata = first.nodata
        dtype = np.dtype(first.dtypes[band - 1])
    if nodata is None:
        if dtype.kind in _INTEGER_KINDS:
            msg = f"mosaic of integer rasters needs a nodata value; {files[0]} declares none"
            raise ValueError(msg)
        nodata = math.nan
    merged, transform = merge(
        [str(path) for path in files], bounds=bounds, res=res, nodata=nodata, indexes=[band]
    )
    data = merged[0]
    grid = BlockGrid(
        crs=first_grid.crs,
        transform=Affine(*transform[:6]),
        height=data.shape[0],
        width=data.shape[1],
    )
    out = _to_float(data, nodata) if as_float else data
    return out, grid


def atomic_write(path: Path | str, write_fn: Callable[[Path], object]) -> None:
    """Run ``write_fn(tmp)`` and rename ``tmp`` to ``path`` on success.

    The temp file sits beside the target with the extra suffix ``.tmp``; on any exception it is
    removed and the exception re-raised, so the target is never left half written. Same
    contract as ``idd_tools.jobmon.atomic_io.atomic_write``, copied here because that module's
    package imports the jobmon client, which this repo does not declare.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        write_fn(tmp)
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def _narrow(data: NDArray[Any], out_dtype: np.dtype[Any], nodata: float | None) -> NDArray[Any]:
    """``data`` as ``out_dtype`` with nothing silently damaged.

    For an integer target: NaN and infinite pixels become ``nodata`` (refused when there is
    none), non-integral values are refused (round explicitly), and values outside the dtype's
    range are refused. Float targets cast as numpy does.
    """
    if data.dtype == out_dtype:
        return data
    if out_dtype.kind not in _INTEGER_KINDS:
        return data.astype(out_dtype)
    values = data
    if values.dtype.kind == "f":
        finite = np.isfinite(values)
        if not finite.all():
            if nodata is None:
                msg = (
                    f"array holds NaN or infinite pixels and no nodata is given to stand for "
                    f"them in {out_dtype.name}"
                )
                raise ValueError(msg)
            values = np.where(finite, values, nodata)
        if not np.array_equal(values, np.floor(values)):
            msg = f"array holds non-integral values; round it explicitly before writing as {out_dtype.name}"
            raise ValueError(msg)
    info = np.iinfo(out_dtype)
    low, high = values.min(), values.max()
    if low < info.min or high > info.max:
        msg = f"values {low}..{high} do not fit {out_dtype.name} ({info.min}..{info.max})"
        raise ValueError(msg)
    return values.astype(out_dtype)


def _check_header(path: Path, grid: BlockGrid, dtype: np.dtype[Any], count: int) -> None:
    """Compare the written file's header with what was asked for; never reads the pixels."""
    problems: list[str] = []
    with rasterio.open(path) as src:
        if src.shape != grid.shape:
            problems.append(f"shape {src.shape} != {grid.shape}")
        if src.count != count:
            problems.append(f"count {src.count} != {count}")
        if src.dtypes[0] != dtype.name:
            problems.append(f"dtype {src.dtypes[0]} != {dtype.name}")
        if src.crs is None or src.crs != RasterioCRS.from_user_input(grid.crs):
            problems.append(f"crs {src.crs} != {grid.crs}")
        if not np.allclose(np.array(src.transform[:6]), np.array(grid.transform[:6])):
            problems.append(f"transform {src.transform} != {grid.transform}")
    if problems:
        msg = f"{path}: header mismatch after write: {'; '.join(problems)}"
        raise ValueError(msg)


def write_geotiff_atomic(
    path: Path | str,
    array: NDArray[Any],
    grid: BlockGrid,
    *,
    nodata: float | None,
    dtype: str | None = None,
) -> Path:
    """Write ``array`` on ``grid`` as a compressed GeoTIFF through ``atomic_write``.

    ``array`` is 2-D or 3-D with bands first. ``nodata`` must be representable in the output
    dtype (NaN is refused for integer types). Narrowing a float array to an integer dtype maps
    NaN to ``nodata`` and refuses non-integral or out-of-range values. After writing, the
    file's header (shape, count, dtype, CRS, transform) is checked against the request; the
    pixels are never read back.
    """
    data = np.asarray(array)
    if data.ndim == _IMAGE_NDIM:
        data = data[np.newaxis]
    elif data.ndim != _STACK_NDIM:
        msg = f"array must be 2-D or 3-D (bands first), got shape {data.shape}"
        raise ValueError(msg)
    if data.shape[1:] != grid.shape:
        msg = f"array shape {data.shape[1:]} does not match the grid {grid.shape}"
        raise ValueError(msg)
    out_dtype = check_nodata(dtype if dtype is not None else data.dtype, nodata)
    data = _narrow(data, out_dtype, nodata)
    count = int(data.shape[0])
    profile = grid.profile(dtype=out_dtype.name, nodata=nodata, count=count)
    target = Path(path)

    def _write(tmp: Path) -> None:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
        _check_header(tmp, grid, out_dtype, count)

    atomic_write(target, _write)
    return target
