"""Regridding: warps through rasterio, or pixel-centre nearest lookups built once, applied per slice.

Pixel-centre nearest: a target pixel takes the value of the source pixel its centre falls in.
That is the convention every idd repo uses when moving between the population frame and the
climate grids; it conserves nothing and is only meaningful when the target is coarser than or
comparable to the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

from idd_spatial_analysis.crs import crs_equal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame.grid import BlockGrid

_OUTSIDE = -1
_IMAGE_NDIM = 2


def _floor_index(values: NDArray[Any], origin: float, step: float, size: int) -> NDArray[np.int32]:
    """Pixel index of each coordinate along one axis; -1 when outside or not finite."""
    with np.errstate(invalid="ignore"):
        index = np.floor((np.asarray(values, dtype=np.float64) - origin) / step)
    valid = np.isfinite(index) & (index >= 0) & (index < size)
    return np.asarray(np.where(valid, index, _OUTSIDE), dtype=np.int32)


def _transform_points(
    src_crs: str, dst_crs: str, xs: NDArray[Any], ys: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out_x, out_y = transformer.transform(xs, ys)
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def _joint_outside(rows: NDArray[np.int32], cols: NDArray[np.int32]) -> None:
    outside = (rows < 0) | (cols < 0)
    rows[outside] = _OUTSIDE
    cols[outside] = _OUTSIDE


@dataclass(frozen=True, eq=False)
class NearestIndexMap:
    """Source pixel for each target pixel (centre-in-pixel), built once, applied per slice.

    Three layouts share one ``apply``: separable 1-D ``rows``/``cols`` (same CRS, no mask),
    full 2-D arrays (different CRS, no mask), or 1-D arrays aligned with ``selected`` flat
    target indices (a mask was given). ``-1`` marks a target pixel whose centre falls outside
    the source grid.
    """

    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    selected: NDArray[np.intp] | None = None

    @classmethod
    def build(
        cls,
        source: BlockGrid,
        target: BlockGrid,
        *,
        target_mask: NDArray[np.bool_] | None = None,
    ) -> NearestIndexMap:
        """The lookup from ``target`` into ``source``; with ``target_mask`` only for True pixels."""
        same_crs = crs_equal(source.crs, target.crs)
        xs, ys = target.pixel_centers()
        if target_mask is not None:
            mask = np.asarray(target_mask, dtype=bool)
            if mask.shape != target.shape:
                msg = f"target_mask shape {mask.shape} does not match the target {target.shape}"
                raise ValueError(msg)
            selected = np.flatnonzero(mask)
            target_rows, target_cols = np.divmod(selected, target.width)
            x, y = xs[target_cols], ys[target_rows]
            if not same_crs:
                x, y = _transform_points(target.crs, source.crs, x, y)
            rows = _floor_index(y, source.transform.f, source.transform.e, source.height)
            cols = _floor_index(x, source.transform.c, source.transform.a, source.width)
            _joint_outside(rows, cols)
            return cls(source.shape, target.shape, rows, cols, selected)
        if same_crs:
            rows = _floor_index(ys, source.transform.f, source.transform.e, source.height)
            cols = _floor_index(xs, source.transform.c, source.transform.a, source.width)
            return cls(source.shape, target.shape, rows, cols)
        grid_x, grid_y = np.meshgrid(xs, ys)
        x, y = _transform_points(target.crs, source.crs, grid_x.ravel(), grid_y.ravel())
        rows = _floor_index(y, source.transform.f, source.transform.e, source.height)
        cols = _floor_index(x, source.transform.c, source.transform.a, source.width)
        _joint_outside(rows, cols)
        return cls(
            source.shape, target.shape, rows.reshape(target.shape), cols.reshape(target.shape)
        )

    @property
    def masked(self) -> bool:
        return self.selected is not None

    def indices(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Source ``(row, col)`` of every target pixel as two 2-D arrays, -1 outside the source."""
        if self.selected is not None:
            msg = "built for a mask; source indices exist only at the masked pixels"
            raise ValueError(msg)
        if self.rows.ndim == 1:
            rows = np.ascontiguousarray(
                np.broadcast_to(self.rows[:, np.newaxis], self.target_shape)
            )
            cols = np.ascontiguousarray(
                np.broadcast_to(self.cols[np.newaxis, :], self.target_shape)
            )
            _joint_outside(rows, cols)
            return rows, cols
        return self.rows, self.cols

    def apply(self, array: NDArray[Any], *, fill: float = math.nan) -> NDArray[Any]:
        """``array`` (..., rows, cols) on the source grid resampled onto the target; ``fill`` elsewhere."""
        src = np.asarray(array)
        if src.ndim < _IMAGE_NDIM or src.shape[-2:] != self.source_shape:
            msg = f"array shape {src.shape} does not end with the source shape {self.source_shape}"
            raise ValueError(msg)
        if src.dtype.kind in "iub" and not float(fill).is_integer():
            msg = f"fill {fill!r} is not representable in {src.dtype.name}"
            raise ValueError(msg)
        leading = src.shape[:-2]
        out = np.full((*leading, *self.target_shape), fill, dtype=src.dtype)
        if self.selected is not None:
            ok = self.rows >= 0
            flat = out.reshape(*leading, -1)
            flat[..., self.selected[ok]] = src[..., self.rows[ok], self.cols[ok]]
        elif self.rows.ndim == 1:
            ok_rows, ok_cols = self.rows >= 0, self.cols >= 0
            out[(..., *np.ix_(ok_rows, ok_cols))] = src[
                (..., *np.ix_(self.rows[ok_rows], self.cols[ok_cols]))
            ]
        else:
            ok = self.rows >= 0
            out[..., ok] = src[..., self.rows[ok], self.cols[ok]]
        return out


def resample_masked(
    array: NDArray[Any],
    source: BlockGrid,
    target: BlockGrid,
    target_mask: NDArray[np.bool_],
    *,
    fill: float = math.nan,
) -> NDArray[Any]:
    """Pixel-centre nearest resampling computed only where ``target_mask`` is True.

    The lookup covers the masked pixels alone, so cost scales with the mask, not the target.
    For repeated slices on the same grids build a ``NearestIndexMap`` once and ``apply`` it.
    """
    return NearestIndexMap.build(source, target, target_mask=target_mask).apply(array, fill=fill)


def reproject_to_grid(
    array: NDArray[Any],
    grid: BlockGrid,
    target: BlockGrid,
    *,
    resampling: str | Resampling = "nearest",
    nodata: float | None = math.nan,
) -> NDArray[Any]:
    """``array`` on ``grid`` warped onto ``target`` with ``rasterio.warp.reproject``.

    ``resampling`` names a rasterio method (``nearest``, ``bilinear``, ``average``, ``sum``,
    ...). ``nodata`` marks missing pixels in both source and output and must be representable
    in the array's dtype; it is NaN by default, so integer arrays need an explicit value.
    """
    src = np.asarray(array)
    if src.ndim < _IMAGE_NDIM or src.shape[-2:] != grid.shape:
        msg = f"array shape {src.shape} does not end with the grid shape {grid.shape}"
        raise ValueError(msg)
    if src.dtype.kind in "iub" and (nodata is None or not float(nodata).is_integer()):
        msg = f"integer array needs an integer nodata, got {nodata!r}"
        raise ValueError(msg)
    method = Resampling[resampling] if isinstance(resampling, str) else resampling
    out = np.full((*src.shape[:-2], *target.shape), nodata, dtype=src.dtype)
    reproject(
        source=src,
        destination=out,
        src_transform=grid.transform,
        src_crs=grid.crs,
        src_nodata=nodata,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=nodata,
        resampling=method,
    )
    return out
