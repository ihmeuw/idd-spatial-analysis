"""Regridding: warps through rasterio, or pixel-centre nearest lookups built once, applied per slice.

Pixel-centre nearest: a target pixel takes the value of the source pixel its centre falls in.
That is the convention every idd repo uses when moving between the population frame and the
climate grids; it conserves nothing and is only meaningful when the target is coarser than or
comparable to the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

from idd_spatial_analysis.crs import crs_equal
from idd_spatial_analysis.frame.grid import pixel_quotient

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame.grid import BlockGrid

_OUTSIDE = -1
_IMAGE_NDIM = 2

CentreTransform = Literal["same_crs", "row_and_column", "every_pixel"]
"""How the target pixel centres reached the source CRS. ``"same_crs"``: no transform.
``"row_and_column"``: one row and one column of centres were transformed, because between the
two CRSs x depends on the column alone and y on the row alone (cylindrical projections of the
same datum: the equal-area block frame against lon/lat). ``"every_pixel"``: every centre was
transformed on its own."""

_SEPARABLE_SAMPLE = 33
"""Lattice positions per axis, both ends included, at which the row-and-column coordinates are
compared with a transform of the point itself before they are trusted."""


def _floor_index(values: NDArray[Any], origin: float, step: float, size: int) -> NDArray[np.int32]:
    """Pixel index of each coordinate along one axis; -1 when outside or not finite.

    Uses the same edge tolerance as ``BlockGrid.rowcol``, so a centre that lands on a source
    pixel edge takes the pixel to its right or below on every lattice.
    """
    with np.errstate(invalid="ignore"):
        index = np.floor(pixel_quotient(values, origin, step))
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


def _sample_positions(size: int) -> NDArray[np.intp]:
    """Up to ``_SEPARABLE_SAMPLE`` indices spread evenly over ``range(size)``, both ends included."""
    count = min(size, _SEPARABLE_SAMPLE)
    return np.unique(np.linspace(0, size - 1, count).round().astype(np.intp))


def _row_and_column_centres(
    src_crs: str, dst_crs: str, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """The lattice ``xs`` by ``ys`` in ``dst_crs`` as one x per column and one y per row.

    Returned only when the lattice supports it: the middle row and the middle column are
    transformed, then a sample of lattice points spread over the whole lattice (edges and
    corners included) is transformed point by point, and every sampled coordinate must equal
    the row-and-column one bit for bit. Any difference, or a coordinate that is not finite
    along the middle row or column, gives ``None`` and the caller transforms every point.

    The equality is exact because a transform that moves x and y independently computes each
    with the same arithmetic whatever the other coordinate is; measured on the equal-area
    block frame against lon/lat, in both directions. It is a check on a sample, not a proof:
    a pair of CRSs that agreed at every sampled point and differed between them would pass.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    mid_x, mid_y = xs[xs.size // 2], ys[ys.size // 2]
    x_of_col = np.asarray(transformer.transform(xs, np.full(xs.shape, mid_y))[0], dtype=np.float64)
    y_of_row = np.asarray(transformer.transform(np.full(ys.shape, mid_x), ys)[1], dtype=np.float64)
    if not (np.isfinite(x_of_col).all() and np.isfinite(y_of_row).all()):
        return None
    cols, rows = _sample_positions(xs.size), _sample_positions(ys.size)
    grid_x, grid_y = np.meshgrid(xs[cols], ys[rows])
    sample_x, sample_y = transformer.transform(grid_x.ravel(), grid_y.ravel())
    same = np.array_equal(
        np.asarray(sample_x, dtype=np.float64).reshape(grid_x.shape),
        np.broadcast_to(x_of_col[cols], grid_x.shape),
    ) and np.array_equal(
        np.asarray(sample_y, dtype=np.float64).reshape(grid_x.shape),
        np.broadcast_to(y_of_row[rows][:, np.newaxis], grid_x.shape),
    )
    return (x_of_col, y_of_row) if same else None


@dataclass(frozen=True, eq=False)
class NearestIndexMap:
    """Source pixel for each target pixel (centre-in-pixel), built once, applied per slice.

    Three layouts share one ``apply``: per-axis 1-D ``rows``/``cols`` (no mask; the same CRS,
    or two CRSs between which rows and columns transform independently), full 2-D arrays (no
    mask, any other pair of CRSs), or 1-D arrays aligned with ``selected`` flat target indices
    (a mask was given). ``-1`` marks a target pixel whose centre falls outside the source
    grid. ``centre_transform`` says how the centres reached the source CRS.
    """

    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    centre_transform: CentreTransform
    selected: NDArray[np.intp] | None = None

    @classmethod
    def build(
        cls,
        source: BlockGrid,
        target: BlockGrid,
        *,
        target_mask: NDArray[np.bool_] | None = None,
        allow_row_and_column: bool = True,
    ) -> NearestIndexMap:
        """The lookup from ``target`` into ``source``; with ``target_mask`` only for True pixels.

        Between two different CRSs the target's pixel centres must be moved into the source
        CRS. When x depends on the column alone and y on the row alone, one row and one
        column of centres give every pixel's coordinates: a few thousand transforms for a
        block instead of tens of millions, with the same result. That route is taken only
        after a sample of points transformed on their own agrees with it exactly
        (``_row_and_column_centres``); otherwise, or with ``allow_row_and_column=False``,
        every centre is transformed.
        """
        xs, ys = target.pixel_centers()
        axes: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        how: CentreTransform = "every_pixel"
        if crs_equal(source.crs, target.crs):
            axes, how = (xs, ys), "same_crs"
        elif allow_row_and_column:
            axes = _row_and_column_centres(target.crs, source.crs, xs, ys)
            how = "every_pixel" if axes is None else "row_and_column"

        selected: NDArray[np.intp] | None = None
        if target_mask is not None:
            mask = np.asarray(target_mask, dtype=bool)
            if mask.shape != target.shape:
                msg = f"target_mask shape {mask.shape} does not match the target {target.shape}"
                raise ValueError(msg)
            selected = np.flatnonzero(mask)

        transform = source.transform
        if axes is not None:
            row_of_row = _floor_index(axes[1], transform.f, transform.e, source.height)
            col_of_col = _floor_index(axes[0], transform.c, transform.a, source.width)
            if selected is None:
                return cls(source.shape, target.shape, row_of_row, col_of_col, how)
            target_rows, target_cols = np.divmod(selected, target.width)
            rows, cols = row_of_row[target_rows], col_of_col[target_cols]
            _joint_outside(rows, cols)
            return cls(source.shape, target.shape, rows, cols, how, selected)

        if selected is None:
            grid_x, grid_y = np.meshgrid(xs, ys)
            x, y = grid_x.ravel(), grid_y.ravel()
        else:
            target_rows, target_cols = np.divmod(selected, target.width)
            x, y = xs[target_cols], ys[target_rows]
        x, y = _transform_points(target.crs, source.crs, x, y)
        rows = _floor_index(y, transform.f, transform.e, source.height)
        cols = _floor_index(x, transform.c, transform.a, source.width)
        _joint_outside(rows, cols)
        if selected is None:
            rows, cols = rows.reshape(target.shape), cols.reshape(target.shape)
        return cls(source.shape, target.shape, rows, cols, how, selected)

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
    in the array's dtype; it is NaN by default, so integer arrays need an explicit value. With
    ``nodata=None`` on a float array, target pixels the source does not reach are still NaN.
    """
    src = np.asarray(array)
    if src.ndim < _IMAGE_NDIM or src.shape[-2:] != grid.shape:
        msg = f"array shape {src.shape} does not end with the grid shape {grid.shape}"
        raise ValueError(msg)
    if src.dtype.kind in "iub" and (nodata is None or not float(nodata).is_integer()):
        msg = f"integer array needs an integer nodata, got {nodata!r}"
        raise ValueError(msg)
    method = Resampling[resampling] if isinstance(resampling, str) else resampling
    fill = math.nan if nodata is None else nodata
    out = np.full((*src.shape[:-2], *target.shape), fill, dtype=src.dtype)
    reproject(
        source=src,
        destination=out,
        src_transform=grid.transform,
        src_crs=grid.crs,
        src_nodata=nodata,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=nodata,
        init_dest_nodata=nodata is not None,
        resampling=method,
    )
    return out
