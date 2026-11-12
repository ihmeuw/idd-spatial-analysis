"""Sparse zone-by-coarse-cell weights: how a coarse grid's cells map onto zones.

Three pieces. ``ZoneCellPairs`` works out, once per block and coarse grid, which coarse cell
every zoned pixel falls in and groups those pixels into (zone, cell) pairs. ``zone_weights``
adds one weight raster onto the pairs. ``apply_zone_weights`` turns a field on the coarse grid
into the two sums per zone that ``weighted_sums`` gives on the fine grid, without touching the
fine grid again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pandas as pd

from idd_spatial_analysis.constants import (
    COARSE_COL_COLUMN,
    COARSE_ROW_COLUMN,
    PIXEL_COUNT_COLUMN,
    SLICE_COLUMN,
    WEIGHT_COLUMN,
    WEIGHT_SUM_COLUMN,
    WEIGHTED_SUM_COLUMN,
    ZONE_COLUMN,
    ZONE_DTYPE,
    ZONE_NODATA,
)
from idd_spatial_analysis.raster.regrid import NearestIndexMap
from idd_spatial_analysis.tables import require_columns
from idd_spatial_analysis.zonal.sums import NanPolicy, check_zone_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame.grid import BlockGrid

_IMAGE_NDIM = 2
_STACK_NDIM = 3


@dataclass(frozen=True, eq=False)
class ZoneCellPairs:
    """The (zone, coarse cell) pairs of one block, and which pair each zoned pixel belongs to.

    Built once per block and coarse grid. A pixel belongs to the coarse cell its centre falls
    in (pixel-centre nearest, any CRS), so the pairing depends on the zone mask and the two
    grids only, not on any weight. ``sum_weight`` then adds a weight raster onto the pairs
    with one pass over the zoned pixels, which is all a further year of population costs.

    ``zones``, ``coarse_rows``, ``coarse_cols`` and ``pixel_counts`` have one entry per pair,
    ordered by zone, then coarse row, then coarse column. ``pixels`` are the flat indices of
    the zoned pixels the coarse grid covers and ``pair_of_pixel`` the pair of each;
    ``uncovered`` are the flat indices of zoned pixels whose centre lies outside the coarse
    grid, which belong to no pair.
    """

    zones: NDArray[np.int32]
    coarse_rows: NDArray[np.int32]
    coarse_cols: NDArray[np.int32]
    pixel_counts: NDArray[np.int64]
    fine_shape: tuple[int, int]
    pixels: NDArray[np.intp]
    pair_of_pixel: NDArray[np.intp]
    uncovered: NDArray[np.intp]

    @classmethod
    def build(cls, mask: NDArray[Any], grid: BlockGrid, coarse: BlockGrid) -> ZoneCellPairs:
        """Pair every zoned pixel of ``mask`` on ``grid`` with the cell of ``coarse`` it lies in."""
        zone = np.asarray(mask)
        check_zone_mask(zone)
        if zone.shape != grid.shape:
            msg = f"mask {zone.shape} must match the grid {grid.shape}"
            raise ValueError(msg)
        lookup = NearestIndexMap.build(coarse, grid, target_mask=zone != ZONE_NODATA)
        zoned = np.asarray(lookup.selected, dtype=np.intp)
        covered = lookup.rows >= 0
        pixels = zoned[covered]
        n_cells = coarse.height * coarse.width
        cell = lookup.rows[covered].astype(np.int64) * coarse.width + lookup.cols[covered]
        key = zone.ravel()[pixels].astype(np.int64) * n_cells + cell
        unique_keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        cells = unique_keys % n_cells
        return cls(
            zones=(unique_keys // n_cells).astype(ZONE_DTYPE),
            coarse_rows=(cells // coarse.width).astype(np.int32),
            coarse_cols=(cells % coarse.width).astype(np.int32),
            pixel_counts=counts.astype(np.int64),
            fine_shape=grid.shape,
            pixels=pixels,
            pair_of_pixel=np.asarray(inverse, dtype=np.intp).ravel(),
            uncovered=zoned[~covered],
        )

    def __len__(self) -> int:
        return int(self.zones.size)

    def _at_pixels(self, weight: NDArray[Any]) -> NDArray[np.float64]:
        """``weight`` at the paired pixels, as a fresh float64 array."""
        raster = np.asarray(weight)
        if raster.shape != self.fine_shape:
            msg = f"weight shape {raster.shape} does not match the block {self.fine_shape}"
            raise ValueError(msg)
        return np.array(raster.ravel()[self.pixels], dtype=np.float64)

    def sum_weight(self, weight: NDArray[Any]) -> NDArray[np.float64]:
        """Per pair, the sum of ``weight`` over its pixels; NaN and infinite weights add nothing."""
        values = self._at_pixels(weight)
        values[~np.isfinite(values)] = 0.0
        sums = np.bincount(self.pair_of_pixel, weights=values, minlength=len(self))
        return np.asarray(sums, dtype=np.float64)

    def finite_count(self, weight: NDArray[Any]) -> NDArray[np.int64]:
        """Per pair, how many of its pixels carry a finite ``weight``."""
        finite = np.isfinite(self._at_pixels(weight))
        counts = np.bincount(self.pair_of_pixel[finite], minlength=len(self))
        return np.asarray(counts, dtype=np.int64)

    def table(self) -> pd.DataFrame:
        """``zone``, ``coarse_row``, ``coarse_col`` and ``pixel_count``, one row per pair."""
        return pd.DataFrame(
            {
                ZONE_COLUMN: self.zones,
                COARSE_ROW_COLUMN: self.coarse_rows,
                COARSE_COL_COLUMN: self.coarse_cols,
                PIXEL_COUNT_COLUMN: self.pixel_counts,
            }
        )


def zone_weights(
    mask: NDArray[Any],
    weight: NDArray[Any],
    grid: BlockGrid,
    coarse: BlockGrid,
    *,
    allow_uncovered: bool = False,
) -> pd.DataFrame:
    """``(zone, coarse_row, coarse_col, weight)``: summed weight of each zone's pixels per coarse cell.

    Each fine pixel is assigned to the coarse cell its centre falls in (pixel-centre nearest,
    any CRS). Pixels with zone ``0`` or a NaN weight do not appear. A zoned, weighted pixel
    whose centre falls outside the coarse grid is an error unless ``allow_uncovered=True``,
    which drops such pixels. Applying these weights to a coarse field with
    ``apply_zone_weights`` gives weighted sums per zone without touching the fine grid again.
    For several weight rasters on the same block (one per year), build ``ZoneCellPairs`` once
    and call its ``sum_weight`` per raster instead.
    """
    zone = np.asarray(mask)
    w = np.asarray(weight, dtype=np.float64)
    check_zone_mask(zone)
    if zone.shape != grid.shape or w.shape != grid.shape:
        msg = f"mask {zone.shape} and weight {w.shape} must match the grid {grid.shape}"
        raise ValueError(msg)
    pairs = ZoneCellPairs.build(zone, grid, coarse)
    lost = w.ravel()[pairs.uncovered]
    weighted = np.isfinite(lost)
    if weighted.any() and not allow_uncovered:
        msg = (
            f"{int(weighted.sum())} zoned pixels carrying {lost[weighted].sum():.6g} of weight "
            "fall outside the coarse grid; enlarge it or pass allow_uncovered=True"
        )
        raise ValueError(msg)
    present = pairs.finite_count(w) > 0
    return pd.DataFrame(
        {
            ZONE_COLUMN: pairs.zones[present],
            COARSE_ROW_COLUMN: pairs.coarse_rows[present],
            COARSE_COL_COLUMN: pairs.coarse_cols[present],
            WEIGHT_COLUMN: pairs.sum_weight(w)[present],
        }
    )


def _empty_sums(*, stacked: bool) -> pd.DataFrame:
    columns: dict[str, Any] = {ZONE_COLUMN: pd.Series([], dtype=ZONE_DTYPE)}
    if stacked:
        columns[SLICE_COLUMN] = pd.Series([], dtype=ZONE_DTYPE)
    columns[WEIGHTED_SUM_COLUMN] = pd.Series([], dtype="float64")
    columns[WEIGHT_SUM_COLUMN] = pd.Series([], dtype="float64")
    return pd.DataFrame(columns)


def apply_zone_weights(
    weights: pd.DataFrame,
    value: NDArray[Any],
    *,
    weight_col: str = WEIGHT_COLUMN,
    nan_policy: NanPolicy = "keep_weight",
) -> pd.DataFrame:
    """Per zone: ``sum(weight * value)`` and ``sum(weight)`` over the zone's coarse cells.

    ``weights`` is a ``zone_weights`` table (``zone``, ``coarse_row``, ``coarse_col`` and
    ``weight_col``) and ``value`` a field on the coarse grid the table was built for: 2-D, or a
    3-D stack ``(slice, rows, cols)``, which yields a ``slice`` column. The result has the
    columns of ``weighted_sums`` and equals what ``weighted_sums`` returns when the field is
    resampled onto the fine grid pixel-centre nearest, because every fine pixel of a pair
    then carries that pair's cell value. It feeds ``collate_blocks`` and ``rollup_hierarchy``
    the same way.

    A NaN cell value adds nothing to the weighted sum. With the default
    ``nan_policy="keep_weight"`` the pair's weight stays in the weight sum, so the missing
    value acts as zero in the mean; ``"drop_pixel"`` leaves the pair out of both sums. The
    table must not name a cell outside ``value``; that means it was built for another grid.
    """
    if nan_policy not in get_args(NanPolicy):
        msg = f"nan_policy must be one of {get_args(NanPolicy)}, got {nan_policy!r}"
        raise ValueError(msg)
    require_columns(
        weights, [ZONE_COLUMN, COARSE_ROW_COLUMN, COARSE_COL_COLUMN, weight_col], "weights"
    )
    field = np.asarray(value)
    if field.ndim not in (_IMAGE_NDIM, _STACK_NDIM):
        msg = f"value must be (rows, cols) or (slices, rows, cols), got shape {field.shape}"
        raise ValueError(msg)
    stacked = field.ndim == _STACK_NDIM
    if weights.empty:
        return _empty_sums(stacked=stacked)
    rows = weights[COARSE_ROW_COLUMN].to_numpy()
    cols = weights[COARSE_COL_COLUMN].to_numpy()
    height, width = field.shape[-2:]
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= height or cols.max() >= width:
        msg = (
            f"weights name cells up to row {rows.max()}, column {cols.max()}, outside a value "
            f"of shape {(height, width)}; the table was built for another grid"
        )
        raise ValueError(msg)
    w = weights[weight_col].to_numpy(dtype=np.float64)
    if not np.isfinite(w).all():
        msg = f"weights column {weight_col!r} holds NaN or infinite weights"
        raise ValueError(msg)
    zones, position = np.unique(weights[ZONE_COLUMN].to_numpy(), return_inverse=True)
    frames: list[pd.DataFrame] = []
    for slab_number, slab in enumerate(field.reshape(-1, height, width)):
        cell_values = np.array(slab[rows, cols], dtype=np.float64)
        finite = np.isfinite(cell_values)
        cell_values[~finite] = 0.0
        counted = w if nan_policy == "keep_weight" else np.where(finite, w, 0.0)
        frame = pd.DataFrame(
            {
                ZONE_COLUMN: zones.astype(ZONE_DTYPE),
                SLICE_COLUMN: np.full(zones.size, slab_number, dtype=ZONE_DTYPE),
                WEIGHTED_SUM_COLUMN: np.bincount(
                    position, weights=w * cell_values, minlength=zones.size
                ),
                WEIGHT_SUM_COLUMN: np.bincount(position, weights=counted, minlength=zones.size),
            }
        )
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out if stacked else out.drop(columns=[SLICE_COLUMN])
