"""Sparse zone-by-coarse-cell weights: how a coarse grid's cells map onto zones."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from idd_spatial_analysis.constants import (
    COARSE_COL_COLUMN,
    COARSE_ROW_COLUMN,
    WEIGHT_COLUMN,
    ZONE_COLUMN,
    ZONE_DTYPE,
    ZONE_NODATA,
)
from idd_spatial_analysis.raster.regrid import NearestIndexMap
from idd_spatial_analysis.zonal.sums import check_zone_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame.grid import BlockGrid


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
    which drops such pixels. Applying these weights to a coarse field gives weighted sums per
    zone without touching the fine grid again.
    """
    zone = np.asarray(mask)
    w = np.asarray(weight, dtype=np.float64)
    check_zone_mask(zone)
    if zone.shape != grid.shape or w.shape != grid.shape:
        msg = f"mask {zone.shape} and weight {w.shape} must match the grid {grid.shape}"
        raise ValueError(msg)
    coarse_rows, coarse_cols = NearestIndexMap.build(coarse, grid).indices()
    counted = (zone != ZONE_NODATA) & np.isfinite(w)
    covered = coarse_rows >= 0
    uncovered = counted & ~covered
    if uncovered.any() and not allow_uncovered:
        msg = (
            f"{int(uncovered.sum())} zoned pixels carrying {w[uncovered].sum():.6g} of weight "
            "fall outside the coarse grid; enlarge it or pass allow_uncovered=True"
        )
        raise ValueError(msg)
    valid = counted & covered
    n_cells = coarse.height * coarse.width
    cell = coarse_rows[valid].astype(np.int64) * coarse.width + coarse_cols[valid].astype(np.int64)
    key = zone[valid].astype(np.int64) * n_cells + cell
    unique_keys, inverse = np.unique(key, return_inverse=True)
    sums = np.bincount(inverse, weights=w[valid], minlength=unique_keys.size)
    zones_out = unique_keys // n_cells
    cells_out = unique_keys % n_cells
    return pd.DataFrame(
        {
            ZONE_COLUMN: zones_out.astype(ZONE_DTYPE),
            COARSE_ROW_COLUMN: (cells_out // coarse.width).astype(np.int32),
            COARSE_COL_COLUMN: (cells_out % coarse.width).astype(np.int32),
            WEIGHT_COLUMN: sums.astype(np.float64),
        }
    )
