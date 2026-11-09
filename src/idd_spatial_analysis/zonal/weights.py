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

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_spatial_analysis.frame.grid import BlockGrid


def zone_weights(
    mask: NDArray[Any],
    weight: NDArray[Any],
    grid: BlockGrid,
    coarse: BlockGrid,
) -> pd.DataFrame:
    """``(zone, coarse_row, coarse_col, weight)``: summed weight of each zone's pixels per coarse cell.

    Each fine pixel is assigned to the coarse cell its centre falls in (pixel-centre nearest,
    any CRS). Pixels with zone ``0``, a NaN weight, or a centre outside the coarse grid do not
    appear. Applying these weights to a coarse field gives weighted sums per zone without
    touching the fine grid again.
    """
    zone = np.asarray(mask)
    w = np.asarray(weight, dtype=np.float64)
    if zone.shape != grid.shape or w.shape != grid.shape:
        msg = f"mask {zone.shape} and weight {w.shape} must match the grid {grid.shape}"
        raise ValueError(msg)
    if zone.dtype.kind not in "iu":
        msg = f"mask must hold integer zone ids, got {zone.dtype}"
        raise TypeError(msg)
    coarse_rows, coarse_cols = NearestIndexMap.build(coarse, grid).indices()
    valid = (zone != ZONE_NODATA) & np.isfinite(w) & (coarse_rows >= 0)
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
