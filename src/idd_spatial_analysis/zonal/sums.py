"""Weighted sums per zone, and their collation across blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd

from idd_spatial_analysis.constants import (
    SLICE_COLUMN,
    WEIGHT_SUM_COLUMN,
    WEIGHTED_SUM_COLUMN,
    ZONE_COLUMN,
    ZONE_DTYPE,
    ZONE_NODATA,
)
from idd_spatial_analysis.tables import require_columns

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

NanPolicy = Literal["keep_weight", "drop_pixel"]
"""What a NaN value at a weighted pixel does: stay in the weight sum, or leave both sums."""

_SUM_COLUMNS = [WEIGHTED_SUM_COLUMN, WEIGHT_SUM_COLUMN]
_IMAGE_NDIM = 2


def check_zone_mask(mask: NDArray[Any]) -> None:
    """Raise unless ``mask`` is a 2-D integer array."""
    if mask.ndim != _IMAGE_NDIM:
        msg = f"mask must be 2-D, got shape {mask.shape}"
        raise ValueError(msg)
    if mask.dtype.kind not in "iu":
        msg = f"mask must hold integer zone ids, got {mask.dtype}"
        raise TypeError(msg)


def zone_index(mask: NDArray[Any]) -> tuple[NDArray[np.int32], NDArray[np.intp]]:
    """The zone ids present (sorted, without ``0``) and each pixel's position in that list.

    The position is meaningless for nodata pixels (it is ``0``), so callers mask those out.
    Binning by position keeps the per-zone arrays as long as the number of zones, not as long
    as the largest id, which can be two billion.
    """
    present = np.unique(mask[mask != ZONE_NODATA]).astype(ZONE_DTYPE)
    index = np.searchsorted(present, mask.ravel())
    return present, np.asarray(index, dtype=np.intp)


def weighted_sums(
    mask: NDArray[Any],
    weight: NDArray[Any],
    value: NDArray[Any],
    *,
    nan_policy: NanPolicy,
) -> pd.DataFrame:
    """Per zone: ``sum(weight * value)`` and ``sum(weight)`` over the zone's pixels.

    ``value`` is 2-D or a 3-D stack ``(slice, rows, cols)``; a stack yields a ``slice`` column.
    Pixels with zone ``0`` or a NaN weight never count. A NaN value at a weighted pixel adds
    nothing to the weighted sum; ``nan_policy="keep_weight"`` keeps its weight in the weight
    sum (the pixel's people exist but have no value), ``"drop_pixel"`` removes it from both.
    Sums accumulate in float64. Zones with no pixel on the mask have no row.
    """
    if nan_policy not in get_args(NanPolicy):
        msg = f"nan_policy must be one of {get_args(NanPolicy)}, got {nan_policy!r}"
        raise ValueError(msg)
    zone = np.asarray(mask)
    w = np.asarray(weight, dtype=np.float64)
    v = np.asarray(value)
    check_zone_mask(zone)
    if w.shape != zone.shape:
        msg = f"weight shape {w.shape} does not match the mask {zone.shape}"
        raise ValueError(msg)
    if v.ndim not in (2, 3) or v.shape[-2:] != zone.shape:
        msg = (
            f"value shape {v.shape} must be (rows, cols) or (slices, rows, cols) matching the mask"
        )
        raise ValueError(msg)
    stack = v.reshape(-1, *zone.shape)
    present, index = zone_index(zone)
    frames: list[pd.DataFrame] = []
    if present.size:
        w_flat = w.ravel()
        base = (zone.ravel() != ZONE_NODATA) & np.isfinite(w_flat)
        for slab_number, slab in enumerate(stack):
            v_flat = slab.astype(np.float64, copy=False).ravel()
            finite_v = np.isfinite(v_flat)
            counts_weight = base if nan_policy == "keep_weight" else base & finite_v
            counts_value = base & finite_v
            weight_sum = np.bincount(
                index[counts_weight], weights=w_flat[counts_weight], minlength=present.size
            )
            weighted = np.bincount(
                index[counts_value],
                weights=w_flat[counts_value] * v_flat[counts_value],
                minlength=present.size,
            )
            frames.append(
                pd.DataFrame(
                    {
                        ZONE_COLUMN: present,
                        SLICE_COLUMN: np.full(present.size, slab_number, dtype=ZONE_DTYPE),
                        WEIGHTED_SUM_COLUMN: weighted,
                        WEIGHT_SUM_COLUMN: weight_sum,
                    }
                )
            )
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            {
                ZONE_COLUMN: pd.Series([], dtype=ZONE_DTYPE),
                SLICE_COLUMN: pd.Series([], dtype=ZONE_DTYPE),
                WEIGHTED_SUM_COLUMN: pd.Series([], dtype="float64"),
                WEIGHT_SUM_COLUMN: pd.Series([], dtype="float64"),
            }
        )
    )
    if v.ndim == _IMAGE_NDIM:
        out = out.drop(columns=[SLICE_COLUMN])
    return out


def collate_blocks(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Per-block ``weighted_sums`` tables added up by zone (and slice) across blocks.

    Every column other than the two sums is a key; a zone that spans blocks gets one row. A
    missing value in a key column is an error (pandas would drop the row without a word).
    """
    tables = list(frames)
    if not tables:
        msg = "collate_blocks needs at least one table"
        raise ValueError(msg)
    for table in tables:
        require_columns(table, [ZONE_COLUMN, *_SUM_COLUMNS], "block table")
    stacked = pd.concat(tables, ignore_index=True)
    keys = [column for column in stacked.columns if column not in _SUM_COLUMNS]
    holes = stacked[keys].isna().any()
    if holes.any():
        msg = f"key columns hold missing values: {holes[holes].index.tolist()}"
        raise ValueError(msg)
    out = stacked.groupby(keys, as_index=False, sort=True)[_SUM_COLUMNS].sum()
    for key in keys:
        if pd.api.types.is_integer_dtype(stacked[key]):
            out[key] = out[key].astype(stacked[key].dtype)
    return out
