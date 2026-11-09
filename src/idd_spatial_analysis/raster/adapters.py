"""Conversions between ``rasterra.RasterArray`` and this package's ``(array, BlockGrid)`` pair.

Needs the optional ``rasterra`` extra (``uv sync --extra rasterra``); the base package never
imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine

from idd_spatial_analysis.crs import crs_string
from idd_spatial_analysis.frame.grid import BlockGrid

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _rasterra() -> Any:
    try:
        import rasterra  # noqa: PLC0415  (optional extra, imported on use)
    except ImportError as exc:
        msg = "rasterra is not installed; install the 'rasterra' extra"
        raise ImportError(msg) from exc
    return rasterra


def to_raster_array(array: NDArray[Any], grid: BlockGrid, *, nodata: float | None = None) -> Any:
    """A ``RasterArray`` holding ``array`` on ``grid``."""
    data = np.asarray(array)
    if data.shape != grid.shape:
        msg = f"array shape {data.shape} does not match the grid {grid.shape}"
        raise ValueError(msg)
    return _rasterra().RasterArray(
        data, transform=grid.transform, crs=grid.crs, no_data_value=nodata
    )


def from_raster_array(raster: Any) -> tuple[NDArray[Any], BlockGrid]:
    """The ``(array, BlockGrid)`` pair of a ``RasterArray``; raises when it carries no CRS."""
    data = np.asarray(raster.to_numpy())
    if raster.crs is None:
        msg = "RasterArray carries no CRS"
        raise ValueError(msg)
    grid = BlockGrid(
        crs=crs_string(raster.crs),
        transform=Affine(*raster.transform[:6]),
        height=int(data.shape[0]),
        width=int(data.shape[1]),
    )
    return data, grid
