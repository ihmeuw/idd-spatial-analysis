"""Zone cell weights: a weight raster summed onto (zone, coarse cell) pairs, block by block.

For a field on a grid coarser than the block frame, every pixel takes the value of one coarse
cell, so a weighted sum over a zone's pixels equals a sum over cells of the cell's value times
the zone's weight inside that cell. This product stores those weights for one zone set, one
coarse grid and any number of weight rasters, each named by a ``label`` (the year, for yearly
population).

A zone cell weights directory::

    block_weights/<label>/<block_key>.parquet   one block and one weight raster
    weights/<label>.parquet                     the blocks of one label added up

Every file has the columns ``zone``, ``coarse_row``, ``coarse_col`` and ``weight`` and carries
the coarse grid it was built for in its schema metadata, so a table can never be applied to a
field without that grid being known.

The work comes in rounds, and each file has exactly one writer:

1. per block: ``zonal.ZoneCellPairs.build`` once, then ``compute_zone_cell_weights_block``
   for each weight raster of the batch, each table written to its own ``block_weights`` file;
2. per label, once its blocks are there: ``collect_zone_cell_weights`` adds them up and
   writes ``weights/<label>.parquet``;
3. any number of times: ``load_zone_cell_weights(directory, label).apply(field, field_grid)``.

Computing and writing are separate steps, as in ``lookup.zone_lookup``. The ``compute_`` and
``combine_`` functions return arrow tables with explicit column types and the parameters in
the schema metadata; ``build_zone_cell_weights_block`` and ``collect_zone_cell_weights`` add
this package's atomic write. A job that must write through another atomic writer (a jobmon
worker using idd-tools' registry) writes the same tables to the same paths itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from idd_spatial_analysis import __version__
from idd_spatial_analysis.constants import (
    COARSE_COL_COLUMN,
    COARSE_ROW_COLUMN,
    WEIGHT_COLUMN,
    ZONE_CELL_WEIGHTS_BLOCKS_DIR,
    ZONE_CELL_WEIGHTS_DIR,
    ZONE_COLUMN,
    ZONE_DTYPE,
)
from idd_spatial_analysis.frame.grid import BlockGrid, grids_match
from idd_spatial_analysis.lookup._io import (
    check_file_stem,
    preview,
    read_parquet_checked,
    write_parquet_atomic,
)
from idd_spatial_analysis.zonal.weights import apply_zone_weights

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    from numpy.typing import NDArray

    from idd_spatial_analysis.zonal.sums import NanPolicy
    from idd_spatial_analysis.zonal.weights import ZoneCellPairs

PRODUCT = "zone_cell_weights"
SCHEMA_VERSION = 1
_PARAMS_KEY = b"idd_spatial_analysis.zone_cell_weights"
_SHARED_KEYS = ("product", "schema_version", "coarse", "fine", "package_version")
"""The parameters every block of one table must share."""

_SCHEMA = pa.schema(
    [
        (ZONE_COLUMN, pa.int32()),
        (COARSE_ROW_COLUMN, pa.int32()),
        (COARSE_COL_COLUMN, pa.int32()),
        (WEIGHT_COLUMN, pa.float64()),
    ]
)


def zone_cell_weights_block_path(directory: Path | str, label: str, block_key: str) -> Path:
    """Where the weights of ``block_key`` for the weight raster ``label`` live under ``directory``."""
    check_file_stem(label, "label")
    check_file_stem(block_key, "block_key")
    return Path(directory) / ZONE_CELL_WEIGHTS_BLOCKS_DIR / label / f"{block_key}.parquet"


def zone_cell_weights_path(directory: Path | str, label: str) -> Path:
    """Where the collected weights of the weight raster ``label`` live under ``directory``."""
    check_file_stem(label, "label")
    return Path(directory) / ZONE_CELL_WEIGHTS_DIR / f"{label}.parquet"


def _with_params(table: pa.Table, params: dict[str, Any]) -> pa.Table:
    return table.replace_schema_metadata({_PARAMS_KEY: json.dumps(params).encode()})


def _params_of(schema: pa.Schema, where: str) -> dict[str, Any]:
    """The parameters stored with a table; refuses a table of another product or schema version."""
    raw = (schema.metadata or {}).get(_PARAMS_KEY)
    if raw is None:
        msg = f"{where} carries no zone cell weights parameters"
        raise ValueError(msg)
    params: dict[str, Any] = json.loads(raw)
    if params.get("product") != PRODUCT or params.get("schema_version") != SCHEMA_VERSION:
        msg = (
            f"{where} describes product {params.get('product')!r} schema "
            f"{params.get('schema_version')!r}; this code reads {PRODUCT!r} schema {SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return params


def _checked(table: pa.Table, where: str) -> dict[str, Any]:
    """The parameters of ``table`` once its columns and types are known to be this product's."""
    if not table.schema.equals(_SCHEMA, check_metadata=False):
        msg = f"{where}: schema {table.schema} is not the expected {_SCHEMA}"
        raise ValueError(msg)
    return _params_of(table.schema, where)


def compute_zone_cell_weights_block(
    pairs: ZoneCellPairs, weight: NDArray[Any], *, allow_uncovered: bool = False
) -> pa.Table:
    """One block and one weight raster: ``weight`` summed onto the block's (zone, cell) pairs.

    ``pairs`` is the block's ``zonal.ZoneCellPairs``, built once and used for every weight
    raster of the block; ``weight`` is on the block's grid. The rows are
    ``ZoneCellPairs.weight_table``'s: a pair appears when at least one of its pixels has a
    finite weight, and a zoned pixel with a finite weight outside the coarse grid is an error
    unless ``allow_uncovered=True``.

    The table's schema metadata records the coarse grid, the block frame's CRS and
    resolution, and what happened in this block: its zoned pixels, the pixels and weight the
    coarse grid did not cover, the total weight, and how the pixel centres were transformed.
    Nothing is written.
    """
    frame = pairs.weight_table(weight, allow_uncovered=allow_uncovered)
    uncovered_pixels, uncovered_weight = pairs.uncovered_weight(weight)
    xres, yres = pairs.grid.resolution
    params: dict[str, Any] = {
        "product": PRODUCT,
        "schema_version": SCHEMA_VERSION,
        "coarse": pairs.coarse.to_dict(),
        "fine": {"crs": pairs.grid.crs, "resolution": [xres, yres]},
        "package_version": __version__,
        "block": {
            "zoned_pixels": int(pairs.pixels.size + pairs.uncovered.size),
            "pair_count": len(frame),
            "weight_total": float(frame[WEIGHT_COLUMN].sum()),
            "uncovered_pixels": uncovered_pixels,
            "uncovered_weight": uncovered_weight,
            "centre_transform": pairs.centre_transform,
        },
    }
    arrays = [pa.array(frame[field.name].to_numpy(), type=field.type) for field in _SCHEMA]
    return _with_params(pa.Table.from_arrays(arrays, schema=_SCHEMA), params)


def build_zone_cell_weights_block(  # noqa: PLR0913  (pairs, weight and destination are all needed)
    pairs: ZoneCellPairs,
    weight: NDArray[Any],
    directory: Path | str,
    label: str,
    block_key: str,
    *,
    allow_uncovered: bool = False,
) -> pd.DataFrame:
    """``compute_zone_cell_weights_block``, written to the block's file with an atomic write.

    Returns the table as a DataFrame. A file of the same block and label is replaced.
    """
    path = zone_cell_weights_block_path(directory, label, block_key)
    table = compute_zone_cell_weights_block(pairs, weight, allow_uncovered=allow_uncovered)
    write_parquet_atomic(table, path)
    out: pd.DataFrame = table.to_pandas()
    return out


def combine_zone_cell_weights(
    tables: Iterable[pa.Table],
    *,
    label: str,
    zone_set: str,
    weight_source: str | None = None,
) -> pa.Table:
    """The block tables of one weight raster added up into the table ``apply`` uses.

    A zone that straddles blocks can share a coarse cell across the boundary; its weights in
    that cell add. Rows are ordered by zone, then coarse row, then coarse column. Raises when
    the tables were built for different coarse grids or block frames, or by different
    versions of this package. ``label``, ``zone_set`` and ``weight_source`` (free text: where
    the weight raster came from) are recorded in the table's schema metadata with the counts.
    Nothing is written.
    """
    blocks = list(tables)
    if not blocks:
        msg = "no block tables to combine"
        raise ValueError(msg)
    shared: dict[str, Any] | None = None
    stats: list[dict[str, Any]] = []
    for number, table in enumerate(blocks):
        params = _checked(table, f"block table {number}")
        own = {key: params[key] for key in _SHARED_KEYS}
        if shared is None:
            shared = own
        elif own != shared:
            msg = f"block table {number} was built with {own}, earlier tables with {shared}"
            raise ValueError(msg)
        stats.append(params["block"])
    if shared is None:  # unreachable: blocks is not empty; kept for the type checker
        msg = "no block table was read"
        raise ValueError(msg)

    coarse = BlockGrid.from_dict(shared["coarse"])
    stacked = pa.concat_tables([table.replace_schema_metadata(None) for table in blocks])
    zones = stacked[ZONE_COLUMN].to_numpy().astype(np.int64)
    rows = stacked[COARSE_ROW_COLUMN].to_numpy().astype(np.int64)
    cols = stacked[COARSE_COL_COLUMN].to_numpy().astype(np.int64)
    n_cells = coarse.height * coarse.width
    keys, position = np.unique(zones * n_cells + rows * coarse.width + cols, return_inverse=True)
    weights = np.bincount(
        position, weights=stacked[WEIGHT_COLUMN].to_numpy(), minlength=keys.size
    ).astype(np.float64)
    cells = keys % n_cells
    combined = pa.Table.from_arrays(
        [
            pa.array((keys // n_cells).astype(ZONE_DTYPE), type=pa.int32()),
            pa.array((cells // coarse.width).astype(np.int32), type=pa.int32()),
            pa.array((cells % coarse.width).astype(np.int32), type=pa.int32()),
            pa.array(weights, type=pa.float64()),
        ],
        schema=_SCHEMA,
    )
    collected: dict[str, Any] = {
        "label": label,
        "zone_set": zone_set,
        "weight_source": weight_source,
        "block_count": len(blocks),
        "pair_count": int(keys.size),
        "zone_count": int(np.unique(keys // n_cells).size),
        "weight_total": float(weights.sum()),
        "zoned_pixels": sum(int(block["zoned_pixels"]) for block in stats),
        "uncovered_pixels": sum(int(block["uncovered_pixels"]) for block in stats),
        "uncovered_weight": float(sum(float(block["uncovered_weight"]) for block in stats)),
        "centre_transforms": sorted({str(block["centre_transform"]) for block in stats}),
        "collected_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    return _with_params(combined, {**shared, "collected": collected})


def _check_blocks_present(directory: Path, label: str, keys: list[str]) -> None:
    """Every key has its table for ``label``, and the label's directory holds no other block."""
    missing = [
        key for key in keys if not zone_cell_weights_block_path(directory, label, key).exists()
    ]
    if missing:
        msg = (
            f"{len(missing)} blocks have no weights for {label!r} in {directory}: "
            f"{preview(missing)}"
        )
        raise FileNotFoundError(msg)
    block_dir = directory / ZONE_CELL_WEIGHTS_BLOCKS_DIR / label
    extra = {path.stem for path in block_dir.glob("*.parquet")} - set(keys)
    if extra:
        msg = (
            f"{block_dir} holds {len(extra)} blocks that are not in block_keys: "
            f"{preview(extra)}; they belong to another build"
        )
        raise ValueError(msg)


def collect_zone_cell_weights(
    directory: Path | str,
    label: str,
    block_keys: Iterable[str],
    *,
    zone_set: str,
    weight_source: str | None = None,
) -> pd.DataFrame:
    """Check that every block of ``label`` is built, add them up and write ``weights/<label>.parquet``.

    ``block_keys`` are the blocks the table is meant to cover. Raises when a block has no
    table for ``label``, when the label's directory holds a block that is not in
    ``block_keys``, or when ``combine_zone_cell_weights`` refuses the tables. Returns the
    collected table as a DataFrame.
    """
    root = Path(directory)
    keys = sorted(block_keys)
    if not keys:
        msg = "block_keys is empty"
        raise ValueError(msg)
    if len(set(keys)) != len(keys):
        msg = "block_keys holds duplicates"
        raise ValueError(msg)
    _check_blocks_present(root, label, keys)
    tables = [pq.read_table(zone_cell_weights_block_path(root, label, key)) for key in keys]
    combined = combine_zone_cell_weights(
        tables, label=label, zone_set=zone_set, weight_source=weight_source
    )
    write_parquet_atomic(combined, zone_cell_weights_path(root, label))
    out: pd.DataFrame = combined.to_pandas()
    return out


def _describe(grid: BlockGrid) -> str:
    xres, yres = grid.resolution
    left, _, _, top = grid.bounds
    return (
        f"{grid.height} x {grid.width} cells of {xres:g} by {yres:g} in {grid.crs}, "
        f"north-west corner at ({left:.6g}, {top:.6g})"
    )


@dataclass(frozen=True, eq=False)
class ZoneCellWeights:
    """A weight table together with the coarse grid it was built for.

    ``table`` has ``zone``, ``coarse_row``, ``coarse_col`` and ``weight``; ``coarse`` is the
    grid whose rows and columns the table names; ``metadata`` is everything stored with the
    table (the block frame, the counts, the label and sources of a collected table).
    """

    table: pd.DataFrame
    coarse: BlockGrid
    metadata: dict[str, Any]

    @classmethod
    def from_table(cls, table: pa.Table) -> ZoneCellWeights:
        """From an arrow table of this product, a block's or a collected one, held in memory."""
        params = _checked(table, "table")
        frame: pd.DataFrame = table.to_pandas()
        return cls(table=frame, coarse=BlockGrid.from_dict(params["coarse"]), metadata=params)

    def apply(
        self,
        value: NDArray[Any],
        grid: BlockGrid,
        *,
        nan_policy: NanPolicy = "keep_weight",
    ) -> pd.DataFrame:
        """Per zone: ``sum(weight * value)`` and ``sum(weight)``, for a field on the table's grid.

        ``value`` is ``(rows, cols)`` or a stack ``(slices, rows, cols)`` and ``grid`` the grid
        it is on, which must be the grid the table was built for (``frame.grids_match``: same
        CRS, same shape, cell edges within a thousandth of a cell). A field on any other grid
        is refused: the table's rows and columns would name the wrong cells, and a grid moved
        by half a cell gives numbers that look right. Build ``grid`` from the field's own
        coordinates with ``frame.grid_from_coords`` or ``frame.grid_from_raster``.

        The result and ``nan_policy`` are those of ``zonal.apply_zone_weights``.
        """
        if not grids_match(grid, self.coarse):
            msg = (
                f"the field is on {_describe(grid)}; these weights were built for "
                f"{_describe(self.coarse)}. Regrid the field or build weights for its grid"
            )
            raise ValueError(msg)
        field = np.asarray(value)
        if field.shape[-2:] != self.coarse.shape:
            msg = f"value shape {field.shape} does not end with the grid shape {self.coarse.shape}"
            raise ValueError(msg)
        return apply_zone_weights(self.table, field, nan_policy=nan_policy)


def load_zone_cell_weights(directory: Path | str, label: str) -> ZoneCellWeights:
    """The collected weights of ``label`` and the coarse grid recorded with them."""
    path = zone_cell_weights_path(directory, label)
    if not path.exists():
        msg = f"{path} is absent; run collect_zone_cell_weights once every block is built"
        raise FileNotFoundError(msg)
    params = _params_of(pq.read_schema(path), str(path))
    frame: pd.DataFrame = read_parquet_checked(path, _SCHEMA).to_pandas()
    return ZoneCellWeights(
        table=frame, coarse=BlockGrid.from_dict(params["coarse"]), metadata=params
    )
