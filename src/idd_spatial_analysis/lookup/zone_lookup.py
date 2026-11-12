"""Zone lookup rasters: the zone id of every pixel, written once per block and read back.

A zone lookup is a directory::

    blocks/<block_key>.tif            int32, 0 = no zone, on the block's grid
    block_zones/<block_key>.parquet   the zones with pixels in the block: pixel count, window
    zones.parquet                     the block tables collected, with a block_key column
    zone_summary.parquet              every input zone: its pixels and blocks, zeros included
    build.yaml                        how the lookup was built

The first two are written per block by ``build_zone_lookup_block`` (blocks are independent, so
one process per block is fine); the last three by ``collect_zone_lookup`` once every block is
there. With the lookup on disk, a population-weighted mean per zone is ``zonal.weighted_sums``
on a block's mask, weight and value, then ``zonal.collate_blocks`` across blocks: the polygons
are never rasterized again.

Computing a block and writing it are separate steps. ``compute_zone_lookup_block`` returns the
block in memory and ``write_zone_lookup_block`` writes it with this package's atomic writers;
``build_zone_lookup_block`` is the two together. A job that must write through another atomic
writer (a jobmon worker using idd-tools' registry) calls the compute step and writes
``ZoneLookupBlock.write_raster`` and ``ZoneLookupBlock.table`` to the two block paths itself.
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
import rasterio
import yaml

from idd_spatial_analysis import __version__
from idd_spatial_analysis.constants import (
    BLOCK_COUNT_COLUMN,
    BLOCK_KEY_COLUMN,
    COL_OFF_COLUMN,
    HEIGHT_COLUMN,
    PIXEL_COUNT_COLUMN,
    ROW_OFF_COLUMN,
    WIDTH_COLUMN,
    ZONE_COLUMN,
    ZONE_DTYPE,
    ZONE_LOOKUP_BLOCK_ZONES_DIR,
    ZONE_LOOKUP_BLOCKS_DIR,
    ZONE_LOOKUP_INDEX_FILENAME,
    ZONE_LOOKUP_METADATA_FILENAME,
    ZONE_LOOKUP_SUMMARY_FILENAME,
    ZONE_NODATA,
)
from idd_spatial_analysis.frame.grid import grid_from_raster
from idd_spatial_analysis.lookup._io import (
    check_file_stem,
    preview,
    read_parquet_checked,
    write_parquet_atomic,
    write_yaml_atomic,
)
from idd_spatial_analysis.raster.io import write_geotiff, write_geotiff_atomic
from idd_spatial_analysis.raster.zones import rasterize_zones

if TYPE_CHECKING:
    from collections.abc import Iterable

    import geopandas as gpd
    import pandas as pd
    from numpy.typing import NDArray
    from rasterio.windows import Window

    from idd_spatial_analysis.frame.grid import BlockGrid
    from idd_spatial_analysis.raster.io import Raster

PRODUCT = "zone_lookup"
SCHEMA_VERSION = 1
_PARAMS_KEY = b"idd_spatial_analysis.zone_lookup"

_BLOCK_SCHEMA = pa.schema(
    [
        (ZONE_COLUMN, pa.int32()),
        (PIXEL_COUNT_COLUMN, pa.int64()),
        (ROW_OFF_COLUMN, pa.int32()),
        (COL_OFF_COLUMN, pa.int32()),
        (HEIGHT_COLUMN, pa.int32()),
        (WIDTH_COLUMN, pa.int32()),
    ]
)
_INDEX_SCHEMA = pa.schema([(BLOCK_KEY_COLUMN, pa.string()), *_BLOCK_SCHEMA])
_SUMMARY_SCHEMA = pa.schema(
    [
        (ZONE_COLUMN, pa.int32()),
        (PIXEL_COUNT_COLUMN, pa.int64()),
        (BLOCK_COUNT_COLUMN, pa.int32()),
    ]
)


def zone_lookup_block_path(directory: Path | str, block_key: str) -> Path:
    """Where the lookup raster of ``block_key`` lives under ``directory``."""
    check_file_stem(block_key, "block_key")
    return Path(directory) / ZONE_LOOKUP_BLOCKS_DIR / f"{block_key}.tif"


def zone_lookup_block_table_path(directory: Path | str, block_key: str) -> Path:
    """Where the zone table of ``block_key`` lives under ``directory``."""
    check_file_stem(block_key, "block_key")
    return Path(directory) / ZONE_LOOKUP_BLOCK_ZONES_DIR / f"{block_key}.parquet"


def _block_params(zone_col: str, grid: BlockGrid, *, all_touched: bool) -> dict[str, Any]:
    """What must be the same in every block of one lookup; stored with each block's table."""
    xres, yres = grid.resolution
    return {
        "schema_version": SCHEMA_VERSION,
        "zone_col": zone_col,
        "all_touched": all_touched,
        "crs": grid.crs,
        "resolution": [xres, yres],
        "package_version": __version__,
    }


def _block_zone_table(
    mask: NDArray[np.int32], windows: dict[int, Window], params: dict[str, Any]
) -> pa.Table:
    """One row per zone with pixels on ``mask``: its pixel count and its window, ids ascending."""
    present, counts = np.unique(mask, return_counts=True)
    zoned = present != ZONE_NODATA
    zones = present[zoned]
    zone_windows = [windows[int(zone)] for zone in zones]
    columns = {
        ZONE_COLUMN: zones,
        PIXEL_COUNT_COLUMN: counts[zoned],
        ROW_OFF_COLUMN: [int(window.row_off) for window in zone_windows],
        COL_OFF_COLUMN: [int(window.col_off) for window in zone_windows],
        HEIGHT_COLUMN: [int(window.height) for window in zone_windows],
        WIDTH_COLUMN: [int(window.width) for window in zone_windows],
    }
    arrays = [pa.array(columns[field.name], type=field.type) for field in _BLOCK_SCHEMA]
    schema = _BLOCK_SCHEMA.with_metadata({_PARAMS_KEY: json.dumps(params).encode()})
    return pa.Table.from_arrays(arrays, schema=schema)


@dataclass(frozen=True, eq=False)
class ZoneLookupBlock:
    """One block of a zone lookup in memory: the mask, its grid and the block's zone table.

    ``mask`` is int32 with ``0`` for no zone. ``table`` is the arrow table that becomes
    ``block_zones/<block_key>.parquet``: ``zone``, ``pixel_count`` and the window holding the
    zone's pixels, with explicit column types and the build parameters in its schema metadata,
    which ``collect_zone_lookup`` requires. Whoever writes the block must put both on disk,
    the raster first: a table on disk means the block is complete.
    """

    mask: NDArray[np.int32]
    grid: BlockGrid
    table: pa.Table

    def write_raster(self, path: Path | str) -> Path:
        """Write the mask as a lookup raster at exactly ``path``: no temp file, no rename.

        The write step for an atomic writer that supplies those, for example
        ``atomic_write(raster_path, block.write_raster)``.
        """
        return write_geotiff(path, self.mask, self.grid, nodata=ZONE_NODATA, dtype=ZONE_DTYPE)

    def zone_table(self) -> pd.DataFrame:
        """``table`` as a DataFrame with the same column types."""
        out: pd.DataFrame = self.table.to_pandas()
        return out


def compute_zone_lookup_block(
    zones_gdf: gpd.GeoDataFrame,
    zone_col: str,
    grid: BlockGrid,
    *,
    all_touched: bool = False,
) -> ZoneLookupBlock:
    """Rasterize ``zones_gdf`` on ``grid`` and return the block's mask and zone table; no file.

    Membership is ``raster.rasterize_zones``'s: a pixel belongs to the zone containing its
    centre (``all_touched=True``: any zone touching it), the later row wins where zones
    overlap, and ``0`` marks pixels in no zone. ``zones_gdf`` may be the table of the whole
    world; only the zones near the grid are reprojected. A block no zone reaches gives an
    all-zero mask and an empty table.
    """
    mask, windows = rasterize_zones(zones_gdf, zone_col, grid, all_touched=all_touched)
    table = _block_zone_table(mask, windows, _block_params(zone_col, grid, all_touched=all_touched))
    return ZoneLookupBlock(mask=mask, grid=grid, table=table)


def write_zone_lookup_block(block: ZoneLookupBlock, directory: Path | str, block_key: str) -> None:
    """Write ``block`` under ``directory`` with this package's atomic writers.

    The raster goes first and the table second, so a table on disk means the block is
    complete. Files of the block that already exist are replaced.
    """
    raster_path = zone_lookup_block_path(directory, block_key)
    table_path = zone_lookup_block_table_path(directory, block_key)
    write_geotiff_atomic(raster_path, block.mask, block.grid, nodata=ZONE_NODATA, dtype=ZONE_DTYPE)
    write_parquet_atomic(block.table, table_path)


def build_zone_lookup_block(  # noqa: PLR0913  (zones, grid and destination are all needed)
    zones_gdf: gpd.GeoDataFrame,
    zone_col: str,
    grid: BlockGrid,
    directory: Path | str,
    block_key: str,
    *,
    all_touched: bool = False,
) -> pd.DataFrame:
    """Rasterize ``zones_gdf`` on ``grid`` and write the block's lookup raster and zone table.

    ``compute_zone_lookup_block`` followed by ``write_zone_lookup_block``; see the first for
    which zone a pixel gets.

    Returns the block's zone table: ``zone``, ``pixel_count`` and the window (``row_off``,
    ``col_off``, ``height``, ``width``) that holds the zone's pixels in this block. A block no
    zone reaches still gets an all-zero raster and an empty table, so ``collect_zone_lookup``
    can tell an empty block from one that was never built. Files of the block that already
    exist are replaced. The raster is written before the table, so a table on disk means the
    block is complete.
    """
    block = compute_zone_lookup_block(zones_gdf, zone_col, grid, all_touched=all_touched)
    write_zone_lookup_block(block, directory, block_key)
    return block.zone_table()


def read_zone_lookup_block(
    directory: Path | str, block_key: str, *, window: Window | None = None
) -> Raster:
    """The zone ids of one block, or of ``window`` within it, and the grid of those pixels.

    The array is int32 with ``0`` for no zone, ready for ``zonal.weighted_sums`` and
    ``zonal.zone_weights``. ``window`` (for example a zone's window from the index) must lie
    inside the block. Raises ``FileNotFoundError`` naming the block when its raster is absent
    and ``ValueError`` when the file is not a single-band int32 raster with nodata 0.
    """
    path = zone_lookup_block_path(directory, block_key)
    if not path.exists():
        msg = f"zone lookup has no raster for block {block_key!r}: {path}"
        raise FileNotFoundError(msg)
    grid = grid_from_raster(path)
    with rasterio.open(path) as src:
        if src.count != 1 or src.dtypes[0] != ZONE_DTYPE or src.nodata != ZONE_NODATA:
            msg = (
                f"{path} is not a zone lookup raster: {src.count} band(s) of {src.dtypes[0]} "
                f"with nodata {src.nodata}, expected one {ZONE_DTYPE} band with nodata {ZONE_NODATA}"
            )
            raise ValueError(msg)
        if window is None:
            return np.asarray(src.read(1), dtype=ZONE_DTYPE), grid
        row_stop = int(window.row_off) + int(window.height)
        col_stop = int(window.col_off) + int(window.width)
        inside = (
            window.row_off >= 0
            and window.col_off >= 0
            and window.height > 0
            and window.width > 0
            and row_stop <= grid.height
            and col_stop <= grid.width
        )
        if not inside:
            msg = f"window {window} does not lie inside block {block_key!r} of shape {grid.shape}"
            raise ValueError(msg)
        return np.asarray(src.read(1, window=window), dtype=ZONE_DTYPE), grid.sub(window)


def _read_block_table(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    """A block's zone table and the build parameters stored with it."""
    table = read_parquet_checked(path, _BLOCK_SCHEMA)
    raw = (pq.read_schema(path).metadata or {}).get(_PARAMS_KEY)
    if raw is None:
        msg = f"{path} carries no zone lookup build parameters"
        raise ValueError(msg)
    params: dict[str, Any] = json.loads(raw)
    return table, params


def _check_blocks_present(directory: Path, keys: list[str]) -> None:
    """Every key has its raster and its table, and the directory holds no other block."""
    missing = [
        key
        for key in keys
        if not zone_lookup_block_path(directory, key).exists()
        or not zone_lookup_block_table_path(directory, key).exists()
    ]
    if missing:
        msg = f"{len(missing)} blocks are not built in {directory}: {preview(missing)}"
        raise FileNotFoundError(msg)
    found = {path.stem for path in (directory / ZONE_LOOKUP_BLOCKS_DIR).glob("*.tif")}
    found |= {path.stem for path in (directory / ZONE_LOOKUP_BLOCK_ZONES_DIR).glob("*.parquet")}
    extra = found - set(keys)
    if extra:
        msg = (
            f"{directory} holds {len(extra)} blocks that are not in block_keys: "
            f"{preview(extra)}; they belong to another build"
        )
        raise ValueError(msg)


def _unique_int_ids(values: Iterable[int], name: str) -> NDArray[np.int32]:
    ids = np.asarray(list(values))
    if ids.size == 0:
        msg = f"{name} is empty"
        raise ValueError(msg)
    if ids.dtype.kind not in "iu":
        msg = f"{name} must be integers, got {ids.dtype}"
        raise TypeError(msg)
    if np.unique(ids).size != ids.size:
        msg = f"{name} holds duplicates"
        raise ValueError(msg)
    if int(ids.min()) <= ZONE_NODATA or int(ids.max()) > np.iinfo(ZONE_DTYPE).max:
        msg = f"{name} must lie in 1..{np.iinfo(ZONE_DTYPE).max} (0 is nodata)"
        raise ValueError(msg)
    return np.sort(ids).astype(ZONE_DTYPE)


def collect_zone_lookup(
    directory: Path | str,
    block_keys: Iterable[str],
    zone_ids: Iterable[int],
    *,
    zone_set: str,
    zone_source: str | None = None,
) -> pd.DataFrame:
    """Check that every block is built, then write the index, the zone summary and ``build.yaml``.

    ``block_keys`` are the blocks the lookup is meant to cover (``frame.blocks_intersecting``
    for the zones) and ``zone_ids`` every id of the zone table the blocks were built from.
    ``zone_set`` names that table in the sidecar; ``zone_source`` is free text saying where it
    came from. Raises when a block lacks its raster or its table, when the directory holds a
    block that is not in ``block_keys``, when blocks were built with different parameters, or
    when a block names a zone that is not in ``zone_ids``.

    Returns the summary: one row per input zone with its ``pixel_count`` over all blocks and
    its ``block_count``. A zone too small to contain any pixel centre has zeros; it will have
    no row in ``zonal.weighted_sums`` either, which a roll-up must be told
    (``rollup_hierarchy(..., require_complete=False)`` or a ``leaves`` set without it).
    """
    root = Path(directory)
    keys = sorted(block_keys)
    if not keys:
        msg = "block_keys is empty"
        raise ValueError(msg)
    if len(set(keys)) != len(keys):
        msg = "block_keys holds duplicates"
        raise ValueError(msg)
    ids = _unique_int_ids(zone_ids, "zone_ids")
    _check_blocks_present(root, keys)

    tables: list[pa.Table] = []
    params: dict[str, Any] | None = None
    for key in keys:
        table, block_params = _read_block_table(zone_lookup_block_table_path(root, key))
        if params is None:
            params = block_params
        elif block_params != params:
            msg = f"block {key!r} was built with {block_params}, earlier blocks with {params}"
            raise ValueError(msg)
        tables.append(
            table.replace_schema_metadata(None).add_column(
                0, BLOCK_KEY_COLUMN, pa.array([key] * table.num_rows, type=pa.string())
            )
        )
    if params is None:  # unreachable: keys is not empty; kept for the type checker
        msg = "no block table was read"
        raise ValueError(msg)
    index = pa.concat_tables(tables).cast(_INDEX_SCHEMA)
    index_df: pd.DataFrame = index.to_pandas()
    unknown = set(index_df[ZONE_COLUMN].tolist()) - set(ids.tolist())
    if unknown:
        msg = f"{len(unknown)} zones in the blocks are not in zone_ids: {preview(unknown)}"
        raise ValueError(msg)

    per_zone = index_df.groupby(ZONE_COLUMN)[PIXEL_COUNT_COLUMN].agg(["sum", "size"])
    per_zone = per_zone.reindex(ids, fill_value=0)
    summary = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int32()),
            pa.array(per_zone["sum"].to_numpy(), type=pa.int64()),
            pa.array(per_zone["size"].to_numpy(), type=pa.int32()),
        ],
        schema=_SUMMARY_SCHEMA,
    )
    summary_df: pd.DataFrame = summary.to_pandas()

    without_pixels = int((summary_df[PIXEL_COUNT_COLUMN] == 0).sum())
    metadata: dict[str, Any] = {
        "product": PRODUCT,
        "schema_version": params["schema_version"],
        "zone_set": zone_set,
        "zone_source": zone_source,
        "zone_col": params["zone_col"],
        "crs": params["crs"],
        "resolution": params["resolution"],
        "all_touched": params["all_touched"],
        "dtype": ZONE_DTYPE,
        "nodata": ZONE_NODATA,
        "overlap_rule": "the later row of the zone table wins",
        "block_count": len(keys),
        "zone_count": int(ids.size),
        "zones_without_pixels": without_pixels,
        "zoned_pixel_count": int(summary_df[PIXEL_COUNT_COLUMN].sum()),
        "package_version": params["package_version"],
        "collected_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    write_parquet_atomic(index, root / ZONE_LOOKUP_INDEX_FILENAME)
    write_parquet_atomic(summary, root / ZONE_LOOKUP_SUMMARY_FILENAME)
    write_yaml_atomic(metadata, root / ZONE_LOOKUP_METADATA_FILENAME)
    return summary_df


def _collected(directory: Path | str, filename: str) -> Path:
    path = Path(directory) / filename
    if not path.exists():
        msg = f"{path} is absent; run collect_zone_lookup once every block is built"
        raise FileNotFoundError(msg)
    return path


def load_zone_lookup_index(directory: Path | str) -> pd.DataFrame:
    """``block_key``, ``zone``, ``pixel_count`` and the zone's window in that block, one row per pair.

    The blocks a zone has pixels in, and where: ``read_zone_lookup_block(directory, block_key,
    window=Window(col_off, row_off, width, height))`` reads just that zone's pixels.
    """
    path = _collected(directory, ZONE_LOOKUP_INDEX_FILENAME)
    out: pd.DataFrame = read_parquet_checked(path, _INDEX_SCHEMA).to_pandas()
    return out


def load_zone_lookup_summary(directory: Path | str) -> pd.DataFrame:
    """``zone``, ``pixel_count`` and ``block_count`` for every input zone, zeros included."""
    path = _collected(directory, ZONE_LOOKUP_SUMMARY_FILENAME)
    out: pd.DataFrame = read_parquet_checked(path, _SUMMARY_SCHEMA).to_pandas()
    return out


def load_zone_lookup_metadata(directory: Path | str) -> dict[str, Any]:
    """The ``build.yaml`` sidecar; refuses a product or schema version this code does not know."""
    path = _collected(directory, ZONE_LOOKUP_METADATA_FILENAME)
    metadata: dict[str, Any] = yaml.safe_load(path.read_text())
    if metadata.get("product") != PRODUCT or metadata.get("schema_version") != SCHEMA_VERSION:
        msg = (
            f"{path} describes product {metadata.get('product')!r} schema "
            f"{metadata.get('schema_version')!r}; this code reads {PRODUCT!r} schema {SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return metadata
