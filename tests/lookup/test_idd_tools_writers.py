"""Both lookup products written through idd-tools' atomic writers, as a jobmon worker writes them.

The library's compute steps return what a worker must put on disk; these tests do that with
idd-tools' real ``atomic_write``, ``atomic_write_parquet`` and ``AtomicRegistry`` and check
that the collect steps accept the result. They need the ``jobmon`` extra, which resolves only
on the cluster, so they are skipped wherever ``idd_tools.jobmon`` cannot be imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from idd_spatial_analysis.frame import BlockGrid
from idd_spatial_analysis.lookup import (
    build_zone_cell_weights_block,
    build_zone_lookup_block,
    collect_zone_cell_weights,
    collect_zone_lookup,
    compute_zone_cell_weights_block,
    compute_zone_lookup_block,
    load_zone_cell_weights,
    read_zone_lookup_block,
    zone_cell_weights_block_path,
    zone_lookup_block_path,
    zone_lookup_block_table_path,
)
from idd_spatial_analysis.zonal import ZoneCellPairs

from .conftest import EAST, WEST, WIDE_ZONE_IDS

if TYPE_CHECKING:
    import geopandas as gpd

atomic_io = pytest.importorskip(
    "idd_tools.jobmon.atomic_io",
    reason="needs the jobmon extra, which resolves on the cluster only",
)


def _registered(directory: Path, task_id: str) -> list[str]:
    """The paths the task's registry sidecar lists, relative to ``directory``."""
    sidecar = json.loads((directory / ".atomic" / f"{task_id}.json").read_text())
    return sorted(str(Path(entry["path"]).relative_to(directory)) for entry in sidecar["files"])


def test_zone_lookup_blocks_written_through_idd_tools_collect(
    tmp_path: Path, blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame
) -> None:
    through, local = tmp_path / "through_idd_tools", tmp_path / "local"
    with atomic_io.AtomicRegistry(
        through, task_id="zone_lookup-0", task_template="zone_lookup"
    ) as registry:
        for key, grid in blocks.items():
            block = compute_zone_lookup_block(wide_zones_gdf, "zone", grid)
            raster_path = zone_lookup_block_path(through, key)
            table_path = zone_lookup_block_table_path(through, key)
            assert not atomic_io.subtask_skip(table_path)
            atomic_io.atomic_write(raster_path, block.write_raster)
            registry.register(raster_path)
            atomic_io.atomic_write_parquet(
                block.zone_table(), table_path, registry=registry, schema=block.table.schema
            )
            assert atomic_io.subtask_skip(raster_path)
            assert atomic_io.subtask_skip(table_path)
            build_zone_lookup_block(wide_zones_gdf, "zone", grid, local, key)
    assert _registered(through, "zone_lookup-0") == sorted(
        [f"blocks/{key}.tif" for key in blocks] + [f"block_zones/{key}.parquet" for key in blocks]
    )

    summary = collect_zone_lookup(through, [WEST, EAST], WIDE_ZONE_IDS, zone_set="synthetic")
    expected = collect_zone_lookup(local, [WEST, EAST], WIDE_ZONE_IDS, zone_set="synthetic")
    pd.testing.assert_frame_equal(summary, expected)
    for key in blocks:
        got, got_grid = read_zone_lookup_block(through, key)
        want, want_grid = read_zone_lookup_block(local, key)
        np.testing.assert_array_equal(got, want)
        assert got_grid == want_grid
    assert not list(through.rglob("*.tmp"))


def test_zone_cell_weights_written_through_idd_tools_collect(
    tmp_path: Path, blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame
) -> None:
    through, local = tmp_path / "through_idd_tools", tmp_path / "local"
    coarse = BlockGrid(
        crs=blocks[WEST].crs,
        transform=Affine(700.0, 0, -150.0, 0, -700.0, 6550.0),
        height=10,
        width=20,
    )
    rng = np.random.default_rng(17)
    with atomic_io.AtomicRegistry(
        through, task_id="zone_cell_weights-0", task_template="zone_cell_weights"
    ) as registry:
        for key, grid in blocks.items():
            mask = compute_zone_lookup_block(wide_zones_gdf, "zone", grid).mask
            pairs = ZoneCellPairs.build(mask, grid, coarse)
            weight = rng.gamma(2.0, 50.0, size=grid.shape).astype("float32")
            table = compute_zone_cell_weights_block(pairs, weight)
            path = zone_cell_weights_block_path(through, "2000", key)
            atomic_io.atomic_write_parquet(
                table.to_pandas(), path, registry=registry, schema=table.schema
            )
            assert atomic_io.subtask_skip(path)
            build_zone_cell_weights_block(pairs, weight, local, "2000", key)
    assert _registered(through, "zone_cell_weights-0") == sorted(
        f"block_weights/2000/{key}.parquet" for key in blocks
    )

    collected = collect_zone_cell_weights(through, "2000", [WEST, EAST], zone_set="synthetic")
    expected = collect_zone_cell_weights(local, "2000", [WEST, EAST], zone_set="synthetic")
    pd.testing.assert_frame_equal(collected, expected)
    assert load_zone_cell_weights(through, "2000").coarse == coarse
    assert not list(through.rglob("*.tmp"))
