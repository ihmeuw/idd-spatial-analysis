"""Zone cell weights: per block and label, collected per label, applied to a coarse field."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from affine import Affine

from idd_spatial_analysis import __version__
from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid, bbox_in_crs, grid_from_bounds
from idd_spatial_analysis.lookup import (
    ZoneCellWeights,
    build_zone_cell_weights_block,
    collect_zone_cell_weights,
    combine_zone_cell_weights,
    compute_zone_cell_weights_block,
    load_zone_cell_weights,
    zone_cell_weights_block_path,
    zone_cell_weights_path,
)
from idd_spatial_analysis.raster import NearestIndexMap, rasterize_zones
from idd_spatial_analysis.zonal import ZoneCellPairs, collate_blocks, weighted_sums, zone_weights

from .conftest import EAST, WEST

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

PARAMS_KEY = b"idd_spatial_analysis.zone_cell_weights"
LABELS = ("2000", "2001")


@pytest.fixture
def coarse(blocks: dict[str, BlockGrid]) -> BlockGrid:
    """20 x 10 cells of 700 m over both blocks; one column of cells straddles their shared edge."""
    return BlockGrid(
        crs=blocks[WEST].crs,
        transform=Affine(700.0, 0, -150.0, 0, -700.0, 6550.0),
        height=10,
        width=20,
    )


@pytest.fixture
def masks(blocks: dict[str, BlockGrid], wide_zones_gdf: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    return {key: rasterize_zones(wide_zones_gdf, "zone", grid)[0] for key, grid in blocks.items()}


@pytest.fixture
def populations(blocks: dict[str, BlockGrid]) -> dict[str, dict[str, np.ndarray]]:
    """Per label and block: positive float32 weights with a zero band and a NaN patch."""
    rng = np.random.default_rng(13)
    out: dict[str, dict[str, np.ndarray]] = {}
    for label in LABELS:
        out[label] = {}
        for key, grid in blocks.items():
            raster = rng.gamma(2.0, 50.0, size=grid.shape).astype("float32")
            raster[:, 20:24] = 0.0
            raster[50:58, 5:13] = np.nan
            out[label][key] = raster
    return out


def _field(shape: tuple[int, ...]) -> np.ndarray:
    """A float32 field on the coarse grid with two missing cells in every slice."""
    field = np.random.default_rng(3).normal(20.0, 5.0, size=shape).astype("float32")
    field[..., 2, 3] = np.nan
    field[..., 7, 15] = np.nan
    return field


def _pixel_pass(  # noqa: PLR0913, PLR0917  (the inputs of the pass it reproduces)
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    field: np.ndarray,
    coarse: BlockGrid,
    policy: str,
) -> pd.DataFrame:
    """The sums the weight table must reproduce: the field on the pixels, then ``weighted_sums``."""
    partials = [
        weighted_sums(
            masks[key],
            weights[key],
            NearestIndexMap.build(coarse, grid).apply(field),
            nan_policy=policy,  # type: ignore[arg-type]
        )
        for key, grid in blocks.items()
    ]
    return collate_blocks(partials)


def _built(
    directory: Path,
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    """Round one for both blocks and both labels: one pairing per block, one file per label."""
    for key, grid in blocks.items():
        pairs = ZoneCellPairs.build(masks[key], grid, coarse)
        for label in LABELS:
            build_zone_cell_weights_block(pairs, populations[label][key], directory, label, key)


def test_block_table_columns_types_and_parameters(
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    weight = populations["2000"][WEST]
    pairs = ZoneCellPairs.build(masks[WEST], blocks[WEST], coarse)
    table = compute_zone_cell_weights_block(pairs, weight)
    assert table.schema.names == ["zone", "coarse_row", "coarse_col", "weight"]
    assert [str(field.type) for field in table.schema] == ["int32", "int32", "int32", "double"]
    frame = table.to_pandas()
    pd.testing.assert_frame_equal(frame, zone_weights(masks[WEST], weight, blocks[WEST], coarse))

    params = json.loads(table.schema.metadata[PARAMS_KEY])
    assert params["product"] == "zone_cell_weights"
    assert params["schema_version"] == 1
    assert params["package_version"] == __version__
    assert BlockGrid.from_dict(params["coarse"]) == coarse
    assert params["fine"] == {"crs": "ESRI:54034", "resolution": [100.0, 100.0]}
    block = params["block"]
    assert block["zoned_pixels"] == 64 * 64
    assert block["pair_count"] == len(frame)
    assert block["uncovered_pixels"] == 0
    assert block["uncovered_weight"] == 0.0
    assert block["centre_transform"] == "same_crs"
    assert block["weight_total"] == pytest.approx(np.nansum(weight.astype(np.float64)), rel=1e-12)


def test_collected_weights_reproduce_the_pixel_pass(
    tmp_path: Path,
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    _built(tmp_path, blocks, masks, populations, coarse)
    field = _field(coarse.shape)
    for label in LABELS:
        collected = collect_zone_cell_weights(
            tmp_path, label, [EAST, WEST], zone_set="synthetic", weight_source="unit test"
        )
        block_rows = sum(
            pq.read_metadata(zone_cell_weights_block_path(tmp_path, label, key)).num_rows
            for key in blocks
        )
        # Zones 1 and 3 share the straddling column of cells across the two blocks: those
        # pairs are added, so the collected table is shorter than the block tables together.
        assert len(collected) < block_rows
        assert not collected.duplicated(["zone", "coarse_row", "coarse_col"]).any()
        ordered = collected.sort_values(["zone", "coarse_row", "coarse_col"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(collected, ordered)

        weights = load_zone_cell_weights(tmp_path, label)
        assert weights.coarse == coarse
        pd.testing.assert_frame_equal(weights.table, collected)
        assert weights.table["zone"].dtype == np.int32
        assert weights.table["weight"].dtype == np.float64
        for policy in ("keep_weight", "drop_pixel"):
            got = weights.apply(field, coarse, nan_policy=policy)  # type: ignore[arg-type]
            expected = _pixel_pass(blocks, masks, populations[label], field, coarse, policy)
            assert got["zone"].tolist() == expected["zone"].tolist() == [1, 2, 3]
            np.testing.assert_allclose(got["weighted_sum"], expected["weighted_sum"], rtol=1e-12)
            np.testing.assert_allclose(got["weight_sum"], expected["weight_sum"], rtol=1e-12)

        total = sum(np.nansum(raster.astype(np.float64)) for raster in populations[label].values())
        recorded = weights.metadata["collected"]
        assert recorded["label"] == label
        assert recorded["zone_set"] == "synthetic"
        assert recorded["weight_source"] == "unit test"
        assert recorded["block_count"] == 2
        assert recorded["pair_count"] == len(collected)
        assert recorded["zone_count"] == 3
        assert recorded["zoned_pixels"] == 2 * 64 * 64
        assert recorded["uncovered_pixels"] == 0
        assert recorded["centre_transforms"] == ["same_crs"]
        assert recorded["weight_total"] == pytest.approx(total, rel=1e-12)
        assert collected["weight"].sum() == pytest.approx(total, rel=1e-12)

    assert zone_cell_weights_block_path(tmp_path, "2000", WEST) == (
        tmp_path / "block_weights" / "2000" / f"{WEST}.parquet"
    )
    assert zone_cell_weights_path(tmp_path, "2001") == tmp_path / "weights" / "2001.parquet"
    assert sorted(path.name for path in (tmp_path / "weights").iterdir()) == [
        "2000.parquet",
        "2001.parquet",
    ]
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_lonlat_coarse_grid_takes_the_row_and_column_route_in_memory(
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
) -> None:
    """No file at all: pair, sum, combine and apply, with the coarse grid in lon/lat."""
    left, bottom, _, _ = blocks[WEST].bounds
    _, _, right, top = blocks[EAST].bounds
    lonlat = bbox_in_crs((left, bottom, right, top), blocks[WEST].crs, WGS84)
    coarse = grid_from_bounds(lonlat, 0.01, WGS84)
    tables = []
    for key, grid in blocks.items():
        pairs = ZoneCellPairs.build(masks[key], grid, coarse)
        assert pairs.centre_transform == "row_and_column"
        tables.append(compute_zone_cell_weights_block(pairs, populations["2000"][key]))
    combined = combine_zone_cell_weights(tables, label="2000", zone_set="synthetic")
    weights = ZoneCellWeights.from_table(combined)
    assert weights.coarse == coarse
    assert weights.metadata["collected"]["centre_transforms"] == ["row_and_column"]
    assert weights.metadata["collected"]["weight_source"] is None
    field = np.random.default_rng(4).normal(size=coarse.shape).astype("float32")
    got = weights.apply(field, coarse)
    expected = _pixel_pass(blocks, masks, populations["2000"], field, coarse, "keep_weight")
    assert got["zone"].tolist() == expected["zone"].tolist()
    np.testing.assert_allclose(got["weighted_sum"], expected["weighted_sum"], rtol=1e-12)
    np.testing.assert_allclose(got["weight_sum"], expected["weight_sum"], rtol=1e-12)


def test_apply_refuses_a_field_on_another_grid(
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    pairs = ZoneCellPairs.build(masks[WEST], blocks[WEST], coarse)
    block_table = compute_zone_cell_weights_block(pairs, populations["2000"][WEST])
    weights = ZoneCellWeights.from_table(block_table)
    field = _field(coarse.shape)
    expected = weights.apply(field, coarse)

    half_a_cell = BlockGrid(
        crs=coarse.crs, transform=Affine(700.0, 0, 200.0, 0, -700.0, 6900.0), height=10, width=20
    )
    with pytest.raises(ValueError, match="these weights were built for"):
        weights.apply(field, half_a_cell)
    one_row_more = BlockGrid(crs=coarse.crs, transform=coarse.transform, height=11, width=20)
    with pytest.raises(ValueError, match="these weights were built for"):
        weights.apply(np.zeros(one_row_more.shape, dtype="float32"), one_row_more)
    other_crs = BlockGrid(crs=WGS84, transform=coarse.transform, height=10, width=20)
    with pytest.raises(ValueError, match="these weights were built for"):
        weights.apply(field, other_crs)
    with pytest.raises(ValueError, match="does not end with the grid shape"):
        weights.apply(field[:5], coarse)

    # The same grid described a second time, a rounding error away, is accepted.
    nearly = BlockGrid(
        crs=coarse.crs,
        transform=Affine(700.0, 0, -150.0 + 0.05, 0, -700.0, 6550.0 - 0.05),
        height=10,
        width=20,
    )
    pd.testing.assert_frame_equal(weights.apply(field, nearly), expected)

    stack = np.stack([field, 2 * field])
    stacked = weights.apply(stack, coarse)
    assert sorted(stacked["slice"].unique().tolist()) == [0, 1]
    second = stacked.loc[stacked["slice"] == 1].reset_index(drop=True)
    np.testing.assert_allclose(second["weighted_sum"], 2 * expected["weighted_sum"], rtol=1e-6)


def test_uncovered_pixels_are_refused_unless_allowed_and_then_counted(
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
) -> None:
    grid, weight = blocks[WEST], populations["2000"][WEST]
    west_half = BlockGrid(
        crs=grid.crs, transform=Affine(400.0, 0, 0, 0, -400.0, 6400.0), height=16, width=8
    )
    pairs = ZoneCellPairs.build(masks[WEST], grid, west_half)
    with pytest.raises(ValueError, match="fall outside the coarse grid"):
        compute_zone_cell_weights_block(pairs, weight)
    table = compute_zone_cell_weights_block(pairs, weight, allow_uncovered=True)
    block = json.loads(table.schema.metadata[PARAMS_KEY])["block"]
    lost = weight[:, 32:].astype(np.float64)
    assert block["uncovered_pixels"] == int(np.isfinite(lost).sum())
    assert block["uncovered_weight"] == pytest.approx(np.nansum(lost), rel=1e-12)
    assert block["zoned_pixels"] == 64 * 64
    kept = np.nansum(weight[:, :32].astype(np.float64))
    assert block["weight_total"] == pytest.approx(kept, rel=1e-12)
    combined = combine_zone_cell_weights([table], label="2000", zone_set="s")
    collected = json.loads(combined.schema.metadata[PARAMS_KEY])["collected"]
    assert collected["uncovered_pixels"] == block["uncovered_pixels"]
    assert collected["uncovered_weight"] == pytest.approx(block["uncovered_weight"])


def test_a_block_without_zones_gives_an_empty_table_that_collects(
    tmp_path: Path,
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    empty_pairs = ZoneCellPairs.build(
        np.zeros(blocks[EAST].shape, dtype="int32"), blocks[EAST], coarse
    )
    empty = build_zone_cell_weights_block(
        empty_pairs, populations["2000"][EAST], tmp_path, "2000", EAST
    )
    assert empty.empty
    assert empty["zone"].dtype == np.int32
    assert empty["weight"].dtype == np.float64
    west_pairs = ZoneCellPairs.build(masks[WEST], blocks[WEST], coarse)
    west = build_zone_cell_weights_block(
        west_pairs, populations["2000"][WEST], tmp_path, "2000", WEST
    )
    collected = collect_zone_cell_weights(tmp_path, "2000", [WEST, EAST], zone_set="s")
    pd.testing.assert_frame_equal(collected, west)


def test_collect_refuses_missing_foreign_and_mixed_blocks(
    tmp_path: Path,
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    weight = populations["2000"]
    west_pairs = ZoneCellPairs.build(masks[WEST], blocks[WEST], coarse)
    build_zone_cell_weights_block(west_pairs, weight[WEST], tmp_path, "2000", WEST)
    with pytest.raises(FileNotFoundError, match="1 blocks have no weights for '2000'"):
        collect_zone_cell_weights(tmp_path, "2000", [WEST, EAST], zone_set="s")
    with pytest.raises(FileNotFoundError, match="1 blocks have no weights for '2001'"):
        collect_zone_cell_weights(tmp_path, "2001", [WEST], zone_set="s")

    other = BlockGrid(
        crs=coarse.crs, transform=Affine(800.0, 0, -150.0, 0, -800.0, 6550.0), height=10, width=20
    )
    east_pairs = ZoneCellPairs.build(masks[EAST], blocks[EAST], other)
    build_zone_cell_weights_block(east_pairs, weight[EAST], tmp_path, "2000", EAST)
    with pytest.raises(ValueError, match="not in block_keys"):
        collect_zone_cell_weights(tmp_path, "2000", [WEST], zone_set="s")
    with pytest.raises(ValueError, match="was built with"):
        collect_zone_cell_weights(tmp_path, "2000", [WEST, EAST], zone_set="s")
    with pytest.raises(ValueError, match="block_keys is empty"):
        collect_zone_cell_weights(tmp_path, "2000", [], zone_set="s")
    with pytest.raises(ValueError, match="block_keys holds duplicates"):
        collect_zone_cell_weights(tmp_path, "2000", [WEST, WEST], zone_set="s")
    assert not (tmp_path / "weights").exists()


def test_combine_and_from_table_refuse_tables_that_are_not_this_product(
    blocks: dict[str, BlockGrid],
    masks: dict[str, np.ndarray],
    populations: dict[str, dict[str, np.ndarray]],
    coarse: BlockGrid,
) -> None:
    pairs = ZoneCellPairs.build(masks[WEST], blocks[WEST], coarse)
    table = compute_zone_cell_weights_block(pairs, populations["2000"][WEST])
    with pytest.raises(ValueError, match="no block tables"):
        combine_zone_cell_weights([], label="2000", zone_set="s")
    bare = table.replace_schema_metadata(None)
    with pytest.raises(ValueError, match="carries no zone cell weights parameters"):
        combine_zone_cell_weights([bare], label="2000", zone_set="s")
    with pytest.raises(ValueError, match="carries no zone cell weights parameters"):
        ZoneCellWeights.from_table(bare)
    foreign = table.replace_schema_metadata(
        {PARAMS_KEY: json.dumps({"product": "something_else", "schema_version": 1}).encode()}
    )
    with pytest.raises(ValueError, match="something_else"):
        combine_zone_cell_weights([foreign], label="2000", zone_set="s")
    narrowed = table.set_column(3, "weight", table["weight"].cast(pa.float32()))
    with pytest.raises(ValueError, match="is not the expected"):
        combine_zone_cell_weights([narrowed], label="2000", zone_set="s")


def test_loader_and_paths_need_a_collected_table_and_plain_names(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="collect_zone_cell_weights"):
        load_zone_cell_weights(tmp_path, "2000")
    for bad in ("", "..", "a/b"):
        with pytest.raises(ValueError, match="label must be a plain file stem"):
            zone_cell_weights_path(tmp_path, bad)
        with pytest.raises(ValueError, match="block_key must be a plain file stem"):
            zone_cell_weights_block_path(tmp_path, "2000", bad)
