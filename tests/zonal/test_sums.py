"""Weighted sums per zone against a numpy ground truth, and block collation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from idd_spatial_analysis.zonal import collate_blocks, weighted_sums

if TYPE_CHECKING:
    from idd_spatial_analysis.frame import BlockGrid

COLUMNS = ["zone", "weighted_sum", "weight_sum"]


def _truth(
    mask: np.ndarray, weight: np.ndarray, value: np.ndarray, policy: str
) -> list[tuple[int, float, float]]:
    rows = []
    for zone in sorted(set(np.unique(mask).tolist()) - {0}):
        w = weight[mask == zone].astype(np.float64)
        v = value[mask == zone].astype(np.float64)
        ok_w = np.isfinite(w)
        ok_v = ok_w & np.isfinite(v)
        weight_sum = w[ok_w].sum() if policy == "keep_weight" else w[ok_v].sum()
        rows.append((zone, float((w[ok_v] * v[ok_v]).sum()), float(weight_sum)))
    return rows


@pytest.mark.parametrize("policy", ["keep_weight", "drop_pixel"])
def test_matches_numpy_ground_truth(
    mask: np.ndarray, population: np.ndarray, value: np.ndarray, policy: str
) -> None:
    got = weighted_sums(mask, population, value, nan_policy=policy)  # type: ignore[arg-type]
    assert list(got.columns) == COLUMNS
    assert got["zone"].dtype == np.int32
    assert got["zone"].tolist() == [1, 2, 3, 4]
    for zone, weighted, weight in _truth(mask, population, value, policy):
        row = got.loc[got["zone"] == zone].iloc[0]
        assert row["weighted_sum"] == pytest.approx(weighted, rel=1e-9)
        assert row["weight_sum"] == pytest.approx(weight, rel=1e-9)


def test_policies_differ_only_where_value_is_nan(
    mask: np.ndarray, population: np.ndarray, value: np.ndarray
) -> None:
    keep = weighted_sums(mask, population, value, nan_policy="keep_weight")
    drop = weighted_sums(mask, population, value, nan_policy="drop_pixel")
    pd.testing.assert_series_equal(keep["weighted_sum"], drop["weighted_sum"])
    assert set(np.unique(mask[np.isnan(value)]).tolist()) == {2}
    assert (
        keep.loc[keep["zone"] == 2, "weight_sum"].item()
        > drop.loc[drop["zone"] == 2, "weight_sum"].item()
    )
    others = keep["zone"] != 2
    np.testing.assert_allclose(keep.loc[others, "weight_sum"], drop.loc[others, "weight_sum"])


def test_stack_adds_a_slice_column(
    mask: np.ndarray, population: np.ndarray, value: np.ndarray
) -> None:
    got = weighted_sums(mask, population, np.stack([value, 2 * value]), nan_policy="drop_pixel")
    assert list(got.columns) == ["zone", "slice", "weighted_sum", "weight_sum"]
    assert got["slice"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    single = weighted_sums(mask, population, value, nan_policy="drop_pixel")
    np.testing.assert_allclose(
        got.loc[got["slice"] == 1, "weighted_sum"].to_numpy(), 2 * single["weighted_sum"].to_numpy()
    )


def test_nan_weight_and_zone_zero_never_count(
    grid: BlockGrid, population: np.ndarray, value: np.ndarray
) -> None:
    mask = np.zeros(grid.shape, dtype="int32")
    mask[:, :10] = 5
    got = weighted_sums(mask, population, value, nan_policy="keep_weight")
    assert got["zone"].tolist() == [5]
    assert got["weight_sum"].item() == pytest.approx(
        np.nansum(population[:, :10].astype(np.float64)), rel=1e-9
    )


def test_empty_mask_gives_an_empty_typed_table(
    grid: BlockGrid, population: np.ndarray, value: np.ndarray
) -> None:
    got = weighted_sums(
        np.zeros(grid.shape, dtype="int32"), population, value, nan_policy="drop_pixel"
    )
    assert got.empty
    assert list(got.columns) == COLUMNS
    assert got["zone"].dtype == np.int32
    assert got["weight_sum"].dtype == np.float64


def test_bad_inputs(mask: np.ndarray, population: np.ndarray, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nan_policy"):
        weighted_sums(mask, population, value, nan_policy="whatever")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight shape"):
        weighted_sums(mask, population[:10], value, nan_policy="drop_pixel")
    with pytest.raises(ValueError, match="value shape"):
        weighted_sums(mask, population, value[:10], nan_policy="drop_pixel")
    with pytest.raises(TypeError, match="integer"):
        weighted_sums(mask.astype(float), population, value, nan_policy="drop_pixel")
    with pytest.raises(ValueError, match="2-D"):
        weighted_sums(mask[0], population[0], value[0], nan_policy="drop_pixel")


def test_large_zone_ids_cost_nothing_extra(
    grid: BlockGrid, population: np.ndarray, value: np.ndarray
) -> None:
    mask = np.full(grid.shape, 2_147_483_647, dtype="int32")
    mask[:, :32] = 7
    got = weighted_sums(mask, population, value, nan_policy="drop_pixel")
    assert got["zone"].tolist() == [7, 2_147_483_647]
    assert got["weight_sum"].iloc[0] == pytest.approx(
        np.nansum(population[:, :32].astype(np.float64)), rel=1e-9
    )


def test_collate_refuses_missing_keys(
    mask: np.ndarray, population: np.ndarray, value: np.ndarray
) -> None:
    table = weighted_sums(mask, population, np.stack([value, value]), nan_policy="drop_pixel")
    holed = table.astype({"slice": "float64"})
    holed.loc[0, "slice"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        collate_blocks([holed])


def test_collate_blocks_adds_partial_sums(
    mask: np.ndarray, population: np.ndarray, value: np.ndarray
) -> None:
    whole = weighted_sums(mask, population, value, nan_policy="drop_pixel")
    left = weighted_sums(mask[:, :40], population[:, :40], value[:, :40], nan_policy="drop_pixel")
    right = weighted_sums(mask[:, 40:], population[:, 40:], value[:, 40:], nan_policy="drop_pixel")
    merged = collate_blocks([left, right])
    pd.testing.assert_frame_equal(merged, whole, rtol=1e-9)
    stacked = weighted_sums(mask, population, np.stack([value, value]), nan_policy="drop_pixel")
    again = collate_blocks([stacked.iloc[:4], stacked.iloc[4:]])
    expect = stacked.sort_values(["zone", "slice"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(again, expect)
    with pytest.raises(ValueError, match="at least one"):
        collate_blocks([])
    with pytest.raises(KeyError, match="block table"):
        collate_blocks([pd.DataFrame({"zone": [1]})])
