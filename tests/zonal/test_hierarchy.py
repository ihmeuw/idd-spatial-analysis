"""Rolling most-detailed sums up a location hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_spatial_analysis.zonal import rollup_hierarchy


@pytest.fixture
def hierarchy() -> pd.DataFrame:
    """Root 1; 2, 3 and 7 under it; 4, 5 under 2; 6 under 3. Leaves: 4, 5, 6 and the level-1 leaf 7."""
    return pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4, 5, 6, 7],
            "parent_id": [0, 1, 1, 2, 2, 3, 1],
            "level": [0, 1, 1, 2, 2, 2, 1],
        }
    )


@pytest.fixture
def leaf_sums() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone": [4, 5, 6, 7],
            "weighted_sum": [10.0, 20.0, 30.0, 40.0],
            "weight_sum": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_sums_add_up_and_means_divide_last(
    leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame
) -> None:
    out = rollup_hierarchy(leaf_sums, hierarchy)
    assert list(out.columns) == ["location_id", "level", "weighted_sum", "weight_sum", "mean"]
    assert out["location_id"].dtype == np.int32
    assert out["level"].dtype == np.int32
    assert out["level"].tolist() == [0, 1, 1, 1, 2, 2, 2]
    by = out.set_index("location_id")
    assert by.loc[2, "weighted_sum"] == 30.0
    assert by.loc[2, "weight_sum"] == 3.0
    assert by.loc[2, "mean"] == 10.0
    assert by.loc[3, "weighted_sum"] == 30.0
    assert by.loc[1, "weighted_sum"] == 100.0
    assert by.loc[1, "weight_sum"] == 10.0
    assert by.loc[1, "mean"] == 10.0
    assert by.loc[7, "level"] == 1
    assert by.loc[7, "mean"] == 10.0


def test_zero_weight_gives_nan_mean(hierarchy: pd.DataFrame) -> None:
    sums = pd.DataFrame({"zone": [4, 5, 6, 7], "weighted_sum": [0.0] * 4, "weight_sum": [0.0] * 4})
    out = rollup_hierarchy(sums, hierarchy)
    assert out["mean"].isna().all()


def test_slices_are_kept_apart(leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    doubled = leaf_sums.assign(slice=1, weighted_sum=leaf_sums["weighted_sum"] * 2)
    stacked = pd.concat([leaf_sums.assign(slice=0), doubled], ignore_index=True)
    out = rollup_hierarchy(stacked, hierarchy)
    assert list(out.columns) == [
        "location_id",
        "slice",
        "level",
        "weighted_sum",
        "weight_sum",
        "mean",
    ]
    root = out.loc[out["location_id"] == 1].set_index("slice")
    assert root.loc[0, "weighted_sum"] == 100.0
    assert root.loc[1, "weighted_sum"] == 200.0


def test_incomplete_leaves_raise_unless_allowed(
    leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame
) -> None:
    partial = leaf_sums.iloc[:3]
    with pytest.raises(ValueError, match="no sums"):
        rollup_hierarchy(partial, hierarchy)
    out = rollup_hierarchy(partial, hierarchy, require_complete=False)
    assert out.set_index("location_id").loc[1, "weighted_sum"] == 60.0


def test_parents_and_unknown_zones_are_refused(
    leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame
) -> None:
    with_parent = pd.concat(
        [leaf_sums, pd.DataFrame({"zone": [2], "weighted_sum": [1.0], "weight_sum": [1.0]})]
    )
    with pytest.raises(ValueError, match="parents"):
        rollup_hierarchy(with_parent, hierarchy)
    with pytest.raises(ValueError, match="not in the hierarchy"):
        rollup_hierarchy(leaf_sums.assign(zone=[4, 5, 6, 99]), hierarchy)
    with pytest.raises(KeyError, match="sums"):
        rollup_hierarchy(leaf_sums.drop(columns=["weight_sum"]), hierarchy)
    with pytest.raises(KeyError, match="hierarchy"):
        rollup_hierarchy(leaf_sums, hierarchy.drop(columns=["level"]))


def test_duplicate_rows_are_refused(leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    doubled = pd.concat([leaf_sums, leaf_sums.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        rollup_hierarchy(doubled, hierarchy)
    sliced = leaf_sums.assign(slice=0)
    stacked = pd.concat([sliced, sliced.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        rollup_hierarchy(stacked, hierarchy)


def test_completeness_is_checked_per_slice(
    leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame
) -> None:
    stacked = pd.concat(
        [leaf_sums.assign(slice=0), leaf_sums.iloc[:3].assign(slice=1)], ignore_index=True
    )
    with pytest.raises(ValueError, match="slice 1"):
        rollup_hierarchy(stacked, hierarchy)
    out = rollup_hierarchy(stacked, hierarchy, require_complete=False)
    root = out.loc[out["location_id"] == 1].set_index("slice")
    assert root.loc[0, "weighted_sum"] == 100.0
    assert root.loc[1, "weighted_sum"] == 60.0


def test_roots_may_sit_above_level_zero(leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    shifted = hierarchy.assign(level=hierarchy["level"] + 1)
    out = rollup_hierarchy(leaf_sums, shifted)
    assert out["level"].tolist() == [1, 2, 2, 2, 3, 3, 3]
    assert out.set_index("location_id").loc[1, "weighted_sum"] == 100.0


def test_inconsistent_hierarchies_raise(leaf_sums: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    skipped = hierarchy.copy()
    skipped.loc[skipped["location_id"] == 4, "level"] = 3
    with pytest.raises(ValueError, match="not at level"):
        rollup_hierarchy(leaf_sums, skipped)
    duplicated = pd.concat(
        [hierarchy, pd.DataFrame({"location_id": [4], "parent_id": [3], "level": [2]})]
    )
    with pytest.raises(ValueError, match="more than one"):
        rollup_hierarchy(leaf_sums, duplicated)
    orphan = hierarchy.copy()
    orphan.loc[orphan["location_id"] == 6, "parent_id"] = 42
    with pytest.raises(ValueError, match="does not list"):
        rollup_hierarchy(leaf_sums, orphan)
