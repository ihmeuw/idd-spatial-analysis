"""Raking shapes and hierarchies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from idd_spatial_analysis.boundaries import load_hierarchy, load_raking_shapes

if TYPE_CHECKING:
    from pathlib import Path


def test_gbd_shapes_keep_most_detailed_of_latest_year(raking_root: Path) -> None:
    shapes = load_raking_shapes(raking_root, "gbd_2023")
    assert sorted(shapes["location_id"]) == [101, 102]
    assert str(shapes["location_id"].dtype) == "int32"


def test_bounds_filter_applies_at_read(raking_root: Path) -> None:
    shapes = load_raking_shapes(raking_root, "gbd_2023", bounds=(6.0, 1.0, 7.0, 2.0))
    assert sorted(shapes["location_id"]) == [102]


def test_lsae_shapes_are_read_as_is(raking_root: Path) -> None:
    shapes = load_raking_shapes(raking_root, "lsae_1285")
    assert sorted(shapes["location_id"]) == [101, 102]


def test_hierarchy_columns_and_dtypes(raking_root: Path) -> None:
    table = load_hierarchy(raking_root, "gbd_2023")
    assert list(table.columns) == ["location_id", "parent_id", "level"]
    assert all(str(dtype) == "int32" for dtype in table.dtypes)


@pytest.mark.parametrize("name", ["gbd2023", "lsae", "something"])
def test_unknown_hierarchy_names_raise(raking_root: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unknown hierarchy"):
        load_raking_shapes(raking_root, name)


def test_fhs_has_no_shapes(raking_root: Path) -> None:
    with pytest.raises(ValueError, match="no shapes"):
        load_raking_shapes(raking_root, "fhs_2021")
