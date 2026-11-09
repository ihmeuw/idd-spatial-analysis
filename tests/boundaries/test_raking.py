"""Raking shapes and hierarchies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from idd_spatial_analysis.boundaries import load_hierarchy, load_raking_shapes
from idd_spatial_analysis.boundaries.raking import _has_covering_bbox

if TYPE_CHECKING:
    from pathlib import Path


def test_gbd_shapes_keep_most_detailed_locations(raking_root: Path) -> None:
    shapes = load_raking_shapes(raking_root, "gbd_2023")
    assert sorted(shapes["location_id"]) == [101, 102]
    assert str(shapes["location_id"].dtype) == "int32"


def test_bounds_filter_through_the_covering_column_and_the_exact_test(raking_root: Path) -> None:
    assert _has_covering_bbox(raking_root / "shapes_gbd_2023.parquet")
    assert not _has_covering_bbox(raking_root / "gbd-inputs" / "shapes_lsae_1285_a2.parquet")
    box_in_102 = (6.0, 1.0, 7.0, 2.0)
    assert sorted(
        load_raking_shapes(raking_root, "gbd_2023", bounds=box_in_102)["location_id"]
    ) == [102]
    assert sorted(
        load_raking_shapes(raking_root, "lsae_1285", bounds=box_in_102)["location_id"]
    ) == [102]
    assert load_raking_shapes(raking_root, "lsae_1285", bounds=(50.0, 50.0, 60.0, 60.0)).empty


def test_year_id_pins_the_flags_when_they_vary(raking_root: Path) -> None:
    population = pd.DataFrame(
        {
            "location_id": [100, 101, 102, 100, 101, 102],
            "year_id": [2020, 2020, 2020, 2023, 2023, 2023],
            "most_detailed": [0, 1, 1, 1, 0, 0],
        }
    )
    population.to_parquet(raking_root / "population_gbd_2023.parquet")
    with pytest.raises(ValueError, match="year_id"):
        load_raking_shapes(raking_root, "gbd_2023")
    assert sorted(load_raking_shapes(raking_root, "gbd_2023", year_id=2020)["location_id"]) == [
        101,
        102,
    ]
    assert sorted(load_raking_shapes(raking_root, "gbd_2023", year_id=2023)["location_id"]) == [100]
    with pytest.raises(ValueError, match="no rows"):
        load_raking_shapes(raking_root, "gbd_2023", year_id=1999)


def test_no_most_detailed_shape_is_an_error(raking_root: Path) -> None:
    population = pd.DataFrame({"location_id": [999], "year_id": [2023], "most_detailed": [1]})
    population.to_parquet(raking_root / "population_gbd_2023.parquet")
    with pytest.raises(ValueError, match="most-detailed"):
        load_raking_shapes(raking_root, "gbd_2023")
    population = pd.DataFrame({"location_id": [100], "year_id": [2023], "most_detailed": [0]})
    population.to_parquet(raking_root / "population_gbd_2023.parquet")
    with pytest.raises(ValueError, match="flags no location"):
        load_raking_shapes(raking_root, "gbd_2023")


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
