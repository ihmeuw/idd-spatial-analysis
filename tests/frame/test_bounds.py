"""Bounds across CRSs."""

from __future__ import annotations

import pytest

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import AntimeridianError, bbox_in_crs, pad_bounds


def test_pad_bounds() -> None:
    assert pad_bounds((0.0, 0.0, 10.0, 10.0), 2.5) == (-2.5, -2.5, 12.5, 12.5)


def test_same_crs_is_identity() -> None:
    bounds = (1.0, 2.0, 3.0, 4.0)
    assert bbox_in_crs(bounds, WGS84, "EPSG:4326") == bounds


def test_equal_area_box_to_lonlat_covers_the_source() -> None:
    bounds = (6_000_000.0, 3_000_000.0, 6_500_000.0, 3_500_000.0)
    left, bottom, right, top = bbox_in_crs(bounds, WORLD_CYLINDRICAL_EQUAL_AREA, WGS84)
    assert 50.0 < left < right < 60.0
    assert 25.0 < bottom < top < 35.0


def test_antimeridian_wrap_is_refused() -> None:
    # A box straddling x = +/-20037508 m (the +/-180 meridian) in the equal-area frame.
    bounds = (19_900_000.0, 0.0, 20_037_508.34, 100_000.0)
    with pytest.raises(AntimeridianError, match="antimeridian"):
        bbox_in_crs(
            (-20_000_000.0, 0.0, 20_000_000.0, 100_000.0), WORLD_CYLINDRICAL_EQUAL_AREA, WGS84
        )
    left, _, right, _ = bbox_in_crs(bounds, WORLD_CYLINDRICAL_EQUAL_AREA, WGS84)
    assert right - left < 2.0


def test_degenerate_bounds_raise() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        bbox_in_crs((5.0, 0.0, 5.0, 1.0), WGS84, WORLD_CYLINDRICAL_EQUAL_AREA)


def test_lonlat_bounds_of_a_grid(grid) -> None:
    left, bottom, right, top = grid.lonlat_bounds()
    assert left == pytest.approx(0.0, abs=1e-9)
    assert bottom == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < right < 0.1
    assert 0.0 < top < 0.1
    padded = grid.lonlat_bounds(pad=1000.0)
    assert padded[0] < left
    assert padded[2] > right
