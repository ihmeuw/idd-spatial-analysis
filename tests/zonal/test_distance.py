"""Nearest-facility distances against brute force, and shares within thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid
from idd_spatial_analysis.zonal import nearest_facility_distance, share_within

if TYPE_CHECKING:
    import geopandas as gpd
    from rasterio.windows import Window


def _brute_force(
    grid: BlockGrid,
    mask: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    facility_zone: np.ndarray | None,
) -> np.ndarray:
    xs, ys = grid.pixel_centers()
    out = np.full(grid.shape, np.nan)
    for row in range(grid.height):
        for col in range(grid.width):
            zone = mask[row, col]
            if zone == 0:
                continue
            chosen = slice(None) if facility_zone is None else facility_zone == zone
            distances = np.hypot(fx[chosen] - xs[col], fy[chosen] - ys[row])
            if distances.size:
                out[row, col] = distances.min()
    return out


def test_any_facility_matches_brute_force(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    got = nearest_facility_distance(mask, facilities_gdf, grid, within_zone=False)
    assert got.dtype == np.float32
    expect = _brute_force(
        grid, mask, facilities_gdf.geometry.x.to_numpy(), facilities_gdf.geometry.y.to_numpy(), None
    )
    np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_within_zone_uses_only_that_zones_facilities(
    grid: BlockGrid,
    mask_windows: tuple[np.ndarray, dict[int, Window]],
    facilities_gdf: gpd.GeoDataFrame,
) -> None:
    mask, windows = mask_windows
    got = nearest_facility_distance(mask, facilities_gdf, grid, within_zone=True, windows=windows)
    expect = _brute_force(
        grid,
        mask,
        facilities_gdf.geometry.x.to_numpy(),
        facilities_gdf.geometry.y.to_numpy(),
        facilities_gdf["zone"].to_numpy(),
    )
    np.testing.assert_allclose(got, expect, rtol=1e-6)
    assert np.isnan(got[mask == 3]).all()
    assert np.isfinite(got[mask == 1]).all()
    without_windows = nearest_facility_distance(mask, facilities_gdf, grid, within_zone=True)
    np.testing.assert_array_equal(got, without_windows)


def test_outside_zones_is_nan_and_reprojected_facilities_agree(
    grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame
) -> None:
    holed = mask.copy()
    holed[:, :5] = 0
    got = nearest_facility_distance(holed, facilities_gdf, grid, within_zone=False)
    assert np.isnan(got[:, :5]).all()
    assert np.isfinite(got[:, 5:]).all()
    lonlat = nearest_facility_distance(holed, facilities_gdf.to_crs(WGS84), grid, within_zone=False)
    np.testing.assert_allclose(lonlat, got, rtol=1e-6)


def test_bad_inputs(grid: BlockGrid, mask: np.ndarray, facilities_gdf: gpd.GeoDataFrame) -> None:
    lonlat_grid = BlockGrid(
        crs=WGS84, transform=Affine(0.001, 0, 0, 0, -0.001, 1), height=64, width=64
    )
    with pytest.raises(ValueError, match="projected"):
        nearest_facility_distance(mask, facilities_gdf, lonlat_grid, within_zone=False)
    with pytest.raises(ValueError, match="CRS"):
        nearest_facility_distance(
            mask, facilities_gdf.set_crs(None, allow_override=True), grid, within_zone=False
        )
    with pytest.raises(ValueError, match="no facilities"):
        nearest_facility_distance(mask, facilities_gdf.iloc[:0], grid, within_zone=False)
    polygons = facilities_gdf.copy()
    polygons["geometry"] = polygons.buffer(10)
    with pytest.raises(TypeError, match="Point"):
        nearest_facility_distance(mask, polygons, grid, within_zone=False)
    with pytest.raises(ValueError, match="does not match"):
        nearest_facility_distance(mask[:10], facilities_gdf, grid, within_zone=False)


def test_share_within_matches_manual_sums(
    grid: BlockGrid,
    mask_windows: tuple[np.ndarray, dict[int, Window]],
    facilities_gdf: gpd.GeoDataFrame,
    population: np.ndarray,
) -> None:
    mask, windows = mask_windows
    distance = nearest_facility_distance(
        mask, facilities_gdf, grid, within_zone=True, windows=windows
    )
    table = share_within(distance, population, mask, [3000.0, 1000.0])
    assert list(table.columns) == ["zone", "threshold", "weight_within", "weight_total", "share"]
    assert table["threshold"].tolist() == [1000.0] * 4 + [3000.0] * 4
    weight = population.astype(np.float64)
    for zone in (1, 2, 3, 4):
        in_zone = (mask == zone) & np.isfinite(weight)
        total = weight[in_zone].sum()
        for threshold in (1000.0, 3000.0):
            row = table.loc[(table["zone"] == zone) & (table["threshold"] == threshold)].iloc[0]
            within = weight[in_zone & (distance <= threshold)].sum()
            assert row["weight_total"] == pytest.approx(total)
            assert row["weight_within"] == pytest.approx(within)
            assert row["share"] == pytest.approx(within / total)
    assert (table.loc[table["zone"] == 3, "weight_within"] == 0).all()
    assert (table.loc[table["zone"] == 3, "share"] == 0).all()
    assert (table.loc[table["zone"] == 1, "share"] > 0).all()


def test_share_within_bad_inputs_and_zero_weight(mask: np.ndarray, population: np.ndarray) -> None:
    with pytest.raises(ValueError, match="empty"):
        share_within(population, population, mask, [])
    with pytest.raises(ValueError, match="share a shape"):
        share_within(population[:10], population, mask, [1.0])
    table = share_within(population, np.zeros_like(population), mask, [1.0])
    assert table["share"].isna().all()
