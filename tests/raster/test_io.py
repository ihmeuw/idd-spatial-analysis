"""Windows, mosaics and atomic GeoTIFF writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.windows import Window

from idd_spatial_analysis.constants import WGS84
from idd_spatial_analysis.frame import BlockGrid, bbox_in_crs, grid_from_raster
from idd_spatial_analysis.raster import (
    atomic_write,
    mosaic,
    read_window,
    write_geotiff,
    write_geotiff_atomic,
)
from idd_spatial_analysis.raster import io as raster_io

if TYPE_CHECKING:
    from pathlib import Path


def test_read_window_subset_matches_slicing(
    population_tif: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    data, sub = read_window(population_tif, (1000.0, 1000.0, 2000.0, 2000.0))
    np.testing.assert_array_equal(data, population[44:54, 10:20])
    assert data.dtype == np.float32
    assert sub.bounds == (1000.0, 1000.0, 2000.0, 2000.0)
    assert sub.crs == grid.crs


def test_read_window_maps_nodata_to_nan_or_keeps_native(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    ints = np.where(np.isnan(population), 0, population).astype("int32")
    path = tmp_path / "ints.tif"
    with rasterio.open(path, "w", **grid.profile(dtype="int32", nodata=0)) as dst:
        dst.write(ints, 1)
    as_float, _ = read_window(path, grid.bounds)
    assert as_float.dtype == np.float32
    assert np.isnan(as_float[:, 20:24]).all()
    assert np.isnan(as_float[:8, :8]).all()
    native, _ = read_window(path, grid.bounds, as_float=False)
    assert native.dtype == np.int32
    np.testing.assert_array_equal(native, ints)


def test_read_window_bounds_in_another_crs(population_tif: Path, grid: BlockGrid) -> None:
    lonlat = bbox_in_crs((1000.0, 1000.0, 2000.0, 2000.0), grid.crs, WGS84)
    data, sub = read_window(population_tif, lonlat, bounds_crs=WGS84)
    assert 10 <= data.shape[0] <= 12
    assert 10 <= data.shape[1] <= 12
    assert sub.bounds[0] <= 1000.0
    assert sub.bounds[2] >= 2000.0


def test_read_window_boundless_fills_outside(population_tif: Path, population: np.ndarray) -> None:
    data, sub = read_window(population_tif, (-500.0, -500.0, 500.0, 500.0), boundless=True)
    assert data.shape == (10, 10)
    assert sub.bounds == (-500.0, -500.0, 500.0, 500.0)
    assert np.isnan(data[5:, :]).all()
    assert np.isnan(data[:, :5]).all()
    np.testing.assert_array_equal(data[:5, 5:], population[59:64, 0:5])


def test_read_window_boundless_integer_without_nodata_raises(
    tmp_path: Path, grid: BlockGrid
) -> None:
    path = tmp_path / "ints.tif"
    with rasterio.open(path, "w", **grid.profile(dtype="int16", nodata=None)) as dst:
        dst.write(np.ones(grid.shape, dtype="int16"), 1)
    with pytest.raises(ValueError, match="nodata"):
        read_window(path, (-500.0, -500.0, 500.0, 500.0), boundless=True)


def test_read_window_disjoint_raises(population_tif: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        read_window(population_tif, (10_000.0, 10_000.0, 11_000.0, 11_000.0))


def _write_columns(
    path: Path, grid: BlockGrid, population: np.ndarray, cols: slice, *, nodata: float | None
) -> None:
    sub = grid.sub(Window(cols.start, 0, cols.stop - cols.start, grid.height))
    with rasterio.open(path, "w", **sub.profile(nodata=nodata)) as dst:
        dst.write(population[:, cols], 1)


def test_mosaic_of_two_halves_restores_the_block(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    left, right = tmp_path / "left.tif", tmp_path / "right.tif"
    _write_columns(left, grid, population, slice(0, 32), nodata=np.nan)
    _write_columns(right, grid, population, slice(32, 64), nodata=np.nan)
    data, merged = mosaic([left, right], grid.bounds)
    assert merged.shape == grid.shape
    assert merged.transform == grid.transform
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, population)
    part, sub = mosaic([left, right], (3000.0, 3000.0, 3500.0, 3500.0))
    np.testing.assert_array_equal(part, population[29:34, 30:35])
    assert sub.bounds == (3000.0, 3000.0, 3500.0, 3500.0)


def test_mosaic_pads_with_nan_when_no_nodata_is_declared(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    left, right = tmp_path / "left.tif", tmp_path / "right.tif"
    _write_columns(left, grid, population, slice(0, 32), nodata=None)
    _write_columns(right, grid, population, slice(32, 64), nodata=None)
    data, merged = mosaic([left, right], (-200.0, -200.0, 6600.0, 6600.0))
    assert merged.shape == (68, 68)
    assert merged.bounds == (-200.0, -200.0, 6600.0, 6600.0)
    assert np.isnan(data[:2, :]).all()
    assert np.isnan(data[:, -2:]).all()
    np.testing.assert_array_equal(data[2:66, 2:66], population)
    assert (data[2:66, 22:26] == 0).all()  # genuine zeros survive the first-wins merge


def test_mosaic_of_integer_rasters_needs_nodata(tmp_path: Path, grid: BlockGrid) -> None:
    path = tmp_path / "ints.tif"
    with rasterio.open(path, "w", **grid.profile(dtype="int16", nodata=None)) as dst:
        dst.write(np.ones(grid.shape, dtype="int16"), 1)
    with pytest.raises(ValueError, match="nodata"):
        mosaic([path], grid.bounds)


def test_mosaic_snaps_off_lattice_bounds_to_the_first_raster(
    population_tif: Path, population: np.ndarray
) -> None:
    data, snapped = mosaic([population_tif], (40.0, 40.0, 1040.0, 1040.0))
    assert snapped.bounds == (0.0, 0.0, 1100.0, 1100.0)
    np.testing.assert_array_equal(data, population[53:64, 0:11])


def test_mosaic_with_an_explicit_resolution_resamples(
    population_tif: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    data, coarse = mosaic([population_tif], grid.bounds, 200.0)
    assert coarse.shape == (32, 32)
    assert coarse.resolution == (200.0, 200.0)
    finite = data[np.isfinite(data)]
    assert finite.size > 0
    assert np.isin(finite, population[np.isfinite(population)]).all()


def test_mosaic_rejects_mixed_crs_and_nothing(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray, population_tif: Path
) -> None:
    lonlat = BlockGrid(crs=WGS84, transform=Affine(0.001, 0, 0, 0, -0.001, 1), height=64, width=64)
    other = tmp_path / "lonlat.tif"
    with rasterio.open(other, "w", **lonlat.profile()) as dst:
        dst.write(population, 1)
    with pytest.raises(ValueError, match="expected"):
        mosaic([population_tif, other], grid.bounds)
    with pytest.raises(ValueError, match="at least one"):
        mosaic([], grid.bounds)


def test_write_geotiff_round_trip(tmp_path: Path, grid: BlockGrid, population: np.ndarray) -> None:
    path = write_geotiff_atomic(tmp_path / "out" / "pop.tif", population, grid, nodata=np.nan)
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))
    with rasterio.open(path) as src:
        assert src.nodata is not None
        assert np.isnan(src.nodata)
        assert src.compression is not None
        np.testing.assert_array_equal(src.read(1), population)
    assert grid_from_raster(path) == grid


def test_write_geotiff_bands_and_integer_nodata(tmp_path: Path, grid: BlockGrid) -> None:
    stack = np.stack([np.ones(grid.shape, dtype="int32"), np.full(grid.shape, 2, dtype="int32")])
    path = write_geotiff_atomic(tmp_path / "stack.tif", stack, grid, nodata=-1)
    with rasterio.open(path) as src:
        assert src.count == 2
        assert src.nodata == -1
        assert src.dtypes == ("int32", "int32")
        assert src.read(2).max() == 2
    narrowed = write_geotiff_atomic(tmp_path / "u8.tif", stack[0], grid, nodata=255, dtype="uint8")
    with rasterio.open(narrowed) as src:
        assert src.dtypes == ("uint8",)


def test_narrowing_maps_nan_to_nodata_and_keeps_every_value(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    counts = np.floor(population)  # integral floats with the NaN corner
    path = write_geotiff_atomic(tmp_path / "counts.tif", counts, grid, nodata=-1, dtype="int32")
    with rasterio.open(path) as src:
        back = src.read(1)
        assert src.nodata == -1
    missing = np.isnan(counts)
    assert (back[missing] == -1).all()
    np.testing.assert_array_equal(back[~missing], counts[~missing].astype("int32"))


def test_narrowing_lands_an_extreme_nodata_exactly(tmp_path: Path, grid: BlockGrid) -> None:
    # float32 cannot hold INT_MAX; the fill must happen in the integer array, not the float one.
    data = np.ones(grid.shape, dtype="float32")
    data[0, 0] = np.nan
    nodata = int(np.iinfo("int32").max)
    path = write_geotiff_atomic(tmp_path / "extreme.tif", data, grid, nodata=nodata, dtype="int32")
    with rasterio.open(path) as src:
        back = src.read(1)
        assert src.nodata == nodata
    assert back[0, 0] == nodata
    assert (back[1:] == 1).all()
    with pytest.raises(ValueError, match="do not fit"):
        write_geotiff_atomic(
            tmp_path / "x.tif",
            np.full(grid.shape, 2.0**31, dtype="float32"),
            grid,
            nodata=-1,
            dtype="int32",
        )


def test_narrowing_compares_wide_integers_exactly(tmp_path: Path, grid: BlockGrid) -> None:
    # INT64_MAX rounds up to 2**63 as a float; the range check must not go through float().
    top = np.iinfo("int64").max
    data = np.full(grid.shape, top, dtype="uint64")
    path = write_geotiff_atomic(tmp_path / "wide.tif", data, grid, nodata=None, dtype="int64")
    with rasterio.open(path) as src:
        assert src.dtypes == ("int64",)
        assert int(src.read(1)[0, 0]) == top
    with pytest.raises(ValueError, match="do not fit"):
        write_geotiff_atomic(
            tmp_path / "x.tif",
            np.full(grid.shape, 2**63, dtype="uint64"),
            grid,
            nodata=None,
            dtype="int64",
        )


def test_mosaic_with_a_resolution_covers_the_bounds(tmp_path: Path) -> None:
    tile_grid = BlockGrid(
        crs=WGS84, transform=Affine(1 / 120, 0, 10.0, 0, -1 / 120, 50.0), height=120, width=120
    )
    path = tmp_path / "tile.tif"
    with rasterio.open(path, "w", **tile_grid.profile()) as dst:
        dst.write(np.ones(tile_grid.shape, dtype="float32"), 1)
    request = (10.0, 49.0, 10.0 + 10.5 / 60, 49.0 + 10.5 / 60)
    data, grid = mosaic([path], request, 1 / 60)
    assert grid.shape == (11, 11)
    assert data.shape == (11, 11)
    assert grid.bounds[0] == request[0]
    assert grid.bounds[3] == pytest.approx(request[3])
    assert grid.bounds[2] >= request[2]
    assert grid.bounds[1] <= request[1]


def test_narrowing_refuses_what_an_integer_cannot_hold(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    target = tmp_path / "x.tif"
    with pytest.raises(ValueError, match="non-integral"):
        write_geotiff_atomic(target, population, grid, nodata=-1, dtype="int32")
    with pytest.raises(ValueError, match="no nodata"):
        write_geotiff_atomic(target, np.floor(population), grid, nodata=None, dtype="int32")
    with pytest.raises(ValueError, match="do not fit"):
        write_geotiff_atomic(
            target, np.full(grid.shape, 300, dtype="int32"), grid, nodata=255, dtype="uint8"
        )
    assert not target.exists()


def test_write_geotiff_refuses_bad_nodata(tmp_path: Path, grid: BlockGrid) -> None:
    target = tmp_path / "ints.tif"
    with pytest.raises(ValueError, match="representable"):
        write_geotiff_atomic(target, np.ones(grid.shape, dtype="int32"), grid, nodata=np.nan)
    assert not target.exists()
    with pytest.raises(ValueError, match="range"):
        write_geotiff_atomic(target, np.ones(grid.shape, dtype="uint8"), grid, nodata=-1)


def test_write_geotiff_refuses_shape_mismatch(tmp_path: Path, grid: BlockGrid) -> None:
    with pytest.raises(ValueError, match="does not match"):
        write_geotiff_atomic(tmp_path / "x.tif", np.ones((3, 3)), grid, nodata=np.nan)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        write_geotiff_atomic(tmp_path / "x.tif", np.ones(10), grid, nodata=np.nan)


def test_failed_write_leaves_nothing_behind(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        msg = "header mismatch (simulated)"
        raise ValueError(msg)

    monkeypatch.setattr(raster_io, "_check_header", broken)
    target = tmp_path / "pop.tif"
    with pytest.raises(ValueError, match="simulated"):
        write_geotiff_atomic(target, population, grid, nodata=np.nan)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_replaces_only_on_success(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old")

    def failing(tmp: Path) -> None:
        tmp.write_text("new")
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        atomic_write(target, failing)
    assert target.read_text() == "old"
    assert not (tmp_path / "a.txt.tmp").exists()
    atomic_write(target, lambda tmp: tmp.write_text("new"))
    assert target.read_text() == "new"


def test_plain_write_geotiff_writes_the_given_path_with_the_same_checks(
    tmp_path: Path, grid: BlockGrid, population: np.ndarray
) -> None:
    target = tmp_path / "plain.tif.tmp"
    assert write_geotiff(target, population, grid, nodata=np.nan) == target
    assert [path.name for path in tmp_path.iterdir()] == ["plain.tif.tmp"]
    atomic = write_geotiff_atomic(tmp_path / "pop.tif", population, grid, nodata=np.nan)
    with rasterio.open(target) as plain, rasterio.open(atomic) as through_rename:
        assert plain.driver == "GTiff"
        plain_profile, renamed_profile = dict(plain.profile), dict(through_rename.profile)
        assert np.isnan(plain_profile.pop("nodata"))
        assert np.isnan(renamed_profile.pop("nodata"))
        assert plain_profile == renamed_profile
        np.testing.assert_array_equal(plain.read(1), through_rename.read(1))
    assert grid_from_raster(target) == grid
    with pytest.raises(ValueError, match="does not match the grid"):
        write_geotiff(tmp_path / "bad.tif", population[:10], grid, nodata=np.nan)
    with pytest.raises(ValueError, match="not representable"):
        write_geotiff(tmp_path / "bad.tif", np.ones(grid.shape, "int32"), grid, nodata=np.nan)
    assert not (tmp_path / "bad.tif").exists()
