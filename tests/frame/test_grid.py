"""BlockGrid: geometry of a north-up raster grid."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.windows import Window

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.frame import BlockGrid, grid_from_bounds, grid_from_raster
from idd_spatial_analysis.frame.grid import check_nodata, pixel_quotient


def test_shape_resolution_bounds(grid: BlockGrid) -> None:
    assert grid.shape == (64, 64)
    assert grid.resolution == (100.0, 100.0)
    assert grid.bounds == (0.0, 0.0, 6400.0, 6400.0)


def test_pixel_centers_are_offset_by_half_a_pixel(grid: BlockGrid) -> None:
    xs, ys = grid.pixel_centers()
    assert xs[0] == 50.0
    assert xs[-1] == 6350.0
    assert ys[0] == 6350.0
    assert ys[-1] == 50.0


def test_rowcol_floors_and_contains(grid: BlockGrid) -> None:
    rows, cols = grid.rowcol(
        np.array([0.0, 99.9, 100.0, 6399.9, -1.0]), np.array([6400.0, 6300.1, 6300.0, 0.1, 7000.0])
    )
    assert cols.tolist() == [0, 0, 1, 63, -1]
    assert rows.tolist() == [0, 0, 1, 63, -6]
    assert grid.contains(rows, cols).tolist() == [True, True, True, True, False]


def test_window_covers_bounds_with_pad_and_clips(grid: BlockGrid) -> None:
    window = grid.window((150.0, 150.0, 350.0, 350.0))
    assert window == Window(1, 60, 3, 3)
    padded = grid.window((150.0, 150.0, 350.0, 350.0), pad=2)
    assert padded == Window(0, 58, 6, 6)
    overflow = grid.window((6000.0, -500.0, 9000.0, 300.0))
    assert overflow == Window(60, 61, 4, 3)


@pytest.mark.parametrize("bounds", [(10.0, 10.0, 5.0, 20.0), (7000.0, 0.0, 8000.0, 100.0)])
def test_window_rejects_degenerate_or_disjoint(
    grid: BlockGrid, bounds: tuple[float, float, float, float]
) -> None:
    with pytest.raises(ValueError, match="bounds"):
        grid.window(bounds)


def test_sub_grid_keeps_lattice(grid: BlockGrid) -> None:
    window = grid.window((1000.0, 1000.0, 2000.0, 2000.0))
    sub = grid.sub(window)
    assert sub.crs == grid.crs
    assert sub.shape == (10, 10)
    assert sub.bounds == (1000.0, 1000.0, 2000.0, 2000.0)


@pytest.mark.parametrize(
    ("transform", "size"),
    [
        (Affine(1 / 120, 0, -180.0, 0, -1 / 120, 60.0), 1200),
        (Affine(0.1, 0, -73.3, 0, -0.1, 45.7), 400),
    ],
)
def test_window_round_trips_sub_grid_bounds_on_any_lattice(transform: Affine, size: int) -> None:
    grid = BlockGrid(crs=WGS84, transform=transform, height=size, width=size)
    for offset in range(0, size - 100, 7):
        window = Window(offset, offset, 100, 100)
        assert grid.window(grid.sub(window).bounds) == window
    assert grid.window(grid.bounds, clip=False) == Window(0, 0, size, size)


def test_rowcol_puts_edge_coordinates_in_the_pixel_to_the_right_and_below() -> None:
    grid = BlockGrid(
        crs=WGS84, transform=Affine(1 / 120, 0, -180.0, 0, -1 / 120, 60.0), height=1200, width=1200
    )
    k = np.arange(1, 50)
    rows, cols = grid.rowcol(-180.0 + k * (1 / 120), 60.0 - k * (1 / 120))
    assert cols.tolist() == k.tolist()
    assert rows.tolist() == k.tolist()


def test_check_nodata_and_pixel_quotient() -> None:
    assert check_nodata("float32", np.nan) == np.dtype("float32")
    assert check_nodata("int32", None) == np.dtype("int32")
    assert check_nodata("uint8", 255) == np.dtype("uint8")
    with pytest.raises(ValueError, match="representable"):
        check_nodata("int16", 1.5)
    with pytest.raises(ValueError, match="range"):
        check_nodata("int16", 40_000)
    assert check_nodata("int64", np.iinfo("int64").max) == np.dtype("int64")
    assert check_nodata("uint64", np.iinfo("uint64").max) == np.dtype("uint64")
    assert check_nodata("int64", np.int64(-5)) == np.dtype("int64")
    with pytest.raises(ValueError, match="range"):
        check_nodata("int64", 2**63)
    quotient = pixel_quotient(np.array([0.0, 100.0 - 1e-9, 250.0]), 0.0, 100.0)
    assert quotient.tolist() == [0.0, 1.0, 2.5]
    assert pixel_quotient(6300.0, 6400.0, -100.0) == 1.0


def test_profile_refuses_nodata_the_dtype_cannot_hold(grid: BlockGrid) -> None:
    with pytest.raises(ValueError, match="representable"):
        grid.profile(dtype="int32")
    with pytest.raises(ValueError, match="range"):
        grid.profile(dtype="uint8", nodata=-1)
    assert grid.profile(dtype="int32", nodata=None)["nodata"] is None


def test_profile_untiled_for_small_grids_tiled_for_large() -> None:
    small = BlockGrid(crs=WGS84, transform=Affine(0.1, 0, 0, 0, -0.1, 10), height=64, width=64)
    assert "tiled" not in small.profile()
    large = BlockGrid(crs=WGS84, transform=Affine(0.1, 0, 0, 0, -0.1, 10), height=1024, width=1024)
    profile = large.profile(dtype="int32", nodata=0)
    assert profile["tiled"] is True
    assert profile["blockxsize"] == 512
    assert profile["dtype"] == "int32"
    assert profile["nodata"] == 0


@pytest.mark.parametrize(
    ("transform", "height", "width", "message"),
    [
        (Affine(100, 0, 0, 0, -100, 0), 0, 10, "positive"),
        (Affine(100, 5, 0, 0, -100, 0), 10, 10, "rotated"),
        (Affine(100, 0, 0, 0, 100, 0), 10, 10, "north-up"),
        (Affine(-100, 0, 0, 0, -100, 0), 10, 10, "north-up"),
    ],
)
def test_invalid_grids_raise(transform: Affine, height: int, width: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BlockGrid(crs=WGS84, transform=transform, height=height, width=width)


def test_grid_from_bounds_rounds_up_to_whole_pixels() -> None:
    grid = grid_from_bounds((0.0, 0.0, 250.0, 200.0), 100.0, WORLD_CYLINDRICAL_EQUAL_AREA)
    assert grid.shape == (2, 3)
    assert grid.bounds == (0.0, 0.0, 300.0, 200.0)
    # Anchored at the top-left corner; a span one part in 1e9 short of a pixel still counts.
    almost = grid_from_bounds((0.0, 0.0, 250.0, 199.9999999), 100.0, WORLD_CYLINDRICAL_EQUAL_AREA)
    assert almost.shape == (2, 3)
    assert almost.bounds[3] == 199.9999999
    rect = grid_from_bounds((0.0, 0.0, 1.0, 1.0), (0.5, 0.25), WGS84)
    assert rect.shape == (4, 2)


@pytest.mark.parametrize(
    ("bounds", "resolution"),
    [((0.0, 0.0, 10.0, 10.0), 0.0), ((0.0, 0.0, 10.0, 10.0), -1.0), ((5.0, 0.0, 5.0, 10.0), 1.0)],
)
def test_grid_from_bounds_rejects_bad_inputs(
    bounds: tuple[float, float, float, float], resolution: float
) -> None:
    with pytest.raises(ValueError, match=r"resolution|degenerate"):
        grid_from_bounds(bounds, resolution, WGS84)


def test_grid_from_raster_round_trips(tmp_path, grid: BlockGrid) -> None:
    path = tmp_path / "g.tif"
    with rasterio.open(path, "w", **grid.profile()) as dst:
        dst.write(np.zeros(grid.shape, dtype="float32"), 1)
    read = grid_from_raster(path)
    assert read.shape == grid.shape
    assert read.transform == grid.transform
    assert read.crs == grid.crs
