"""Longitude conventions: cutting, shifting and welding at the seams."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from idd_spatial_analysis.constants import WGS84, WORLD_CYLINDRICAL_EQUAL_AREA
from idd_spatial_analysis.geometry import (
    check_lon_180,
    split_antimeridian,
    split_at_meridian,
    to_lon_180,
    to_lon_180_gdf,
    to_lon_360,
    to_lon_360_gdf,
)


def test_polygon_entirely_positive_is_unchanged() -> None:
    poly = box(10, 10, 20, 20)
    assert to_lon_360(poly).equals(poly)
    assert to_lon_180(poly).equals(poly)


def test_polygon_entirely_negative_shifts_whole() -> None:
    poly = box(-120, 30, -100, 40)
    moved = to_lon_360(poly)
    assert moved.equals(box(240, 30, 260, 40))
    assert to_lon_180(moved).equals(poly)


def test_prime_meridian_crossing_splits_then_welds_back() -> None:
    poly = box(-10, -5, 10, 5)
    moved = to_lon_360(poly)
    assert isinstance(moved, MultiPolygon)
    assert len(moved.geoms) == 2
    assert moved.bounds == (0.0, -5.0, 360.0, 5.0)
    assert moved.area == pytest.approx(poly.area)
    back = to_lon_180(moved)
    assert isinstance(back, Polygon)
    assert back.equals(poly) or back.symmetric_difference(poly).area < 1e-9


def test_antimeridian_crossing_in_360_splits_into_edge_pieces_and_welds_back() -> None:
    fiji_like = box(170, -20, 190, -10)
    pieces = to_lon_180(fiji_like)
    assert isinstance(pieces, MultiPolygon)
    assert pieces.bounds == (-180.0, -20.0, 180.0, -10.0)
    assert pieces.area == pytest.approx(fiji_like.area)
    whole = to_lon_360(pieces)
    assert isinstance(whole, Polygon)
    assert whole.symmetric_difference(fiji_like).area < 1e-9


def test_antimeridian_halves_from_a_shapefile_weld_despite_a_hairline_gap() -> None:
    west = box(170, -20, 180 - 2e-7, -10)
    east = box(-180, -20, -170, -10)
    whole = to_lon_360(MultiPolygon([west, east]))
    assert isinstance(whole, Polygon)
    assert whole.area == pytest.approx(box(170, -20, 190, -10).area, rel=1e-6)


def test_polygon_with_hole_keeps_its_hole() -> None:
    outer = [(-5, -5), (-5, 5), (-1, 5), (-1, -5)]
    hole = [(-4, -4), (-4, 4), (-2, 4), (-2, -4)]
    poly = Polygon(outer, [hole])
    moved = to_lon_360(poly)
    assert moved.area == pytest.approx(poly.area)
    assert moved.bounds == (355.0, -5.0, 359.0, 5.0)


def test_points_and_lines() -> None:
    assert to_lon_360(Point(-10, 0)).equals(Point(350, 0))
    line = LineString([(-10, 0), (10, 0)])
    moved = to_lon_360(line)
    assert moved.length == pytest.approx(line.length)
    assert moved.bounds == (0.0, 0.0, 360.0, 0.0)


def test_points_and_lines_on_the_cut_stay_put() -> None:
    assert to_lon_360(Point(0.0, 10.0)).equals(Point(0.0, 10.0))
    assert to_lon_180(Point(180.0, 10.0)).equals(Point(180.0, 10.0))
    multi = to_lon_360(MultiPoint([(0.0, 10.0), (-5.0, 5.0)]))
    assert sorted(point.x for point in multi.geoms) == [0.0, 355.0]
    along = LineString([(0.0, -5.0), (0.0, 5.0)])
    assert to_lon_360(along).equals(along)


def test_lines_crossing_the_seam_move_whatever_their_vertex_order() -> None:
    for line in (LineString([(170, 0), (190, 0)]), LineString([(190, 0), (170, 0)])):
        moved = to_lon_180(line)
        assert moved.bounds == (-180.0, 0.0, 180.0, 0.0)
        assert moved.length == pytest.approx(20.0)


def test_empty_geometries_keep_their_type() -> None:
    for empty in (Polygon(), MultiPolygon(), Point(), LineString()):
        assert to_lon_360(empty).geom_type == empty.geom_type
        assert to_lon_180(empty).is_empty


def test_split_at_meridian_returns_input_when_not_crossing() -> None:
    poly = box(10, 10, 20, 20)
    assert split_at_meridian(poly, 0.0) == [poly]
    touching = box(0, 0, 10, 10)
    assert len(split_at_meridian(touching, 0.0)) == 1


def test_unsupported_geometry_raises() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        to_lon_360(GeometryCollection([Point(0, 0)]))


def test_split_antimeridian_cuts_at_180() -> None:
    pieces = split_antimeridian(box(170, -20, 190, -10))
    assert len(pieces) == 2
    assert sorted(piece.bounds[0] for piece in pieces) == [170.0, 180.0]


def test_gdf_wrappers_keep_columns_index_and_crs() -> None:
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[box(-10, 0, -5, 5), box(5, 0, 10, 5)],
        crs=WGS84,
        index=[7, 9],
    )
    moved = to_lon_360_gdf(gdf)
    assert list(moved.columns) == ["name", "geometry"]
    assert list(moved.index) == [7, 9]
    assert moved.crs == gdf.crs
    assert moved.geometry.iloc[0].bounds == (350.0, 0.0, 355.0, 5.0)
    assert to_lon_180_gdf(moved).geometry.iloc[0].equals(gdf.geometry.iloc[0])


def test_gdf_wrappers_use_the_active_geometry_column() -> None:
    gdf = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[box(-10, 0, -5, 5)], crs=WGS84
    ).rename_geometry("geom")
    moved = to_lon_360_gdf(gdf)
    assert list(moved.columns) == ["name", "geom"]
    assert moved.geometry.name == "geom"
    assert moved.geometry.iloc[0].bounds == (350.0, 0.0, 355.0, 5.0)
    assert moved.crs == gdf.crs


def test_check_lon_180() -> None:
    fine = gpd.GeoDataFrame(geometry=[box(-180, -10, 180, 10)], crs=WGS84)
    check_lon_180(fine)
    check_lon_180(fine.to_crs(WORLD_CYLINDRICAL_EQUAL_AREA))
    check_lon_180(fine.iloc[:0])
    with pytest.raises(ValueError, match="to_lon_180"):
        check_lon_180(gpd.GeoDataFrame(geometry=[box(170, -20, 190, -10)], crs=WGS84))
