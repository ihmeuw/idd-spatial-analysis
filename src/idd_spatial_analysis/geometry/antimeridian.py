"""Move geometries between the -180..180 and 0..360 longitude conventions.

Both conventions are plain degrees; the CRS metadata stays ``EPSG:4326`` either way, so a
GeoDataFrame in the 0..360 convention must be treated as such by its consumer (a 0..360 raster,
a Pacific-centred map). Polygons are cut at the meridian that becomes the new seam, the pieces
on the far side are translated by 360 degrees, and pieces that meet at the new seam are welded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from shapely import affinity
from shapely.geometry import LineString
from shapely.ops import split, unary_union

from idd_spatial_analysis.crs import is_geographic

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapely.geometry.base import BaseGeometry

_FULL_TURN = 360.0
_HALF_TURN = 180.0
_WELD_EPS = 1e-6
"""Degrees; closes hairline gaps between pieces that were split by a GIS at the meridian."""
_SEAM_TOL = 1e-6
_LON_TOL = 1e-9
_BOTH_SIDES = 2
_POLYGON_TYPES = {"Polygon", "MultiPolygon"}
_SUPPORTED_TYPES = _POLYGON_TYPES | {"Point", "MultiPoint", "LineString", "MultiLineString"}


def _check_supported(geom: BaseGeometry) -> None:
    if geom.geom_type not in _SUPPORTED_TYPES:
        msg = f"unsupported geometry type {geom.geom_type}"
        raise TypeError(msg)


def _parts(geom: BaseGeometry) -> list[BaseGeometry]:
    if geom.geom_type.startswith("Multi"):
        return list(geom.geoms)
    return [geom]


def _crosses(part: BaseGeometry, meridian: LineString) -> bool:
    """True when ``part`` has material on both sides of the meridian.

    A point cannot straddle it, a touching polygon only meets it, and a line lying along it
    has nothing to cut; all three stay whole.
    """
    if part.geom_type == "Point":
        return False
    return bool(
        part.intersects(meridian) and not part.touches(meridian) and not part.within(meridian)
    )


def split_at_meridian(geom: BaseGeometry, longitude: float) -> list[BaseGeometry]:
    """Pieces of ``geom`` on either side of ``longitude``; the geometry itself if it does not cross."""
    _check_supported(geom)
    meridian = LineString([(longitude, -91.0), (longitude, 91.0)])
    pieces: list[BaseGeometry] = []
    for part in _parts(geom):
        if part.is_empty:
            continue
        if _crosses(part, meridian):
            pieces.extend(piece for piece in split(part, meridian).geoms if not piece.is_empty)
        else:
            pieces.append(part)
    return pieces


def split_antimeridian(geom: BaseGeometry) -> list[BaseGeometry]:
    """``split_at_meridian`` at 180 degrees: the pieces of a 0..360 geometry on either side."""
    return split_at_meridian(geom, _HALF_TURN)


def _shift_far_side(
    geom: BaseGeometry, *, cut_at: float, far_side_is_west: bool, shift: float
) -> list[BaseGeometry]:
    pieces = split_at_meridian(geom, cut_at)
    moved: list[BaseGeometry] = []
    for piece in pieces:
        # Every piece lies on one side of the cut (or along it), so the midpoint of its
        # x-extent is strictly on that side; a representative point could sit on the cut.
        anchor = (piece.bounds[0] + piece.bounds[2]) / 2
        far = anchor < cut_at if far_side_is_west else anchor > cut_at
        moved.append(affinity.translate(piece, xoff=shift) if far else piece)
    return moved


def _touches_seam(piece: BaseGeometry, seam: float) -> bool:
    minx, _, maxx, _ = piece.bounds
    return bool(abs(minx - seam) < _SEAM_TOL or abs(maxx - seam) < _SEAM_TOL)


def _assemble(pieces: list[BaseGeometry], *, seam: float, polygonal: bool) -> BaseGeometry:
    merged = unary_union(pieces)
    if not polygonal:
        return merged
    parts = _parts(merged)
    at_seam = [part for part in parts if _touches_seam(part, seam)]
    west = any(abs(part.bounds[2] - seam) < _SEAM_TOL for part in at_seam)
    east = any(abs(part.bounds[0] - seam) < _SEAM_TOL for part in at_seam)
    if len(at_seam) < _BOTH_SIDES or not (west and east):
        return merged
    others = [part for part in parts if not _touches_seam(part, seam)]
    welded = unary_union([part.buffer(_WELD_EPS) for part in at_seam]).buffer(-_WELD_EPS)
    return unary_union([*others, *_parts(welded)])


def to_lon_360(geom: BaseGeometry) -> BaseGeometry:
    """``geom`` in the 0..360 convention: western pieces (x < 0) move east by 360 degrees.

    A polygon crossing the prime meridian becomes two pieces at 0 and 360; a polygon that was
    split at the antimeridian becomes whole again (its halves meet at 180 and are welded). An
    empty geometry comes back unchanged.
    """
    _check_supported(geom)
    if geom.is_empty:
        return geom
    moved = _shift_far_side(geom, cut_at=0.0, far_side_is_west=True, shift=_FULL_TURN)
    return _assemble(moved, seam=_HALF_TURN, polygonal=geom.geom_type in _POLYGON_TYPES)


def to_lon_180(geom: BaseGeometry) -> BaseGeometry:
    """``geom`` in the -180..180 convention: pieces beyond 180 move west by 360 degrees.

    A polygon crossing 180 becomes two pieces at the edges; pieces that meet at the prime
    meridian are welded. An empty geometry comes back unchanged.
    """
    _check_supported(geom)
    if geom.is_empty:
        return geom
    moved = _shift_far_side(geom, cut_at=_HALF_TURN, far_side_is_west=False, shift=-_FULL_TURN)
    return _assemble(moved, seam=0.0, polygonal=geom.geom_type in _POLYGON_TYPES)


def _apply(
    gdf: gpd.GeoDataFrame, convert: Callable[[BaseGeometry], BaseGeometry]
) -> gpd.GeoDataFrame:
    out = gdf.copy()
    name = out.geometry.name
    converted = gpd.GeoSeries(out.geometry.apply(convert), index=out.index, crs=gdf.crs)
    out[name] = converted  # the active geometry column, whatever it is called; CRS kept
    return out


def to_lon_360_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Every geometry through ``to_lon_360``; columns, index and CRS metadata unchanged."""
    return _apply(gdf, to_lon_360)


def to_lon_180_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Every geometry through ``to_lon_180``; columns, index and CRS metadata unchanged."""
    return _apply(gdf, to_lon_180)


def check_lon_180(gdf: gpd.GeoDataFrame) -> None:
    """Raise ``ValueError`` when a lon/lat GeoDataFrame holds longitudes beyond -180..180.

    Reprojecting such geometries (the 0..360 convention) wraps them around the world; convert
    with ``to_lon_180_gdf`` first. Projected or CRS-less frames pass.
    """
    if gdf.empty or gdf.crs is None or not is_geographic(gdf.crs):
        return
    minx, _, maxx, _ = gdf.total_bounds
    if minx < -_HALF_TURN - _LON_TOL or maxx > _HALF_TURN + _LON_TOL:
        msg = (
            f"longitudes span {minx:.6g}..{maxx:.6g}, beyond -180..180; "
            "convert with to_lon_180_gdf first"
        )
        raise ValueError(msg)
