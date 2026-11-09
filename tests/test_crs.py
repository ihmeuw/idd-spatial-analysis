"""CRS helpers."""

from __future__ import annotations

from pyproj import CRS

from idd_spatial_analysis.crs import crs_equal, crs_string, is_geographic, is_projected_metres


def test_crs_string_prefers_an_authority_and_falls_back_to_wkt() -> None:
    assert crs_string("EPSG:4326") == "EPSG:4326"
    assert crs_string(CRS.from_epsg(4326)) == "EPSG:4326"
    assert crs_string("ESRI:54034") == "ESRI:54034"
    custom = "+proj=cea +lon_0=12 +lat_ts=0 +datum=WGS84 +units=m +no_defs"
    text = crs_string(custom)
    assert text.startswith(("PROJCRS", "PROJCS"))
    assert "12" in text


def test_crs_equal_sees_through_representations() -> None:
    assert crs_equal("EPSG:4326", CRS.from_epsg(4326).to_wkt())
    assert crs_equal(CRS.from_user_input("ESRI:54034"), "ESRI:54034")
    assert not crs_equal("EPSG:4326", "ESRI:54034")


def test_geographic_and_metre_checks() -> None:
    assert is_geographic("EPSG:4326")
    assert not is_geographic("ESRI:54034")
    assert is_projected_metres("ESRI:54034")
    assert is_projected_metres("EPSG:32601")
    assert not is_projected_metres("EPSG:4326")
    assert not is_projected_metres("EPSG:2227")  # US survey feet
