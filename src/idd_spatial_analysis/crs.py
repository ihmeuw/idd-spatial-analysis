"""Small CRS helpers shared by every subpackage."""

from __future__ import annotations

from typing import Any

from pyproj import CRS


def crs_string(crs: Any) -> str:
    """A compact authority string (``EPSG:4326``) when one exists, else WKT."""
    parsed = CRS.from_user_input(crs)
    authority = parsed.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return str(parsed.to_wkt())


def crs_equal(left: Any, right: Any) -> bool:
    """True when the two CRS descriptions denote the same CRS."""
    return bool(CRS.from_user_input(left).equals(CRS.from_user_input(right)))


def is_geographic(crs: Any) -> bool:
    """True for lat/lon CRSs (degrees), False for projected ones (metres)."""
    return bool(CRS.from_user_input(crs).is_geographic)


def is_projected_metres(crs: Any) -> bool:
    """True when the CRS is projected and its axis unit is the metre."""
    parsed = CRS.from_user_input(crs)
    if not parsed.is_projected:
        return False
    units = {axis.unit_name for axis in parsed.axis_info}
    return units <= {"metre", "meter", "m"}
