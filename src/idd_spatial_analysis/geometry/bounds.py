"""Bound strings such as ``260E`` and ``45S``."""

from __future__ import annotations

import re
from typing import Literal

_BOUND = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([NSEWnsew])\s*$")
_FULL_TURN = 360.0
_HALF_TURN = 180.0

Wrap = Literal["none", "180", "360"]


def parse_bound(text: str, *, wrap: Wrap = "none") -> float:
    """Degrees for a bound like ``"260E"``, ``"45S"`` or ``"100W"``.

    ``S`` and ``W`` are negative. ``wrap="180"`` maps longitudes into -180..180 and
    ``wrap="360"`` into 0..360; the default leaves the number as written, so ``"260E"`` stays
    260. Latitudes are never wrapped.
    """
    match = _BOUND.match(text)
    if match is None:
        msg = f"not a bound string: {text!r} (expected e.g. '260E', '45S')"
        raise ValueError(msg)
    value = float(match.group(1))
    hemisphere = match.group(2).upper()
    if hemisphere in "SW":
        value = -value
    if hemisphere in "NS":
        return value
    if wrap == "180":
        return ((value + _HALF_TURN) % _FULL_TURN) - _HALF_TURN
    if wrap == "360":
        return value % _FULL_TURN
    return value
