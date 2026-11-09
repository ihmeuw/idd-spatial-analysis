"""Bound strings."""

from __future__ import annotations

import pytest

from idd_spatial_analysis.geometry import parse_bound


@pytest.mark.parametrize(
    ("text", "wrap", "expected"),
    [
        ("260E", "none", 260.0),
        ("260E", "180", -100.0),
        ("260E", "360", 260.0),
        ("100W", "none", -100.0),
        ("100W", "360", 260.0),
        ("180W", "180", -180.0),
        ("45S", "none", -45.0),
        ("45S", "360", -45.0),
        ("12.5N", "180", 12.5),
        (" 30e ", "none", 30.0),
    ],
)
def test_parse_bound(text: str, wrap: str, expected: float) -> None:
    assert parse_bound(text, wrap=wrap) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["260", "E260", "-45S", "45X", ""])
def test_parse_bound_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="bound string"):
        parse_bound(text)
