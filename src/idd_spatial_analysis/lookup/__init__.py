"""Lookup products: spatial assignments computed once, written to disk and read by every job.

``zone_lookup`` is the zone id of every pixel of a block frame. With it on disk a weighted
mean per zone needs no polygons: read the block's mask, then ``zonal.weighted_sums``.
"""

from idd_spatial_analysis.lookup.zone_lookup import (
    build_zone_lookup_block,
    collect_zone_lookup,
    load_zone_lookup_index,
    load_zone_lookup_metadata,
    load_zone_lookup_summary,
    read_zone_lookup_block,
    zone_lookup_block_path,
)

__all__ = [
    "build_zone_lookup_block",
    "collect_zone_lookup",
    "load_zone_lookup_index",
    "load_zone_lookup_metadata",
    "load_zone_lookup_summary",
    "read_zone_lookup_block",
    "zone_lookup_block_path",
]
