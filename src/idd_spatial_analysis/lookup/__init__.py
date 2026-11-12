"""Lookup products: spatial assignments computed once, written to disk and read by every job.

``zone_lookup`` is the zone id of every pixel of a block frame. With it on disk a weighted
mean per zone needs no polygons: read the block's mask, then ``zonal.weighted_sums``.

``zone_cell_weights`` is a weight raster summed onto (zone, coarse cell) pairs, with the
coarse grid recorded. With it on disk a weighted mean of any field on that grid is a product
over the table's rows: ``load_zone_cell_weights(directory, label).apply(field, field_grid)``.
"""

from idd_spatial_analysis.lookup.zone_cell_weights import (
    ZoneCellWeights,
    build_zone_cell_weights_block,
    collect_zone_cell_weights,
    combine_zone_cell_weights,
    compute_zone_cell_weights_block,
    load_zone_cell_weights,
    zone_cell_weights_block_path,
    zone_cell_weights_path,
)
from idd_spatial_analysis.lookup.zone_lookup import (
    ZoneLookupBlock,
    build_zone_lookup_block,
    collect_zone_lookup,
    compute_zone_lookup_block,
    load_zone_lookup_index,
    load_zone_lookup_metadata,
    load_zone_lookup_summary,
    read_zone_lookup_block,
    write_zone_lookup_block,
    zone_lookup_block_path,
    zone_lookup_block_table_path,
)

__all__ = [
    "ZoneCellWeights",
    "ZoneLookupBlock",
    "build_zone_cell_weights_block",
    "build_zone_lookup_block",
    "collect_zone_cell_weights",
    "collect_zone_lookup",
    "combine_zone_cell_weights",
    "compute_zone_cell_weights_block",
    "compute_zone_lookup_block",
    "load_zone_cell_weights",
    "load_zone_lookup_index",
    "load_zone_lookup_metadata",
    "load_zone_lookup_summary",
    "read_zone_lookup_block",
    "write_zone_lookup_block",
    "zone_cell_weights_block_path",
    "zone_cell_weights_path",
    "zone_lookup_block_path",
    "zone_lookup_block_table_path",
]
