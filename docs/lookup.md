# Lookup products

A lookup product is a spatial assignment computed once, written to disk and read by every job
that needs it. There is one so far, the zone lookup; the zone-by-cell weight table is planned
(last section).

## Zone lookup: the zone id of every pixel

For a table of zone polygons and a block frame, a zone lookup holds one raster per block
whose pixel values are zone ids. With it on disk, a population-weighted mean per zone needs no
polygons: for each block read the mask, the weight and the value, call
`zonal.weighted_sums`, and add the blocks up with `zonal.collate_blocks`.

### What the directory holds

| Path | Written by | Content |
|---|---|---|
| `blocks/<block_key>.tif` | `build_zone_lookup_block` | int32, `0` = no zone, on the block's grid; LZW, tiled |
| `block_zones/<block_key>.parquet` | `build_zone_lookup_block` | the zones with pixels in the block: `zone`, `pixel_count`, and the window (`row_off`, `col_off`, `height`, `width`) holding the zone's pixels there |
| `zones.parquet` | `collect_zone_lookup` | the block tables together, with a `block_key` column |
| `zone_summary.parquet` | `collect_zone_lookup` | every input zone: `pixel_count` over all blocks and `block_count`, zeros included |
| `build.yaml` | `collect_zone_lookup` | zone set, id column, CRS, resolution, `all_touched`, counts, package version, time |

A block that no zone reaches still gets an all-zero raster and an empty table, so a block with
nothing in it can be told from a block that was never built.

### Which zone a pixel gets

The rule is `raster.rasterize_zones`'s: a pixel belongs to the zone that contains its centre
(`all_touched=True`: any zone that touches it); where polygons overlap, the later row of the
zone table wins; a pixel in no polygon is `0`. A zone smaller than a pixel can hold no pixel
centre at all. Such zones are listed with `pixel_count == 0` in `zone_summary.parquet` and
have no row in `weighted_sums`, so a roll-up over them needs
`rollup_hierarchy(..., require_complete=False)` or a `leaves` set without them.

On the equal-area frame every pixel has the same area, so `pixel_count` is also the zone's
area in pixels (hectares at 100 m).

### Building one

Blocks are independent. Build them in any order, in one process or many, then collect.

```python
from idd_spatial_analysis.frame import block_grid, blocks_intersecting, load_tile_index
from idd_spatial_analysis.lookup import build_zone_lookup_block, collect_zone_lookup

tiles = load_tile_index(tile_index_path)
keys = blocks_intersecting(tiles, zones_gdf)
for key in keys:
    grid = block_grid(tiles, key, resolution=100.0)
    build_zone_lookup_block(zones_gdf, "zone_id", grid, directory, key)
collect_zone_lookup(directory, keys, zones_gdf["zone_id"], zone_set="my_zones")
```

`zones_gdf` can be the table of the whole world: only the polygons near a block are
reprojected. `collect_zone_lookup` refuses to finish when a block is missing, when the
directory holds a block of another build, when blocks were built with different parameters,
or when a block names a zone the table does not have.

For the population model's raking hierarchies the command line does the same from
`paths.yaml`:

```
idd-spatial build-zone-lookup --hierarchy lsae_1285 --all-blocks
idd-spatial build-zone-lookup --hierarchy lsae_1285 --block-key B-0032X-0003Y --block-key B-0034X-0004Y
idd-spatial collect-zone-lookup --hierarchy lsae_1285
```

Both need `project_output_root` under `write:` in `paths.yaml`. `--scratch <label>` sends a
test run to the component's `scratch/<label>/` instead of `working/`, and
`collect-zone-lookup --block-key ...` collects just the blocks of such a run.

### Using one

```python
from idd_spatial_analysis.lookup import load_zone_lookup_summary, read_zone_lookup_block
from idd_spatial_analysis.raster import read_window
from idd_spatial_analysis.zonal import collate_blocks, rollup_hierarchy, weighted_sums

partials = []
for key in keys:
    mask, grid = read_zone_lookup_block(directory, key)
    population, _ = read_window(population_tif(key), grid.bounds, boundless=True)
    value, _ = read_window(value_tif(key), grid.bounds, boundless=True)
    partials.append(weighted_sums(mask, population, value))
sums = collate_blocks(partials)
summary = load_zone_lookup_summary(directory)
with_pixels = summary.loc[summary["pixel_count"] > 0, "zone"]
by_level = rollup_hierarchy(sums, hierarchy, leaves=with_pixels)
```

Numerators and denominators add across blocks and up the hierarchy; the divide happens once,
at the end. Passing the zones that have pixels as `leaves` makes the roll-up check that every
one of them arrived with sums. `weighted_sums` treats a NaN value as zero by default
(`nan_policy="keep_weight"`); `"drop_pixel"` leaves such pixels out of both sums
(`docs/conventions.md`).

`load_zone_lookup_index(directory)` says which blocks hold a zone and where, so one zone can
be read without its whole blocks:
`read_zone_lookup_block(directory, block_key, window=Window(col_off, row_off, width, height))`.
`load_zone_lookup_summary` and `load_zone_lookup_metadata` return the other two files.

### Where the shared builds live

`<project_output_root>/02-processed_data/zone_lookup/<zone_set>/<resolution>m/` is one
versioned component (STANDARDS §Output management). Builds write to its `working/` slot,
test runs to `scratch/<label>/`, and a snapshot exists only through `idd-versions freeze`.
Consumers read `current/`, never `working/`.

### Measured on the cluster, 2026-11-11

`lsae_1285` (47,855 admin 2 polygons) on the 100 m frame, 528 blocks of up to 8192 x 8192
pixels:

| Quantity | Value |
|---|---|
| Build, one block | about 3 s, of which about 1 s is the rasterization; 10 to 14 s for the frame's east-edge column, where every polygon is reprojected, and for one Antarctic block |
| Build, all 528 blocks | 8 minutes with four processes, 29 core-minutes in all; no scheduler needed |
| Peak memory, one process | 4.6 to 7.8 GB (the polygon table accounts for most of it) |
| Size on disk | 602 MB for the 528 rasters (median 0.9 MB per block, largest 4.7 MB, against 268 MB in memory) and 8 MB of tables |
| Read one block | 0.6 s |
| `weighted_sums`, one block, one slice | 3.2 s |
| Block and zone pairs in the index | 53,895; 42,587 zones lie in one block, one zone spans 71 |
| Zones without a pixel | 1 of 47,855: a polygon of 8,608 square metres, smaller than one pixel |
| Zoned pixels | 14.73 billion, 147.3 million square kilometres |

Checks made on that build:

- The stored mask of a block is identical, pixel for pixel, to rasterizing its polygons again,
  and the population model's block rasters sit on exactly the same grid.
- Reprojecting only the polygons near a block gives the same mask as reprojecting all of them
  (nine blocks compared, including both edges of the frame, the northernmost row and a block
  at 55 degrees south).
- Zone areas from pixel counts agree with polygon areas on the equal-area frame: the totals
  differ by 7 parts in a million, and each of the 36,779 zones of at least 100 square
  kilometres is within 0.7 percent (1st to 99th percentile within 0.11 percent). The 497 zones
  under 1 square kilometre differ more, as a few pixels must.
- The 47,855 zone ids are exactly the leaves of the `lsae_1285` hierarchy.

### What it saves and what it does not

Rasterizing one block from the polygons takes about 1 s once the polygon table is loaded
(about 4 s per process), and reading the stored mask takes 0.6 s. A job that already loops
over blocks therefore gains little time from the lookup. What the lookup provides is
different:

- one fixed assignment that every consumer shares, since a frozen snapshot is the definition
  of which pixel belongs to which zone;
- no polygons and no reprojection at run time;
- the index of which blocks and windows hold each zone, and each zone's pixel count;
- the input from which a zone-by-cell weight table is built.

A covariate that exists at the frame's resolution still needs one pass over the pixels of
every block for every slice; `weighted_sums` is that pass and dominates the cost above.

## Planned: zone cell weights

For a covariate on a coarser grid, each pixel takes the value of one coarse cell, so the
weighted sum over pixels equals a sum over cells of the cell's value times the zone's weight
inside that cell. Those weights depend on the zones, the weight raster and the coarse grid,
not on the covariate.

The functions exist: `zonal.ZoneCellPairs.build(mask, grid, coarse)` pairs every zoned pixel
of a block with its coarse cell once, `sum_weight` adds a weight raster (one per year) onto
the pairs, and `zonal.apply_zone_weights(table, field)` returns the same two sums per zone
that `weighted_sums` gives on the pixels. The coarse grid comes from
`frame.grid_from_coords(lon, lat, crs, registration=...)`, where the registration is stated,
never assumed (`docs/conventions.md`). The stored product (a table for a whole frame with its
grid recorded, and the commands to build it) is not built yet.

Measured on the cluster on 2026-11-12: six blocks of `lsae_1285`, population of three years,
annual mean temperature on a 0.1 degree grid. Both ways use the same pixel-centre lookup and
gave the same zone means (largest difference 6e-13 degrees over 4,581 zone-years).

| Cost | Pixel pass | Weight table |
|---|---|---|
| Once per block | 9.2 s to build the pixel lookup, for every covariate | 9.8 s to pair pixels with cells, once |
| Per block and year | 4.9 s (read population, put the surface on the pixels, sum), for every covariate and scenario | 1.2 s (read population, sum onto pairs), once per population year |
| Per surface-year afterwards | the line above again | 0.003 s for the six blocks |

Scaled by block count to the 528 blocks of `lsae_1285`, one surface-year costs about 43
core-minutes as a pixel pass. The table costs about 86 core-minutes once plus 10 per
population year, and a surface-year is then a product over the table's rows.
