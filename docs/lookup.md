# Lookup products

A lookup product is a spatial assignment computed once, written to disk and read by every job
that needs it. There are two: the zone lookup (the zone id of every pixel) and the zone cell
weights (a weight raster summed onto pairs of zone and coarse cell). The functions for both
are here; the shared build of the first exists in the output node, the shared build of the
second does not yet.

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

## Zone cell weights: a weight raster on (zone, coarse cell) pairs

For a covariate on a coarser grid than the block frame, each pixel takes the value of one
coarse cell, so the weighted sum over a zone's pixels equals a sum over cells of the cell's
value times the zone's weight inside that cell. Those weights depend on the zones, the weight
raster and the coarse grid, not on the covariate. Once they are on disk, the two sums per
zone for any field on that grid are a product over the table's rows; the pixels are not
touched again.

### What the directory holds

| Path | Written by | Content |
|---|---|---|
| `block_weights/<label>/<block_key>.parquet` | `build_zone_cell_weights_block` | one block and one weight raster: `zone`, `coarse_row`, `coarse_col`, `weight` |
| `weights/<label>.parquet` | `collect_zone_cell_weights` | the blocks of one label added up, same columns; the table `apply` uses |

`label` names the weight raster, normally the population year. Every file carries, in its
parquet schema metadata, the coarse grid it was built for (CRS, transform, shape), the block
frame's CRS and resolution, and counts: zoned pixels, pairs, total weight, the pixels and
weight the coarse grid did not cover, and how the pixel centres were transformed. A constant
such as the year is in the path and the metadata, not in a column.

A pair appears when at least one of its pixels has a finite weight: a pair whose weights are
all zero stays (the zone still gets a row with a zero weight sum, as in `weighted_sums`), a
pair whose weights are all NaN does not.

### Building one, in rounds

Each file has exactly one writer, so the work splits into rounds that can run as separate
tasks.

```python
from idd_spatial_analysis.frame import grid_from_coords
from idd_spatial_analysis.lookup import (
    build_zone_cell_weights_block,
    collect_zone_cell_weights,
    read_zone_lookup_block,
)
from idd_spatial_analysis.raster import read_window
from idd_spatial_analysis.zonal import ZoneCellPairs

coarse = grid_from_coords(lon, lat, "EPSG:4326", registration="center")

# Round 1, per block: pair the zoned pixels with coarse cells once, then one table per year.
for key in keys:
    mask, grid = read_zone_lookup_block(zone_lookup_directory, key)
    pairs = ZoneCellPairs.build(mask, grid, coarse)
    for year in years:
        population, _ = read_window(population_tif(year, key), grid.bounds, boundless=True)
        build_zone_cell_weights_block(pairs, population, directory, str(year), key)

# Round 2, per year, once its blocks are there: add the blocks up.
for year in years:
    collect_zone_cell_weights(directory, str(year), keys, zone_set="my_zones")
```

The pairing depends on the zones and the two grids only, so one pairing serves every year of
a block, and a later batch of years repeats round 1 for those years alone. A zone that
straddles blocks can share a coarse cell across the boundary; `collect_zone_cell_weights`
adds those weights. It refuses to finish when a block has no table for the label, when the
label's directory holds a block that is not in `block_keys`, or when blocks were built for
different coarse grids.

A zoned pixel with a finite weight whose centre lies outside the coarse grid is an error,
because its weight would vanish from every mean. `allow_uncovered=True` leaves such pixels
out and records their count and weight in the file's metadata.

### Using one

```python
from idd_spatial_analysis.frame import grid_from_coords
from idd_spatial_analysis.lookup import load_zone_cell_weights
from idd_spatial_analysis.zonal import rollup_hierarchy

weights = load_zone_cell_weights(directory, "2020")
field_grid = grid_from_coords(field_lon, field_lat, "EPSG:4326", registration="center")
sums = weights.apply(field, field_grid)  # zone, weighted_sum, weight_sum
by_level = rollup_hierarchy(sums, hierarchy, leaves=with_pixels)
```

`apply` takes the field and the grid the field is on, and refuses a grid that is not the one
the table was built for: same CRS, same shape, cell edges within a thousandth of a cell
(`frame.grids_match`, `GRID_MATCH_TOLERANCE`). The tolerance lets the same grid be described
from float32 and from float64 coordinates; a grid half a cell away, which is what reading
cell-centre coordinates as cell edges produces, is refused. The result has the columns of
`weighted_sums` and equals what `weighted_sums` gives when the field is put on the pixels
pixel-centre nearest, including the `nan_policy` choice. A stack `(slices, rows, cols)` gives
a `slice` column.

`ZoneCellWeights.from_table(table)` wraps a table that is still in memory, so the whole
calculation can run without writing anything:
`combine_zone_cell_weights([compute_zone_cell_weights_block(pairs, weight), ...], label=..., zone_set=...)`.

### One transformed row and column instead of every pixel

Pairing a pixel with a coarse cell needs the pixel's centre in the coarse grid's CRS. Between
the equal-area block frame and lon/lat, longitude depends on the pixel's column alone and
latitude on its row alone, so transforming one row and one column of centres gives every
pixel's coordinates. `NearestIndexMap.build` takes that route only after checking it: it
transforms a sample of lattice points one by one (33 positions per axis, edges and corners
included) and requires every coordinate to equal the row-and-column one exactly. If any
differs, or the middle row or column does not transform to finite coordinates, every centre
is transformed as before. `centre_transform` on the lookup and on `ZoneCellPairs` says which
route was taken, and each weight file records it. The check is a sample, not a proof.

## Writing through another atomic writer

Computing and writing are separate steps in both products. The `build_` and `collect_`
functions above write with this package's own atomic writer (temp file beside the target,
header or footer check, rename). A job that has to write through another one, such as a
jobmon worker whose writes must enter idd-tools' task registry, calls the compute step and
writes the result itself:

```python
from idd_tools.jobmon.atomic_io import AtomicRegistry, atomic_write, atomic_write_parquet

from idd_spatial_analysis.lookup import (
    compute_zone_cell_weights_block,
    compute_zone_lookup_block,
    zone_cell_weights_block_path,
    zone_lookup_block_path,
    zone_lookup_block_table_path,
)

with AtomicRegistry(directory, task_id=task_id, task_template=template) as registry:
    block = compute_zone_lookup_block(zones_gdf, "zone_id", grid)
    raster_path = zone_lookup_block_path(directory, key)
    atomic_write(raster_path, block.write_raster)  # the raster first
    registry.register(raster_path)
    atomic_write_parquet(
        block.zone_table(),
        zone_lookup_block_table_path(directory, key),
        registry=registry,
        schema=block.table.schema,  # column types and build parameters
    )

    table = compute_zone_cell_weights_block(pairs, population)
    atomic_write_parquet(
        table.to_pandas(),
        zone_cell_weights_block_path(weights_directory, label, key),
        registry=registry,
        schema=table.schema,
    )
```

Passing the arrow schema keeps the explicit column types and the parameters in the schema
metadata, which the collect steps require. `tests/lookup/test_idd_tools_writers.py` writes
both products this way with idd-tools' real functions and collects them; it runs where the
`jobmon` extra is installed and is skipped elsewhere.

## Measured on the cluster, 2026-11-12

Six blocks of `lsae_1285`, population of three years, annual mean temperature on a global 0.1
degree grid. The pixel pass and the weight table use the same pixel-centre lookup and gave the
same zone means (largest difference 6e-13 degrees over 4,581 zone-years).

| Cost | Pixel pass | Weight table |
|---|---|---|
| Once per block | 9.2 s to build the pixel lookup, for every covariate | 9.8 s to pair pixels with cells, once |
| Per block and year | 4.9 s (read population, put the surface on the pixels, sum), for every covariate and scenario | 1.2 s (read population, sum onto pairs), once per population year |
| Per surface-year afterwards | the line above again | 0.003 s for the six blocks |

Those timings transformed every pixel centre. With one transformed row and column, on the
same six blocks:

| Step, mean per block | Every pixel | Row and column |
|---|---|---|
| Pixel-to-cell lookup of the zoned pixels | 6.3 s | 0.6 s |
| `ZoneCellPairs.build` in full (lookup, then grouping into pairs) | 9.8 s | 3.3 s (4.8 s for a block with every pixel zoned) |

The two routes gave the same cell for every zoned pixel of all six blocks. What remains in
`ZoneCellPairs.build` is the grouping of up to 67 million pixels into pairs.

These are means over six blocks, one process at a time, on one node. They show what each step
costs relative to the others; they are not a basis for sizing jobs, which comes from probes of
the code that will run.

## Running the builds as cluster jobs

This package ships the functions. The launcher and the worker that run them over hundreds of
blocks belong to the repo that needs the result, written with that repo's own job tooling; for
the idd repos that is idd-tools' jobmon layer, and idd-tools' README and docs say how. What
only this package can say is below: what the units of work are, what they share, which file
marks one as done, and which tasks should wait for which. idd-tools names appear as pointers
(as of November 2026); its own documentation governs their use.

A repo that runs these as jobmon tasks asks for `idd-tools[jobmon]` in an optional extra of
its own, so that its CI and its consumers never need the IHME artifactory. This package's own
`jobmon` extra is for the builds this repo runs; a consumer does not need it.

### The units of work

| Kind of work | One cell is | Functions | File that marks the cell done |
|---|---|---|---|
| Zone lookup, per block | a block, for one zone set and resolution | `compute_zone_lookup_block` | `block_zones/<block_key>.parquet` (written after `blocks/<block_key>.tif`) |
| Zone lookup, collect | the whole lookup | `collect_zone_lookup` | `build.yaml` |
| Weights, per block | a block and a weight raster (a year) | `ZoneCellPairs.build` once per block, then `compute_zone_cell_weights_block` per year | `block_weights/<year>/<block_key>.parquet` |
| Weights, combine | a year | `combine_zone_cell_weights` | `weights/<year>.parquet` |
| Apply | a field: a covariate, scenario, year, draw | `load_zone_cell_weights(...).apply`, then `zonal.rollup_hierarchy` | the consumer's own output |

Every file has one writer. That holds as long as a cell is in exactly one task, which is what
a partition of the cells guarantees.

### What shares a load

- The years of one block share the read of its zone mask and the pairing of its pixels with
  coarse cells, so a block's years belong in the same task and are not split across tasks
  (group the cells on the block axis; idd-tools: `rectangular_partition` with the block as
  the fixed axis, or `pack_to_target` grouped by block). One task per block also means no two
  tasks read the same block's files at once.
- Blocks share nothing with each other. How many blocks go into one task is a sizing
  question, answered by probes.
- Apply cells of the same year share the read of that year's table, so group them by year.

### Which tasks wait for which

| You want | Wire the tasks like this |
|---|---|
| Weights for some years, and the zone lookup for your zones is on disk | Weights tasks grouped by block, waiting for nothing. One combine task per year, waiting for every weights task that holds that year. |
| The same, but no zone lookup exists for your zones and others will want it | A lookup round first: lookup tasks per group of blocks, and a collect task waiting for all of them. Each weights task waits for the lookup tasks that hold its blocks. Combine as above. |
| The same, and nobody else needs the lookup | No lookup round. The weights task computes the mask itself (`compute_zone_lookup_block(...).mask`) and writes only the weights. |
| More years later | Run the weights and combine rounds for the new years only. Files that exist are skipped by the done-check; the pairing is recomputed per block, which takes seconds. |
| Weights, then apply, in one workflow | Apply tasks grouped by year, each waiting for that year's combine task. |
| Apply as a separate launch | Nothing to wait for. At task entry, check that `weights/<year>.parquet` exists and fail if it does not (idd-tools: `validate_inputs`). |
| One year, used once | The first row, written to a directory of your own and removed afterwards. For a small zone set no file is needed at all: `ZoneCellWeights.from_table(combine_zone_cell_weights(...))`. |
| A covariate that really is at the frame's resolution | The weights do not apply; this is a pass over the pixels. Cells are block by year by covariate; group by block (shares the mask) and year (shares the weight raster); each task writes the partial sums of its cells (`zonal.weighted_sums`); a combine per covariate and year waits for all blocks and calls `zonal.collate_blocks`. |

Two things to keep when wiring:

- Combine with `combine_zone_cell_weights` and write the table it returns with its schema, not
  with a generic concatenation of the block files. A zone that straddles blocks can have the
  same (zone, cell) pair in two files, which must be added, and the coarse grid in the
  metadata must reach the collected file.
- When a round is launched again by itself, drop the tasks whose files exist (idd-tools:
  `filter_already_done`), so later tasks do not wait for work that is already on disk.

### What to measure

The phases of a weights task that deserve their own timing label: read the mask, pair, read
the weight raster, sum, write. From the single-process measurements above, pairing is the
per-block load and reading the weight raster is most of each year. A full block holds a 268 MB
mask, a 268 MB weight raster and about 1 GB of pair indices. The one process that ran these
steps together with the pixel pass on six blocks (2026-11-12, every pixel centre transformed)
peaked at 6.8 GB; the weights steps alone have not been measured separately.

Those are observations of one process on one node. Bundle sizes and resource asks come from
probes of the code that will run, at the concurrency it will run at: many tasks reading the
same weight rasters at once is the case a lone probe under-measures.

### What this repo builds itself

Only products that many repos share. At present that is the `lsae_1285` zone lookup at 100 m.
Weights for a particular grid, population product and set of years are built by the repo that
needs them, into a directory of its choosing: its own, removed after use, or a shared one.
