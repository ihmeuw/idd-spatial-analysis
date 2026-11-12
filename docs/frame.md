# The frame: tile index, blocks, grids, windows

The population model partitions the world into fixed tiles in the equal-area frame
(`ESRI:54034`). Tiles are grouped into blocks; a block is the unit of work for raster jobs.
This package models that with three objects and a handful of functions.

## Tile index

`<tile_index_root>/<resolution>m/tile_index.parquet` holds one row per tile with a
`block_key`, a `tile_key` and the tile polygon in `ESRI:54034`. `frame.load_tile_index`
reads it and checks the CRS and the `block_key` column; it raises on anything else.

## Blocks

- `frame.block_bounds(tiles, block_key)`: the union bounds of a block's tiles.
- `frame.block_grid(tiles, block_key, resolution)`: the `BlockGrid` of that block at the given
  pixel size, anchored at the block's top-left corner. Pixel counts round up, so the grid
  stays on the lattice of its origin and may overhang the block by less than one pixel.
- `frame.blocks_intersecting(tiles, polygons)`: the sorted keys of every block whose
  footprint intersects at least one polygon (dissolve the tiles by block, spatial join).
  Polygons in another CRS are reprojected. No block at all is an error, so a wrong input
  cannot silently skip every block.
- `frame.block_for_point(tiles, x, y, crs)`: the block containing a point.

## BlockGrid

`BlockGrid(crs, transform, height, width)` is an immutable dataclass. Beyond `shape`,
`resolution` and `bounds` it offers:

- `pixel_centers()`: x of every column centre and y of every row centre.
- `rowcol(xs, ys)` and `contains(rows, cols)`: coordinates to pixel indices by floor.
- `window(bounds, pad=0, clip=True)`: the integer window covering some bounds, optionally
  padded, clipped to the grid unless `clip=False` (for boundless reads). Bounds a rounding
  error from a pixel edge snap to it, so `grid.window(grid.sub(w).bounds) == w` on every
  lattice.
- `sub(window)`: the grid of that window.
- `lonlat_bounds(pad=0)`: the lon/lat box covering the grid.
- `profile(dtype, nodata, count)`: a GeoTIFF profile (LZW, tiled at 512 when the grid is
  large enough); it refuses a nodata the dtype cannot hold, so an integer profile needs an
  explicit integer nodata or `None`.

`grid_from_bounds(bounds, resolution, crs)` builds a grid for arbitrary bounds and
`grid_from_raster(path)` reads one from a file. `grid_from_coords(xs, ys, crs, registration=...)`
builds the grid that a field's coordinate arrays label: `xs` ascending, `ys` descending, evenly
spaced, and `registration` stated (`"center"` or `"lower_left"`, no default).

## Bounds across CRSs

`frame.bbox_in_crs(bounds, src_crs, dst_crs)` densifies the edges before reprojecting, so a
box in the equal-area frame maps to a box that really covers it in lon/lat. It raises when the
result is not finite (outside the valid area) or, for a geographic target, when the box
crosses the antimeridian, which GDAL reports as a left edge east of the right edge, or when it
is wider than 180 degrees (`AntimeridianError` in both cases): split the geometry with
`geometry.split_antimeridian` first.

## A block loop

```python
tiles = load_tile_index(paths["tile_index_root"] / "100m" / "tile_index.parquet")
zones = load_raking_shapes(paths["raking_root"], "gbd_2023")
partials = []
for block_key in blocks_intersecting(tiles, zones):
    grid = block_grid(tiles, block_key, resolution=100.0)
    population, _ = read_window(population_tif, grid.bounds, boundless=True)
    mask, windows = rasterize_zones(zones, "location_id", grid)
    partials.append(weighted_sums(mask, population, value_on_grid, nan_policy="drop_pixel"))
sums = collate_blocks(partials)
```

Zones that straddle blocks get one partial row per block; `collate_blocks` adds them.
