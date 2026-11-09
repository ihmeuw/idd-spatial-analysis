# Memory: blocks, windows and what each function allocates

The package never loads a country-sized raster. Work is done per block of the tile index
(3200 x 3200 pixels at 100 m), per window inside a block, or per slice of a stack, and partial
results are small tables that `zonal.collate_blocks` adds up.

## Sizes to keep in mind

| Object | Size |
|---|---|
| One 3200 x 3200 float32 slice | 41 MB |
| The int32 zone mask of a block | 41 MB |
| `NearestIndexMap`, same CRS (separable) | two 1-D int32 arrays, negligible |
| `NearestIndexMap`, different CRS, no mask | two int32 arrays of the target shape, 82 MB for a block |
| `NearestIndexMap` with a mask | proportional to the masked pixels, not the target |
| `nearest_facility_distance` | Earth-centred coordinates of the zoned pixels (24 B per pixel; all at once for `within_zone=False`, one zone at a time otherwise, with `windows` narrowing where each zone's pixels are looked for) plus the KD-tree of the facilities |

## Strategies

- Read windows, not files: `raster.read_window(path, grid.bounds)` reads exactly a block's
  pixels; `boundless=True` pads a block at the raster's edge with NaN instead of returning a
  smaller array.
- Build lookups once: for a stack of slices on the same grids, build one `NearestIndexMap`
  and `apply` it per slice, or pass the stack and let `apply` handle the leading axis.
- Compute only where it matters: `raster.resample_masked` resamples at the masked target
  pixels only (for example the populated ones). In the March 2026 benchmark of the
  tropical-storms stage 3 rerun this masked nearest lookup was 1.5 to 3.7 times faster than
  the full resample and produced the same values; a max-based variant was both wrong and
  slower, and rasterra ignores a masked target.
- Zonal sums are one `numpy.bincount` per slice over the block: no per-zone loops, no copies
  of the block.
- Windows per zone (from `rasterize_zones`) bound the per-zone work in
  `nearest_facility_distance`; pass them when a block holds many zones.
- Stacks of many slices (daily climate fields) are processed per slice inside
  `weighted_sums`; keep only the slices you need in memory.

## Sizing a job

This package exposes shapes and dtypes so a caller can budget (`grid.shape`,
`array.nbytes`); asking Slurm for memory and auditing what a task used is idd-tools' job
(`idd_tools.jobmon` resources and resource audit), not this package's.
