# Conventions

Every function in the package assumes the conventions below. They are fixed in
`src/idd_spatial_analysis/constants.py`; nothing picks a different one silently.

## Coordinate reference systems

- CRSs are passed and stored as strings and compared with pyproj (`crs.crs_equal`), so
  `"EPSG:4326"` and the equivalent WKT count as the same CRS.
- `WGS84 = "EPSG:4326"`: what shapefiles, GPS points and climate grids arrive in.
- `WORLD_CYLINDRICAL_EQUAL_AREA = "ESRI:54034"`: the population model's block frame. Metres,
  equal area, the CRS of the tile index and of every 100 m block.
- Metric cell areas (`geometry.grid_cell_areas_km2`) need a metre CRS and refuse others;
  lat/lon grids get exact spherical areas.

## Distances

"Actual distance" means metres along the Earth's surface. Nothing in the package measures a
planar distance in a projected CRS: in the equal-area block frame that is off by the cosine
of the latitude (15 percent at 30 degrees, a factor of two at 60), and in lon/lat it is not
a distance at all. When code elsewhere needs a distance, it uses these, in any CRS:

- `geometry.surface_distance(x1, y1, x2, y2, crs)`: paired points, geodesic metres on the
  WGS84 ellipsoid (pyproj `Geod`), broadcasting like numpy.
- `geometry.nearest_surface_distance(xs, ys, crs, target_x, target_y, target_crs)`: each
  point's metres to its nearest target and the target's index, through a KD-tree on
  Earth-centred coordinates (`GEOCENTRIC_CRS`, EPSG:4978) with the chord turned into an arc;
  within a few parts in a million of the geodesic at block scales. `zonal.nearest_facility_distance`
  is this applied to a zone mask.
- `geometry.geocentric_points` and `geometry.chord_to_arc` are the building blocks for a
  search that needs its own tree.

## Grids and rasters

- A raster is the pair `(array, BlockGrid)`. The array is `(rows, cols)` or a stack
  `(slices, rows, cols)`; the grid holds the CRS, the affine transform and the shape.
- Grids are north-up: positive pixel width, negative pixel height, no rotation or shear.
  Row 0 is the northern (top) edge; latitude decreases down the rows.
- Pixel `(row, col)` covers `[c + col*a, c + (col+1)*a)` in x and `(f + (row+1)*e, f + row*e]`
  in y. A point on a shared edge belongs to the pixel to its right or below it (floor), the
  same rule rasterio uses. "On the edge" allows a rounding error of `PIXEL_EDGE_DECIMALS`
  (one millionth of a pixel), so the rule holds on lattices such as 1/120 degree whose edges
  are not exact binary fractions.
- Pixel centres sit half a pixel in from the edges; every nearest lookup and every
  rasterization decides membership by the centre.
- A nearest lookup between two CRSs moves the pixel centres into the other CRS. When x
  depends on the column alone and y on the row alone (the equal-area block frame against
  lon/lat), one row and one column of centres are transformed instead of every pixel, after
  a sample of points transformed one by one has agreed with them exactly; otherwise every
  centre is transformed. The result is the same either way (`docs/lookup.md`).
- A field that arrives with coordinate arrays instead of a transform (a NetCDF variable with
  `longitude` and `latitude`) becomes a grid through `frame.grid_from_coords`, which requires
  `registration`: `"center"` when a coordinate is the middle of its cell, `"lower_left"` when
  it is the cell's western and southern edge. There is no default. Reading centre coordinates
  as edges moves the whole field half a cell north-east, and every fine pixel then takes the
  value of a neighbouring cell three times out of four.
- Windows (`BlockGrid.window`) are integer `rasterio.windows.Window`s that cover the
  requested bounds; they are clipped to the grid unless asked not to be.

## Longitude

- The default convention is -180..180 (shapefiles, GPS).
- `geometry.to_lon_360` / `to_lon_180` move geometries between conventions by cutting at the
  new seam, translating the far pieces and welding pieces that meet at the seam. The CRS
  metadata stays `EPSG:4326` in both conventions; the consumer must know which one it holds.
- `geometry.parse_bound("260E", wrap="180")` turns bound strings into degrees with an explicit
  wrap; without `wrap` the number stays as written.
- `frame.bbox_in_crs` raises `AntimeridianError` instead of returning a box that spans the
  world.

## Nodata

- Value rasters are float32 with NaN as nodata (`RASTER_FLOAT_DTYPE`). `raster.read_window`
  maps the file's nodata value to NaN unless `as_float=False`.
- Zone masks are int32 with `0` for "no zone" (`ZONE_DTYPE`, `ZONE_NODATA`); zone ids are
  positive.
- `BlockGrid.profile` and `raster.write_geotiff_atomic` refuse a nodata value the dtype
  cannot hold (NaN for an integer raster), the mistake that produced the `nodata value, nan,
  is beyond the valid range of its data type, int32` failure in older notebooks. Narrowing a
  float array to an integer dtype maps NaN to the nodata value and refuses non-integral or
  out-of-range values rather than wrapping them.
- `raster.mosaic` pads pixels no tile covers with NaN (float) or the first tile's nodata;
  integer tiles must declare one.
- Zonal sums take a `nan_policy` for pixels whose value is NaN. The default, `"keep_weight"`,
  leaves the pixel's weight in the denominator, so a missing value counts as zero in the mean;
  that is right when missing means zero (a raster padded to a common extent) and matches the
  forecast-mbp and climate-data aggregations. `"drop_pixel"` removes the pixel from both
  sums, so the mean is over the pixels that have a value. The choice changes the result
  wherever values are missing: say which one a pipeline uses.

## Identifiers and tables

- Admin codes (`ADM{level}_CODE`), zone ids and `location_id` are int32. `loc_id` from LBD
  shapefiles may be missing and is pandas nullable `Int32`, as is the zone id
  `geometry.points_to_zones` returns (`<NA>` outside every zone; a point on a boundary is
  outside with the default `within` predicate and, with `intersects`, takes the zone listed
  last, the same rule `rasterize_zones` applies to overlapping polygons).
- `nearest_facility_distance(..., within_zone=True)` reads each facility's zone from an
  integer column the caller supplies (`facility_zone_col`), normally the output of
  `points_to_zones` on the same polygons the mask came from, so facilities outside the block
  still count for their zone and a facility without a zone is an error.
- Zonal tables use the column names in `constants.py`: `zone`, `slice`, `weighted_sum`,
  `weight_sum`, `mean`, `coarse_row`, `coarse_col`, `weight`, `threshold`,
  `weight_within`, `weight_total`, `share`. Sums accumulate in float64.
- Hierarchy tables carry `location_id`, `parent_id`, `level`, each parent one level up.

## Errors

Functions raise; they never skip or fill in. `ValueError` for inconsistent inputs,
`KeyError` for missing columns or keys, `TypeError` for wrong dtypes or geometry types,
`FileNotFoundError` naming the release directory for a missing shapefile,
`AntimeridianError` for a wrapping box and `PathsFileMissingError` for a missing
`paths.yaml`. A block that no zone reaches is not an error: `rasterize_zones` returns an
all-zero mask and `weighted_sums` an empty table, so a block loop can tell "nothing here"
from "something broke". An empty zones table, a 0..360 lon/lat table that would have to be
reprojected, a missing key in a table to collate, duplicate or incomplete rows in a roll-up,
a roll-up zone whose ancestor also carries sums, negative zone ids, a facility with zone 0,
and zoned pixels a coarse grid does not cover are errors, because each would otherwise
produce a plausible wrong number.
