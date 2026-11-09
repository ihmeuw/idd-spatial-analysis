# Conventions

Every function in the package assumes the conventions below. They are fixed in
`src/idd_spatial_analysis/constants.py`; nothing picks a different one silently.

## Coordinate reference systems

- CRSs are passed and stored as strings and compared with pyproj (`crs.crs_equal`), so
  `"EPSG:4326"` and the equivalent WKT count as the same CRS.
- `WGS84 = "EPSG:4326"`: what shapefiles, GPS points and climate grids arrive in.
- `WORLD_CYLINDRICAL_EQUAL_AREA = "ESRI:54034"`: the population model's block frame. Metres,
  equal area, the CRS of the tile index and of every 100 m block.
- Functions that need metres (`nearest_facility_distance`, metric cell areas) refuse a
  geographic CRS instead of returning degrees.

## Grids and rasters

- A raster is the pair `(array, BlockGrid)`. The array is `(rows, cols)` or a stack
  `(slices, rows, cols)`; the grid holds the CRS, the affine transform and the shape.
- Grids are north-up: positive pixel width, negative pixel height, no rotation or shear.
  Row 0 is the northern (top) edge; latitude decreases down the rows.
- Pixel `(row, col)` covers `[c + col*a, c + (col+1)*a)` in x and `(f + (row+1)*e, f + row*e]`
  in y. A point on a shared edge belongs to the pixel to its right or below it (floor), the
  same rule rasterio uses.
- Pixel centres sit half a pixel in from the edges; every nearest lookup and every
  rasterization decides membership by the centre.
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
- `raster.write_geotiff_atomic` refuses a nodata value the dtype cannot hold (NaN for an
  integer raster), the mistake that produced the `nodata value, nan, is beyond the valid
  range of its data type, int32` failure in older notebooks.
- Zonal sums take a required `nan_policy`: `"keep_weight"` leaves a pixel's weight in the
  denominator when its value is NaN, `"drop_pixel"` removes the pixel from both sums. There is
  no default because the choice changes the result.

## Identifiers and tables

- Admin codes (`ADM{level}_CODE`), zone ids and `location_id` are int32. `loc_id` from LBD
  shapefiles may be missing and is pandas nullable `Int32`.
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
from "something broke".
