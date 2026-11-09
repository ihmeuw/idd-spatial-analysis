# Boundaries: LBD releases and raking inputs

## LBD standard admin shapefiles

A release is a dated directory under `admin_shapefile_root`:

```
<admin_shapefile_root>/2024_07_29/
    lbd_standard_admin_0.shp   lbd_standard_admin_0_simplified.shp
    lbd_standard_admin_1.shp   lbd_standard_admin_1_simplified.shp
    lbd_standard_admin_2.shp   lbd_standard_admin_2_simplified.shp
    lbd_disputed_mask.shp      lbd_disputed_mask_simplified.shp
```

`AdminRelease(root, "2024_07_29", simplified=True)` names one release and one geometry
variant; `.path(level)` and `.require(level)` give the file, the latter raising a
`FileNotFoundError` that names the release directory when it is absent.

`load_admin(release, level, countries=None, columns=None, crs=None)` returns the admin units
of one level:

- `countries` restricts the read to those `ADM0_CODE` values with a `where` clause, so the
  file is filtered before anything is loaded.
- The code columns of every level up to `level` and the level's name column are always
  included; `columns` adds more (for example `loc_id`).
- Admin codes come back as int32; `loc_id`, which can be missing, as nullable `Int32`.
- `crs` reprojects the result; otherwise the file's CRS (lon/lat) is kept.
- An empty result (unknown country codes) is an error, not an empty frame.

## Pinning a release

Nothing in the package chooses "the latest" release. A project names its release once, in its
own configuration, and passes that string everywhere; changing it is a recorded decision. Two
projects can read different releases from the same root.

## Copies on the project store

The archive mount that holds the releases is not visible on every compute node. `copy_release`
copies a release (every shapefile sidecar, both geometry variants by default) under another
root, byte for byte, and returns the `AdminRelease` that reads the copy. The command line does
the same from `paths.yaml`:

```
idd-spatial materialize-admin-release --release 2024_07_29 [--levels 0,1,2] [--simplified-only]
```

The copy lands in `<project_output_root>/admin_shapefiles/<release>/`; `paths.yaml` must list
`project_output_root` under `write` (the one-off grant of STANDARDS §Path safety). Existing
files are never overwritten.

## Population-model raking inputs

Under `raking_root` (the population model's `admin-inputs/raking` directory):

- `load_raking_shapes(root, "gbd_2023")`: the most-detailed polygons of a GBD hierarchy,
  taken from `shapes_gbd_2023.parquet` and filtered to the locations flagged `most_detailed`
  in `population_gbd_2023.parquet`. The flag is expected to be the same in every year of that
  file and the loader checks it; when it differs, `year_id=` names the year to use, so no
  year is ever picked silently.
- `load_raking_shapes(root, "lsae_1285")`: `gbd-inputs/shapes_lsae_1285_a2.parquet` as is.
- `fhs_*` hierarchies have no shapes of their own; use the GBD parent.
- `bounds=` keeps the polygons intersecting a box in the file's CRS: the read itself is
  filtered when the file carries a GeoParquet covering column (otherwise the whole file is
  read first), and an exact intersects test applies either way. With `bounds` an empty result
  is possible; without it, it is an error.
- `load_hierarchy(root, name)`: `location_id`, `parent_id`, `level` as int32.

`location_id` is int32 everywhere, which is also what `rasterize_zones` needs for a zone id.
