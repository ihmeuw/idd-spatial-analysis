# idd-spatial-analysis

Shared geospatial tools for IDD research code: block frames, admin boundaries,
antimeridian-safe geometry, raster windows and regridding, zonal statistics, and lookup
products built from them. It is a library first: it draws no maps (that is idd-figures) and
holds no scheduler code of its own (that is idd-tools). The one thing it produces is a small
set of shared lookup products in its own output node (`docs/lookup.md`), built with functions
and commands any consumer can run for a zone set of its own. The jobs that build the shared
products go through idd-tools' jobmon layer, which needs the cluster-only `jobmon` extra; no
consumer of the library needs it.

## What is here

| Subpackage | Purpose | Entry points |
|---|---|---|
| `frame` | The grid a block of pixels lives on and how blocks are chosen | `BlockGrid`, `grid_from_bounds`, `grid_from_coords`, `grid_from_raster`, `grids_match`, `load_tile_index`, `block_grid`, `blocks_intersecting`, `block_for_point`, `bbox_in_crs` |
| `boundaries` | Admin polygons from a pinned release, population-model raking shapes and hierarchies | `AdminRelease`, `load_admin`, `copy_release`, `load_raking_shapes`, `load_hierarchy` |
| `geometry` | Longitude conventions, bound strings, cell areas, points to zones, distances along the Earth | `to_lon_360`, `to_lon_180`, `split_antimeridian`, `parse_bound`, `cell_area_km2`, `grid_cell_areas_km2`, `points_to_zones`, `surface_distance`, `nearest_surface_distance` |
| `raster` | Windows, mosaics, regridding, zone masks, atomic GeoTIFF writes | `read_window`, `mosaic`, `reproject_to_grid`, `NearestIndexMap`, `resample_masked`, `rasterize_zones`, `write_geotiff_atomic`, `write_geotiff` |
| `zonal` | Population-weighted sums, sparse zone weights and their application to a coarse field, hierarchy roll-up, nearest facility (surface metres) | `weighted_sums`, `zone_weights`, `ZoneCellPairs`, `apply_zone_weights`, `rollup_hierarchy`, `collate_blocks`, `nearest_facility_distance`, `share_within` |
| `lookup` | Spatial assignments computed once and written to disk: the zone id of every pixel per block, and a weight raster summed onto pairs of zone and coarse cell | `build_zone_lookup_block`, `compute_zone_lookup_block`, `collect_zone_lookup`, `read_zone_lookup_block`, `load_zone_lookup_index`, `load_zone_lookup_summary`, `load_zone_lookup_metadata`, `build_zone_cell_weights_block`, `compute_zone_cell_weights_block`, `collect_zone_cell_weights`, `combine_zone_cell_weights`, `load_zone_cell_weights`, `ZoneCellWeights` |

A raster is the pair `(array, BlockGrid)`: a 2-D numpy array and the grid that places it.
Repos that keep their rasters as `rasterra.RasterArray` convert at the edge with
`idd_spatial_analysis.raster.adapters` (install the `rasterra` extra).

`docs/` holds five pages (`conventions`, `frame`, `boundaries`, `lookup`, `memory`) and every subpackage
has an import-only notebook under `notebooks/vignettes/` that runs on synthetic data in
seconds. Start with `docs/conventions.md`: it fixes the CRS strings, the north-up grid
convention, the surface-distance rule, nodata handling and the integer id columns every
function assumes.

## Install

Requirements: [uv](https://docs.astral.sh/uv/) 0.12 or newer, a uv-managed Python 3.12
(`uv python install 3.12`; interpreter downloads are deliberate, never implicit), and SSH
access to the `ihmeuw` GitHub organisation for the `idd-tools` dependency.

```bash
git clone git@github.com:ihmeuw/idd-spatial-analysis.git
cd idd-spatial-analysis
uv sync --inexact --extra rasterra   # base dependencies, the dev group and the rasterra adapters
cp paths.yaml.example paths.yaml     # then fill in the roots described below
.venv/bin/pre-commit install
```

Everything runs through `.venv/bin/python`; there is nothing to activate. Plain `uv sync`
rebuilds the environment exactly from `uv.lock` (vet it with `uv sync --dry-run` first).

To build the shared lookup products as cluster jobs, add `--extra jobmon`. That extra pulls
the IHME jobmon client, which resolves only from the IHME artifactory and so only on the IHME
network; `pyproject.toml` declares that index for it. Nothing else in the package needs it,
and CI does not install it.

## Paths and data

`paths.yaml` is gitignored and is the only file that may hold an absolute path. The package
reads it through `idd_spatial_analysis.paths.load_paths()`; every key in
`paths.yaml.example` is required.

| Key | What lives there | Layout expected |
|---|---|---|
| `admin_shapefile_root` | LBD standard admin shapefile releases | `<root>/<YYYY_MM_DD>/lbd_standard_admin_{0,1,2}.shp` and the `_simplified` twins |
| `tile_index_root` | The equal-area tile index per resolution | `<root>/<resolution>m/tile_index.parquet`, one row per tile, a `block_key` column |
| `population_grid_root` | Gridded population rasters | GeoTIFFs on the tile index frame |
| `raking_root` | Population-model raking inputs | `shapes_<gbd_YYYY>.parquet`, `population_<gbd_YYYY>.parquet`, `gbd-inputs/shapes_<lsae_N>_a2.parquet`, `gbd-inputs/hierarchy_<name>.parquet` |
| `project_output_root` | This repo's output node | Written only by the `idd-spatial` commands (the admin release copy and the lookup products under `02-processed_data/`); list the key under `write:` to allow them |

Loaders pin what they read: `AdminRelease(root, "YYYY_MM_DD")` names the release
explicitly, `load_raking_shapes(root, "gbd_2023")` names the hierarchy. Nothing picks "the
latest" for you; the one place a choice could hide, the year of the population file's
most-detailed flags, is checked to be the same in every year, and `year_id` must be given
when it is not.

## Use

```python
from idd_spatial_analysis.paths import load_paths
from idd_spatial_analysis.boundaries import AdminRelease, load_admin
from idd_spatial_analysis.frame import block_grid, blocks_intersecting, load_tile_index
from idd_spatial_analysis.raster import rasterize_zones, read_window
from idd_spatial_analysis.zonal import collate_blocks, weighted_sums

paths = load_paths()
release = AdminRelease(paths["admin_shapefile_root"], "2024_07_29")
adm2 = load_admin(release, 2, countries=[165])
population_tif = paths["population_grid_root"] / "<your population raster>.tif"
temperature_tif = paths["project_output_root"] / "<a value raster on the same frame>.tif"

tiles = load_tile_index(paths["tile_index_root"] / "100m" / "tile_index.parquet")
partials = []
for block_key in blocks_intersecting(tiles, adm2):
    grid = block_grid(tiles, block_key, resolution=100.0)
    population, _ = read_window(population_tif, grid.bounds, boundless=True)
    temperature, _ = read_window(temperature_tif, grid.bounds, boundless=True)
    mask, windows = rasterize_zones(adm2, "ADM2_CODE", grid)
    partials.append(weighted_sums(mask, population, temperature, nan_policy="drop_pixel"))
sums = collate_blocks(partials)
```

`boundless=True` pads a block at the raster's edge with NaN so every array has the block's
shape; `collate_blocks` adds the partial sums of zones that straddle blocks.

The loop above rasterizes the polygons in every block. When the same zones are used again
and again, build their lookup once and read each block's mask from disk instead:
`read_zone_lookup_block(directory, block_key)` replaces the `rasterize_zones` line
(`docs/lookup.md`). `nan_policy` defaults to `"keep_weight"`, under which a missing value
counts as zero; `"drop_pixel"` leaves such pixels out of both sums.

When the value is on a coarser grid than the blocks (a 0.1 degree climate field), the loop
over pixels is needed once per weight raster, not once per field: build the zone cell weights
(`docs/lookup.md`), then `load_zone_cell_weights(directory, label).apply(field, field_grid)`
returns the same two sums per zone for any field on that grid.

The command line has three commands; `idd-spatial --help` lists them and their options.

- `idd-spatial materialize-admin-release` copies a release under `project_output_root` for
  compute nodes that cannot see the archive mount.
- `idd-spatial build-zone-lookup` writes the zone lookup of the blocks it is given, or of
  every block a raking hierarchy's shapes intersect.
- `idd-spatial collect-zone-lookup` checks that every block is there and writes the index, the
  per-zone summary and the sidecar.

## Testing

```bash
.venv/bin/pytest                               # synthetic data, runs anywhere; tests/qc deselected
.venv/bin/pytest tests/qc -m qc                # real cluster data; skipped without paths.yaml
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/mypy src
```

Every public function has a unit test under `tests/`, mirroring the source layout. `tests/qc`
holds checks against the cluster data named in `paths.yaml` (schema, consistency, population
conservation on one real block) and is excluded from CI. CI (`.github/workflows`) runs ruff
check and format, mypy and pytest in the locked environment.

## Versions

Python 3.12 (uv-managed), uv 0.12. Dependencies are declared as floors in `pyproject.toml`;
the exact set, including the `idd-tools` commit from its `main` branch, is in `uv.lock`.
Bump idd-tools with `uv lock --upgrade-package idd-tools`.

## Requests from other repos

Drop a note in `inbox/` (see `inbox/README.md`); processed requests move to
`inbox/processed/`.
