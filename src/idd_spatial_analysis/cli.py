"""The ``idd-spatial`` command line."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import click
from idd_tools.versions import scratch_dir, working_dir

from idd_spatial_analysis import __version__
from idd_spatial_analysis.boundaries.lbd import AdminRelease, copy_release
from idd_spatial_analysis.boundaries.raking import load_raking_shapes
from idd_spatial_analysis.constants import (
    ADMIN_SHAPEFILES_DIR,
    LOCATION_ID_COLUMN,
    PIXEL_COUNT_COLUMN,
    TILE_INDEX_FILENAME,
    TILE_INDEX_RESOLUTION_DIR,
    ZONE_LOOKUP_NODE,
)
from idd_spatial_analysis.frame.blocks import block_grid, blocks_intersecting, load_tile_index
from idd_spatial_analysis.lookup.zone_lookup import build_zone_lookup_block, collect_zone_lookup
from idd_spatial_analysis.paths import PathsFileMissingError, load_paths

_OUTPUT_KEY = "project_output_root"
_SOURCE_KEY = "admin_shapefile_root"
_TILE_INDEX_KEY = "tile_index_root"
_RAKING_KEY = "raking_root"
_OUTPUT_UMASK = 0o002
"""Directories 775 and files 664 on the output node (STANDARDS §Output management)."""

_paths_file_option = click.option(
    "--paths-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="paths.yaml to read; defaults to the repo's.",
)
_hierarchy_option = click.option(
    "--hierarchy",
    required=True,
    help="Raking hierarchy whose most-detailed shapes are the zones, for example lsae_1285.",
)
_resolution_option = click.option(
    "--resolution",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Pixel size in metres; also names the tile index to use.",
)
_scratch_option = click.option(
    "--scratch",
    default=None,
    help="Write to the component's scratch/<label>/ instead of working/ (test runs).",
)


@click.group()
@click.version_option(__version__, prog_name="idd-spatial")
def main() -> None:
    """Spatial tools for IDD repos: block frames, admin boundaries, rasters, zonal statistics."""


def _writable_paths(paths_file: Path | None) -> dict[str, Any]:
    """``paths.yaml`` as a mapping, once ``project_output_root`` is known to be writable."""
    try:
        paths = load_paths(paths_file)
    except (PathsFileMissingError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    if _OUTPUT_KEY not in paths["write"]:
        msg = f"{_OUTPUT_KEY} is not in paths.yaml's write list; add it to allow this command"
        raise click.ClickException(msg)
    return paths


@main.command("materialize-admin-release")
@click.option("--release", required=True, help="Release directory name, for example 2024_07_29.")
@click.option(
    "--levels", default="0,1,2", show_default=True, help="Comma-separated admin levels to copy."
)
@click.option("--simplified-only", is_flag=True, help="Copy only the _simplified shapefiles.")
@_paths_file_option
def materialize_admin_release(
    *, release: str, levels: str, simplified_only: bool, paths_file: Path | None
) -> None:
    """Copy an LBD admin release to <project_output_root>/admin_shapefiles/<release>.

    For compute nodes that cannot see the archive mount. The copy is byte-identical and loads
    through the same AdminRelease(root, release) call with root pointing at the copy.
    paths.yaml must list project_output_root under write.
    """
    paths = _writable_paths(paths_file)
    try:
        level_list = [int(part) for part in levels.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"--levels must be comma-separated integers, got {levels!r}"
        raise click.ClickException(msg) from exc
    if not level_list:
        msg = f"--levels names no level, got {levels!r}"
        raise click.ClickException(msg)
    source = AdminRelease(paths[_SOURCE_KEY], release, simplified=True)
    try:
        copied = copy_release(
            source,
            paths[_OUTPUT_KEY] / ADMIN_SHAPEFILES_DIR,
            levels=level_list,
            both_variants=not simplified_only,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"copied release {release} (levels {level_list}) to {copied.directory}")


def _zone_lookup_directory(
    paths: dict[str, Any], hierarchy: str, resolution: int, scratch: str | None
) -> Path:
    """The slot of the lookup's versioned component a run writes to: working/ or scratch/<label>/."""
    node = paths[_OUTPUT_KEY] / ZONE_LOOKUP_NODE.format(zone_set=hierarchy, resolution=resolution)
    slot: Path = scratch_dir(node, scratch) if scratch is not None else working_dir(node)
    return slot


def _tile_index_path(paths: dict[str, Any], resolution: int) -> Path:
    directory: Path = paths[_TILE_INDEX_KEY] / TILE_INDEX_RESOLUTION_DIR.format(
        resolution=resolution
    )
    return directory / TILE_INDEX_FILENAME


@main.command("build-zone-lookup")
@_hierarchy_option
@_resolution_option
@click.option(
    "--block-key", "block_keys", multiple=True, help="Block to build; repeat for several."
)
@click.option(
    "--all-blocks",
    is_flag=True,
    help="Build every block the zones intersect, one after another.",
)
@_scratch_option
@_paths_file_option
def build_zone_lookup(  # noqa: PLR0913  (one parameter per command-line option)
    *,
    hierarchy: str,
    resolution: int,
    block_keys: tuple[str, ...],
    all_blocks: bool,
    scratch: str | None,
    paths_file: Path | None,
) -> None:
    """Write the zone lookup raster and zone table of each block named.

    The zones are the hierarchy's raking shapes with location_id as the zone id; every pixel
    of a block gets the id of the shape containing its centre, 0 where there is none. Output
    goes to the working/ slot of
    <project_output_root>/02-processed_data/zone_lookup/<hierarchy>/<resolution>m. Blocks are
    independent: run one process per block, or --all-blocks for a small zone set, then
    collect-zone-lookup.
    """
    os.umask(_OUTPUT_UMASK)
    if bool(block_keys) == all_blocks:
        msg = "name blocks with --block-key or pass --all-blocks, one or the other"
        raise click.ClickException(msg)
    paths = _writable_paths(paths_file)
    try:
        tiles = load_tile_index(_tile_index_path(paths, resolution))
        zones = load_raking_shapes(paths[_RAKING_KEY], hierarchy)
        keys = blocks_intersecting(tiles, zones) if all_blocks else list(block_keys)
        grids = {key: block_grid(tiles, key, float(resolution)) for key in keys}
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    directory = _zone_lookup_directory(paths, hierarchy, resolution, scratch)
    for key, grid in grids.items():
        started = time.monotonic()
        table = build_zone_lookup_block(zones, LOCATION_ID_COLUMN, grid, directory, key)
        click.echo(
            f"{key}: {len(table)} zones, {int(table[PIXEL_COUNT_COLUMN].sum())} zoned pixels "
            f"of {grid.height * grid.width}, {time.monotonic() - started:.1f} s"
        )
    click.echo(f"built {len(grids)} block(s) in {directory}")


@main.command("collect-zone-lookup")
@_hierarchy_option
@_resolution_option
@click.option(
    "--block-key",
    "block_keys",
    multiple=True,
    help="Collect only these blocks (test runs); default: every block the zones intersect.",
)
@_scratch_option
@_paths_file_option
def collect_zone_lookup_command(
    *,
    hierarchy: str,
    resolution: int,
    block_keys: tuple[str, ...],
    scratch: str | None,
    paths_file: Path | None,
) -> None:
    """Check every block is built, then write zones.parquet, zone_summary.parquet and build.yaml.

    Run after build-zone-lookup has written every block the hierarchy's shapes intersect.
    Fails, naming the blocks, when one is missing or when the directory holds blocks of
    another build.
    """
    os.umask(_OUTPUT_UMASK)
    paths = _writable_paths(paths_file)
    try:
        tiles = load_tile_index(_tile_index_path(paths, resolution))
        zones = load_raking_shapes(paths[_RAKING_KEY], hierarchy)
        keys = list(block_keys) if block_keys else blocks_intersecting(tiles, zones)
        directory = _zone_lookup_directory(paths, hierarchy, resolution, scratch)
        summary = collect_zone_lookup(
            directory,
            keys,
            zones[LOCATION_ID_COLUMN].tolist(),
            zone_set=hierarchy,
            zone_source=f"population-model raking shapes of {hierarchy}",
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    empty = int((summary[PIXEL_COUNT_COLUMN] == 0).sum())
    click.echo(
        f"collected {len(keys)} block(s) and {len(summary)} zones "
        f"({empty} without a pixel) in {directory}"
    )
