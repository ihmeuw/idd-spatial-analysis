"""The ``idd-spatial`` command line."""

from __future__ import annotations

from pathlib import Path

import click

from idd_spatial_analysis import __version__
from idd_spatial_analysis.boundaries.lbd import AdminRelease, copy_release
from idd_spatial_analysis.constants import ADMIN_SHAPEFILES_DIR
from idd_spatial_analysis.paths import PathsFileMissingError, load_paths

_OUTPUT_KEY = "project_output_root"
_SOURCE_KEY = "admin_shapefile_root"


@click.group()
@click.version_option(__version__, prog_name="idd-spatial")
def main() -> None:
    """Spatial tools for IDD repos: block frames, admin boundaries, rasters, zonal statistics."""


@main.command("materialize-admin-release")
@click.option("--release", required=True, help="Release directory name, for example 2024_07_29.")
@click.option(
    "--levels", default="0,1,2", show_default=True, help="Comma-separated admin levels to copy."
)
@click.option("--simplified-only", is_flag=True, help="Copy only the _simplified shapefiles.")
@click.option(
    "--paths-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="paths.yaml to read; defaults to the repo's.",
)
def materialize_admin_release(
    *, release: str, levels: str, simplified_only: bool, paths_file: Path | None
) -> None:
    """Copy an LBD admin release to <project_output_root>/admin_shapefiles/<release>.

    For compute nodes that cannot see the archive mount. The copy is byte-identical and loads
    through the same AdminRelease(root, release) call with root pointing at the copy.
    paths.yaml must list project_output_root under write.
    """
    try:
        paths = load_paths(paths_file)
    except (PathsFileMissingError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    if _OUTPUT_KEY not in paths["write"]:
        msg = f"{_OUTPUT_KEY} is not in paths.yaml's write list; add it to allow this copy"
        raise click.ClickException(msg)
    try:
        level_list = [int(part) for part in levels.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"--levels must be comma-separated integers, got {levels!r}"
        raise click.ClickException(msg) from exc
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
