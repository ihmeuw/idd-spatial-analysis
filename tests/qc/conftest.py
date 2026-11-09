"""Fixtures for the cluster-data checks: everything skips unless paths.yaml points at real data.

Which release, hierarchy, block and population raster to check come from environment
variables so that no cluster path or project choice is written into the repo:

- ``IDD_SPATIAL_QC_RELEASE``: an LBD release name (default: the newest directory under
  ``admin_shapefile_root``; the QC checks a release's schema, projects pin their own).
- ``IDD_SPATIAL_QC_HIERARCHY``: a raking hierarchy name (default ``gbd_2023``).
- ``IDD_SPATIAL_QC_RESOLUTION``: tile index resolution in metres (default ``100``).
- ``IDD_SPATIAL_QC_POPULATION_TIF``: a population GeoTIFF on the tile index frame (no default).
- ``IDD_SPATIAL_QC_BLOCK_KEY``: the block to check (default: the first block the hierarchy's
  shapes intersect).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from idd_spatial_analysis.boundaries import AdminRelease
from idd_spatial_analysis.constants import TILE_INDEX_FILENAME, TILE_INDEX_RESOLUTION_DIR
from idd_spatial_analysis.paths import PathsFileMissingError, load_paths

pytestmark = pytest.mark.qc


@pytest.fixture(scope="session")
def paths() -> dict[str, Path | list[str]]:
    try:
        return load_paths()
    except PathsFileMissingError:
        pytest.skip("paths.yaml absent: the qc checks need cluster data")


def _existing(root: Path, what: str) -> Path:
    if not root.exists():
        pytest.skip(f"{what} {root} is not reachable from this node")
    return root


@pytest.fixture(scope="session")
def admin_release(paths: dict[str, Path | list[str]]) -> AdminRelease:
    root = _existing(Path(paths["admin_shapefile_root"]), "admin_shapefile_root")
    name = os.environ.get("IDD_SPATIAL_QC_RELEASE")
    if name is None:
        releases = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not releases:
            pytest.skip(f"no release directories under {root}")
        name = releases[-1]
    return AdminRelease(root, name, simplified=True)


@pytest.fixture(scope="session")
def hierarchy_name() -> str:
    return os.environ.get("IDD_SPATIAL_QC_HIERARCHY", "gbd_2023")


@pytest.fixture(scope="session")
def raking_root(paths: dict[str, Path | list[str]]) -> Path:
    return _existing(Path(paths["raking_root"]), "raking_root")


@pytest.fixture(scope="session")
def tile_index_path(paths: dict[str, Path | list[str]]) -> Path:
    resolution = os.environ.get("IDD_SPATIAL_QC_RESOLUTION", "100")
    root = _existing(Path(paths["tile_index_root"]), "tile_index_root")
    path = root / TILE_INDEX_RESOLUTION_DIR.format(resolution=resolution) / TILE_INDEX_FILENAME
    return _existing(path, "tile index")


@pytest.fixture(scope="session")
def population_tif() -> Path:
    value = os.environ.get("IDD_SPATIAL_QC_POPULATION_TIF")
    if value is None:
        pytest.skip("IDD_SPATIAL_QC_POPULATION_TIF not set")
    return _existing(Path(value), "population raster")
