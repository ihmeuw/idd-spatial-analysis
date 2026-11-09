"""LBD standard admin shapefiles, one dated release at a time.

A release is a directory ``<root>/<YYYY_MM_DD>/`` holding ``lbd_standard_admin_{level}.shp``
and its ``_simplified`` twin for levels 0 to 2. Loading pins the release explicitly so that a
project never drifts to a newer geometry by accident.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from idd_spatial_analysis.constants import (
    ADM_CODE_COLUMN,
    ADM_NAME_COLUMN,
    LBD_ADMIN_FILENAME,
    LBD_DISPUTED_FILENAME,
    LBD_MAX_LEVEL,
    LBD_SIMPLIFIED_SUFFIX,
    LOC_ID_COLUMN,
)
from idd_spatial_analysis.crs import crs_equal

if TYPE_CHECKING:
    from collections.abc import Iterable

_SHAPEFILE_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml", ".qix")
_CODE_DTYPE = "int32"


@dataclass(frozen=True)
class AdminRelease:
    """One dated LBD release under ``root``; ``simplified`` selects the generalised geometry."""

    root: Path
    release: str
    simplified: bool = True

    @property
    def directory(self) -> Path:
        return Path(self.root) / self.release

    @property
    def suffix(self) -> str:
        return LBD_SIMPLIFIED_SUFFIX if self.simplified else ""

    def path(self, level: int) -> Path:
        """The shapefile for an admin level, whether or not it exists."""
        if not 0 <= level <= LBD_MAX_LEVEL:
            msg = f"admin level must be 0..{LBD_MAX_LEVEL}, got {level}"
            raise ValueError(msg)
        return self.directory / LBD_ADMIN_FILENAME.format(level=level, suffix=self.suffix)

    def disputed_path(self) -> Path:
        return self.directory / LBD_DISPUTED_FILENAME.format(suffix=self.suffix)

    def require(self, level: int) -> Path:
        """The shapefile for ``level``, or ``FileNotFoundError`` naming the release directory."""
        path = self.path(level)
        if not path.exists():
            msg = f"{path} not found (release {self.release!r} under {self.root})"
            raise FileNotFoundError(msg)
        return path


def _code_columns(level: int) -> list[str]:
    return [ADM_CODE_COLUMN.format(level=lvl) for lvl in range(level + 1)]


def load_admin(
    release: AdminRelease,
    level: int,
    *,
    countries: Iterable[int] | None = None,
    columns: Iterable[str] | None = None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Admin units of one level from a pinned release, with integer codes.

    ``countries`` restricts the read to those ``ADM0_CODE`` values before anything is loaded.
    The code columns of every level up to ``level`` and the level's name column are always
    read; ``columns`` adds more (``loc_id``, say). ``crs`` reprojects the result.
    """
    path = release.require(level)
    wanted = [*_code_columns(level), ADM_NAME_COLUMN.format(level=level)]
    if columns is not None:
        wanted = list(dict.fromkeys([*wanted, *columns]))
    where = None
    if countries is not None:
        codes = sorted({int(code) for code in countries})
        if not codes:
            msg = "countries is empty"
            raise ValueError(msg)
        where = f"{ADM_CODE_COLUMN.format(level=0)} IN ({', '.join(str(c) for c in codes)})"
    gdf = gpd.read_file(path, columns=wanted, where=where)
    missing = [column for column in wanted if column not in gdf.columns]
    if missing:
        msg = f"{path} lacks columns {missing}"
        raise KeyError(msg)
    if gdf.empty:
        msg = f"{path} returned no rows for countries={countries}"
        raise ValueError(msg)
    for column in _code_columns(level):
        gdf[column] = gdf[column].astype(_CODE_DTYPE)
    if LOC_ID_COLUMN in gdf.columns:
        gdf[LOC_ID_COLUMN] = pd.array(
            np.where(pd.isna(gdf[LOC_ID_COLUMN]), None, gdf[LOC_ID_COLUMN]), dtype="Int32"
        )
    if crs is not None and not crs_equal(gdf.crs, crs):
        gdf = gdf.to_crs(crs)
    return gdf.reset_index(drop=True)


def copy_release(
    source: AdminRelease,
    destination_root: Path,
    *,
    levels: Iterable[int] = (0, 1, 2),
    both_variants: bool = True,
) -> AdminRelease:
    """Copy a release's shapefiles (every sidecar) under ``destination_root/<release>/``.

    For the case where the source mount is absent on compute nodes; the copy is byte-identical
    and loads with the same ``AdminRelease`` API. Existing files are not overwritten.
    """
    variants = (True, False) if both_variants else (source.simplified,)
    target_dir = Path(destination_root) / source.release
    target_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        for simplified in variants:
            variant = AdminRelease(source.root, source.release, simplified=simplified)
            stem = variant.require(level).with_suffix("")
            for sidecar in _SHAPEFILE_SIDECARS:
                file = Path(f"{stem}{sidecar}")
                if not file.exists():
                    continue
                target = target_dir / file.name
                if target.exists():
                    msg = f"{target} exists; refusing to overwrite"
                    raise FileExistsError(msg)
                shutil.copy2(file, target)
    return AdminRelease(Path(destination_root), source.release, simplified=source.simplified)
