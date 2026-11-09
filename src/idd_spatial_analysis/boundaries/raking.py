"""Population-model raking inputs: most-detailed shapes and location hierarchies.

Layout under ``raking_root`` (the population model's ``admin-inputs/raking``):
``shapes_<gbd_YYYY>.parquet`` with ``population_<gbd_YYYY>.parquet`` marking most-detailed
locations, ``gbd-inputs/shapes_<lsae_N>_a2.parquet`` already at admin 2, and
``gbd-inputs/hierarchy_<name>.parquet`` tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import box

from idd_spatial_analysis.constants import LEVEL_COLUMN, LOCATION_ID_COLUMN, PARENT_ID_COLUMN

if TYPE_CHECKING:
    from idd_spatial_analysis.frame.bounds import Bounds

_GBD = re.compile(r"^gbd_\d{4}$")
_LSAE = re.compile(r"^lsae_\d+$")
_FHS = re.compile(r"^fhs_\d{4}$")
_ID_DTYPE = "int32"


def _check_hierarchy_name(hierarchy: str) -> None:
    if not (_GBD.match(hierarchy) or _LSAE.match(hierarchy) or _FHS.match(hierarchy)):
        msg = f"unknown hierarchy name {hierarchy!r} (expected gbd_YYYY, lsae_N or fhs_YYYY)"
        raise ValueError(msg)


def _has_covering_bbox(path: Path) -> bool:
    """True when the GeoParquet file carries a bbox covering column (GeoParquet 1.1)."""
    metadata = pq.read_metadata(path).metadata or {}
    geo = metadata.get(b"geo")
    if geo is None:
        return False
    parsed = json.loads(geo)
    primary = parsed.get("primary_column", "geometry")
    return "covering" in parsed.get("columns", {}).get(primary, {})


def _read_shapes(path: Path, bounds: Bounds | None) -> gpd.GeoDataFrame:
    bbox = bounds if bounds is not None and _has_covering_bbox(path) else None
    shapes = gpd.read_parquet(path, bbox=bbox)
    if bounds is not None:
        shapes = shapes.loc[shapes.intersects(box(*bounds))]
    return shapes


def load_raking_shapes(
    raking_root: Path, hierarchy: str, *, bounds: Bounds | None = None
) -> gpd.GeoDataFrame:
    """Most-detailed polygons of ``hierarchy`` with an ``int32`` ``location_id``.

    ``bounds`` (in the file's CRS) filters at read time so a block task never loads the globe.
    """
    _check_hierarchy_name(hierarchy)
    root = Path(raking_root)
    if _LSAE.match(hierarchy):
        shapes = _read_shapes(root / "gbd-inputs" / f"shapes_{hierarchy}_a2.parquet", bounds)
    elif _GBD.match(hierarchy):
        shapes = _read_shapes(root / f"shapes_{hierarchy}.parquet", bounds)
        population = pd.read_parquet(root / f"population_{hierarchy}.parquet")
        latest = population.loc[population["year_id"] == population["year_id"].max()]
        detailed = latest.loc[latest["most_detailed"] == 1, [LOCATION_ID_COLUMN]].drop_duplicates()
        shapes = shapes.merge(detailed, on=LOCATION_ID_COLUMN, how="inner")
    else:
        msg = f"{hierarchy!r} has no shapes in the raking inputs; use its gbd parent"
        raise ValueError(msg)
    if LOCATION_ID_COLUMN not in shapes.columns:
        msg = f"raking shapes for {hierarchy} lack {LOCATION_ID_COLUMN!r}"
        raise KeyError(msg)
    shapes[LOCATION_ID_COLUMN] = shapes[LOCATION_ID_COLUMN].astype(_ID_DTYPE)
    return shapes.reset_index(drop=True)


def load_hierarchy(raking_root: Path, hierarchy: str) -> pd.DataFrame:
    """``location_id``, ``parent_id`` and ``level`` of every location, as ``int32``."""
    _check_hierarchy_name(hierarchy)
    table = pd.read_parquet(Path(raking_root) / "gbd-inputs" / f"hierarchy_{hierarchy}.parquet")
    required = [LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN]
    missing = [column for column in required if column not in table.columns]
    if missing:
        msg = f"hierarchy {hierarchy} lacks columns {missing}"
        raise KeyError(msg)
    out = table[required].astype(dict.fromkeys(required, _ID_DTYPE))
    return out.reset_index(drop=True)
