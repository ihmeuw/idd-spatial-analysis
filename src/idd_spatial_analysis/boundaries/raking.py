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
from idd_spatial_analysis.tables import require_columns

if TYPE_CHECKING:
    from idd_spatial_analysis.frame.bounds import Bounds

_GBD = re.compile(r"^gbd_\d{4}$")
_LSAE = re.compile(r"^lsae_\d+$")
_FHS = re.compile(r"^fhs_\d{4}$")
_ID_DTYPE = "int32"
_YEAR_COLUMN = "year_id"
_MOST_DETAILED_COLUMN = "most_detailed"


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


def _most_detailed(population: pd.DataFrame, name: str, year_id: int | None) -> pd.DataFrame:
    """The ``location_id`` rows flagged most detailed, for ``year_id`` or, without one, for all years at once."""
    require_columns(population, [LOCATION_ID_COLUMN, _YEAR_COLUMN, _MOST_DETAILED_COLUMN], name)
    if year_id is not None:
        rows = population.loc[population[_YEAR_COLUMN] == year_id]
        if rows.empty:
            msg = f"{name} has no rows for year_id={year_id}"
            raise ValueError(msg)
    else:
        flagged = population.loc[population[_MOST_DETAILED_COLUMN] == 1]
        if flagged.empty:
            msg = f"{name} flags no location as most detailed"
            raise ValueError(msg)
        per_year = flagged.groupby(_YEAR_COLUMN)[LOCATION_ID_COLUMN].apply(frozenset)
        years = set(population[_YEAR_COLUMN].tolist())
        if set(per_year.index.tolist()) != years or len(set(per_year.tolist())) > 1:
            msg = (
                f"{name}: the set of most-detailed locations differs between years; "
                "pass year_id to pick the year"
            )
            raise ValueError(msg)
        rows = population
    detailed = rows.loc[rows[_MOST_DETAILED_COLUMN] == 1, [LOCATION_ID_COLUMN]].drop_duplicates()
    if detailed.empty:
        msg = f"{name} flags no location as most detailed"
        raise ValueError(msg)
    return detailed


def load_raking_shapes(
    raking_root: Path,
    hierarchy: str,
    *,
    bounds: Bounds | None = None,
    year_id: int | None = None,
) -> gpd.GeoDataFrame:
    """Most-detailed polygons of ``hierarchy`` with an ``int32`` ``location_id``.

    ``bounds`` (in the file's CRS) keeps the polygons intersecting a box: the read itself is
    filtered when the file carries a GeoParquet covering column, and an exact intersects test
    applies either way. With ``bounds`` an empty result means no polygon reaches the box;
    without it an empty result is an error.

    For ``gbd_*`` hierarchies the most-detailed locations come from the population file; the
    set is normally the same in every year, and the function checks that rather than picking a
    year. When the set differs between years (a flag changes, or a location appears in some
    years only), ``year_id`` must name the year to use.
    """
    _check_hierarchy_name(hierarchy)
    root = Path(raking_root)
    if _LSAE.match(hierarchy):
        shapes = _read_shapes(root / "gbd-inputs" / f"shapes_{hierarchy}_a2.parquet", bounds)
    elif _GBD.match(hierarchy):
        shapes = _read_shapes(root / f"shapes_{hierarchy}.parquet", bounds)
        population_name = f"population_{hierarchy}.parquet"
        population = pd.read_parquet(root / population_name)
        detailed = _most_detailed(population, population_name, year_id)
        if LOCATION_ID_COLUMN not in shapes.columns:
            msg = f"raking shapes for {hierarchy} lack {LOCATION_ID_COLUMN!r}"
            raise KeyError(msg)
        shapes = shapes.merge(detailed, on=LOCATION_ID_COLUMN, how="inner")
    else:
        msg = f"{hierarchy!r} has no shapes in the raking inputs; use its gbd parent"
        raise ValueError(msg)
    if LOCATION_ID_COLUMN not in shapes.columns:
        msg = f"raking shapes for {hierarchy} lack {LOCATION_ID_COLUMN!r}"
        raise KeyError(msg)
    if shapes.empty and bounds is None:
        msg = f"no shape of {hierarchy} is a most-detailed location"
        raise ValueError(msg)
    shapes[LOCATION_ID_COLUMN] = shapes[LOCATION_ID_COLUMN].astype(_ID_DTYPE)
    return shapes.reset_index(drop=True)


def load_hierarchy(raking_root: Path, hierarchy: str) -> pd.DataFrame:
    """``location_id``, ``parent_id`` and ``level`` of every location, as ``int32``."""
    _check_hierarchy_name(hierarchy)
    table = pd.read_parquet(Path(raking_root) / "gbd-inputs" / f"hierarchy_{hierarchy}.parquet")
    required = [LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN]
    require_columns(table, required, f"hierarchy {hierarchy}")
    out = table[required].astype(dict.fromkeys(required, _ID_DTYPE))
    return out.reset_index(drop=True)
