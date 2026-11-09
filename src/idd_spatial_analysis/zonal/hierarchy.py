"""Roll most-detailed zone sums up a location hierarchy: sum first, divide last."""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_spatial_analysis.constants import (
    LEVEL_COLUMN,
    LOCATION_ID_COLUMN,
    MEAN_COLUMN,
    PARENT_ID_COLUMN,
    SLICE_COLUMN,
    WEIGHT_SUM_COLUMN,
    WEIGHTED_SUM_COLUMN,
    ZONE_COLUMN,
)
from idd_spatial_analysis.tables import require_columns

_SUM_COLUMNS = [WEIGHTED_SUM_COLUMN, WEIGHT_SUM_COLUMN]
_ID_DTYPE = "int32"
_SHOW = 10


def _preview(values: set[int]) -> str:
    shown = sorted(values)[:_SHOW]
    more = "" if len(values) <= _SHOW else f" (+{len(values) - _SHOW} more)"
    return f"{shown}{more}"


def rollup_hierarchy(
    sums: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Sums and means at every level of ``hierarchy`` from most-detailed ``weighted_sums``.

    ``sums`` has ``zone`` (a most-detailed ``location_id``), ``weighted_sum``, ``weight_sum``
    and optionally ``slice``; ``hierarchy`` has ``location_id``, ``parent_id``, ``level`` with
    each parent one level up. Sums are added up the tree and ``mean = weighted_sum /
    weight_sum`` is taken once at the end (NaN where the weight sum is zero). Every
    most-detailed location must be present unless ``require_complete=False``; zones that are
    not most-detailed are refused because they would be counted twice.
    """
    require_columns(sums, [ZONE_COLUMN, *_SUM_COLUMNS], "sums")
    require_columns(hierarchy, [LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN], "hierarchy")
    extra = [SLICE_COLUMN] if SLICE_COLUMN in sums.columns else []
    tree = hierarchy[[LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN]].drop_duplicates()
    if tree[LOCATION_ID_COLUMN].duplicated().any():
        msg = "hierarchy lists some location_id with more than one parent or level"
        raise ValueError(msg)
    ids = set(tree[LOCATION_ID_COLUMN].tolist())
    parents = set(tree.loc[tree[LEVEL_COLUMN] > 0, PARENT_ID_COLUMN].tolist())
    if parents - ids:
        msg = f"hierarchy names parents it does not list: {_preview(parents - ids)}"
        raise ValueError(msg)
    leaves = ids - parents
    zones = set(sums[ZONE_COLUMN].tolist())
    if zones - ids:
        msg = f"{len(zones - ids)} zones are not in the hierarchy: {_preview(zones - ids)}"
        raise ValueError(msg)
    if zones & parents:
        msg = f"sums must be at most-detailed locations; these are parents: {_preview(zones & parents)}"
        raise ValueError(msg)
    if require_complete and leaves - zones:
        msg = f"{len(leaves - zones)} most-detailed locations have no sums: {_preview(leaves - zones)}"
        raise ValueError(msg)

    level_of = tree.set_index(LOCATION_ID_COLUMN)[LEVEL_COLUMN]
    parent_of = tree.set_index(LOCATION_ID_COLUMN)[PARENT_ID_COLUMN]
    leaf_rows = sums.rename(columns={ZONE_COLUMN: LOCATION_ID_COLUMN})[
        [LOCATION_ID_COLUMN, *extra, *_SUM_COLUMNS]
    ].copy()
    leaf_rows[LEVEL_COLUMN] = leaf_rows[LOCATION_ID_COLUMN].map(level_of)
    collected = [leaf_rows]
    for level in range(int(tree[LEVEL_COLUMN].max()), 0, -1):
        at_level = pd.concat([rows.loc[rows[LEVEL_COLUMN] == level] for rows in collected])
        if at_level.empty:
            continue
        upward = at_level.assign(
            **{LOCATION_ID_COLUMN: at_level[LOCATION_ID_COLUMN].map(parent_of)}
        )
        aggregated = upward.groupby([LOCATION_ID_COLUMN, *extra], as_index=False)[
            _SUM_COLUMNS
        ].sum()
        aggregated[LEVEL_COLUMN] = aggregated[LOCATION_ID_COLUMN].map(level_of)
        if (aggregated[LEVEL_COLUMN] != level - 1).any():
            msg = f"a level-{level} location's parent is not at level {level - 1}"
            raise ValueError(msg)
        collected.append(aggregated)
    out = pd.concat(collected, ignore_index=True)
    out[MEAN_COLUMN] = out[WEIGHTED_SUM_COLUMN] / out[WEIGHT_SUM_COLUMN].where(
        out[WEIGHT_SUM_COLUMN] > 0, np.nan
    )
    for column in (LOCATION_ID_COLUMN, LEVEL_COLUMN, *extra):
        out[column] = out[column].astype(_ID_DTYPE)
    order = [LEVEL_COLUMN, LOCATION_ID_COLUMN, *extra]
    return out.sort_values(order).reset_index(drop=True)[
        [LOCATION_ID_COLUMN, *extra, LEVEL_COLUMN, *_SUM_COLUMNS, MEAN_COLUMN]
    ]
