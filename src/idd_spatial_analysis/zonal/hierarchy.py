"""Roll most-detailed zone sums up a location hierarchy: sum first, divide last."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Iterable

_SUM_COLUMNS = [WEIGHTED_SUM_COLUMN, WEIGHT_SUM_COLUMN]
_ID_DTYPE = "int32"
_SHOW = 10


def _preview(values: set[int]) -> str:
    shown = sorted(values)[:_SHOW]
    more = "" if len(values) <= _SHOW else f" (+{len(values) - _SHOW} more)"
    return f"{shown}{more}"


def _check_tree(tree: pd.DataFrame) -> tuple[int, set[int], set[int]]:
    """The root level, the parent ids and the structural leaf ids of a consistent hierarchy."""
    if tree[LOCATION_ID_COLUMN].duplicated().any():
        msg = "hierarchy lists some location_id with more than one parent or level"
        raise ValueError(msg)
    root_level = int(tree[LEVEL_COLUMN].min())
    ids = set(tree[LOCATION_ID_COLUMN].tolist())
    parents = set(tree.loc[tree[LEVEL_COLUMN] > root_level, PARENT_ID_COLUMN].tolist())
    if parents - ids:
        msg = f"hierarchy names parents it does not list: {_preview(parents - ids)}"
        raise ValueError(msg)
    return root_level, parents, ids - parents


def _check_no_double_counting(
    zones: set[int], parent_of: dict[int, int], root_level: int, level_of: dict[int, int]
) -> None:
    """No zone with sums may have an ancestor that also has sums."""
    for zone in zones:
        ancestor = parent_of[zone]
        while level_of.get(zone, root_level) > root_level and ancestor in level_of:
            if ancestor in zones:
                msg = (
                    f"zone {zone} and its ancestor {ancestor} both have sums; the ancestor "
                    "would count the same people twice"
                )
                raise ValueError(msg)
            if level_of[ancestor] <= root_level:
                break
            ancestor = parent_of[ancestor]


def _check_sums(sums: pd.DataFrame, ids: set[int], extra: list[str]) -> set[int]:
    """One row per zone (and slice); every zone a known location. Returns the zone ids."""
    keys = [ZONE_COLUMN, *extra]
    repeated = sums.duplicated(keys)
    if repeated.any():
        msg = (
            f"sums has more than one row per {keys}: zones "
            f"{_preview(set(sums.loc[repeated, ZONE_COLUMN].tolist()))}; "
            "add per-block partials with collate_blocks first"
        )
        raise ValueError(msg)
    zones = set(sums[ZONE_COLUMN].tolist())
    if zones - ids:
        msg = f"{len(zones - ids)} zones are not in the hierarchy: {_preview(zones - ids)}"
        raise ValueError(msg)
    return zones


def _check_complete(sums: pd.DataFrame, leaves: set[int], extra: list[str]) -> None:
    """Every expected location has a row, in every slice when there are slices."""
    groups = sums.groupby(SLICE_COLUMN) if extra else [(None, sums)]
    for label, part in groups:
        missing = leaves - set(part[ZONE_COLUMN].tolist())
        if missing:
            where = "" if label is None else f" in slice {label}"
            msg = f"{len(missing)} most-detailed locations have no sums{where}: {_preview(missing)}"
            raise ValueError(msg)


def rollup_hierarchy(
    sums: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    leaves: Iterable[int] | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Sums and means at every level of ``hierarchy`` from most-detailed ``weighted_sums``.

    ``sums`` has ``zone`` (a ``location_id``), ``weighted_sum``, ``weight_sum`` and
    optionally ``slice``, with exactly one row per zone (and slice): add per-block partials
    with ``collate_blocks`` first. ``hierarchy`` has ``location_id``, ``parent_id``, ``level``
    with each parent one level up; the roots are the rows at the lowest level present and
    their ``parent_id`` is ignored.

    ``leaves`` are the locations expected to carry sums: the most-detailed set the zone mask
    was built from (``load_raking_shapes(...)["location_id"]``), which may sit at mixed
    levels and above the hierarchy's structural leaves, as the population model's raking set
    does. Without it, the structural leaves are expected. Every expected location must be
    present in every slice unless ``require_complete=False``. A zone whose ancestor also has
    sums is refused, since the ancestor would count the same people twice.

    Sums are added up the tree and ``mean = weighted_sum / weight_sum`` is taken once at the
    end (NaN where the weight sum is zero).
    """
    require_columns(sums, [ZONE_COLUMN, *_SUM_COLUMNS], "sums")
    require_columns(hierarchy, [LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN], "hierarchy")
    extra = [SLICE_COLUMN] if SLICE_COLUMN in sums.columns else []
    tree = hierarchy[[LOCATION_ID_COLUMN, PARENT_ID_COLUMN, LEVEL_COLUMN]].drop_duplicates()
    root_level, parents, structural_leaves = _check_tree(tree)
    ids = parents | structural_leaves
    expected = structural_leaves if leaves is None else {int(leaf) for leaf in leaves}
    if expected - ids:
        msg = f"{len(expected - ids)} expected leaves are not in the hierarchy: {_preview(expected - ids)}"
        raise ValueError(msg)
    zones = _check_sums(sums, ids, extra)
    level_of = dict(zip(tree[LOCATION_ID_COLUMN], tree[LEVEL_COLUMN], strict=True))
    parent_of = dict(zip(tree[LOCATION_ID_COLUMN], tree[PARENT_ID_COLUMN], strict=True))
    _check_no_double_counting(zones, parent_of, root_level, level_of)
    if require_complete:
        _check_complete(sums, expected, extra)

    leaf_rows = sums.rename(columns={ZONE_COLUMN: LOCATION_ID_COLUMN})[
        [LOCATION_ID_COLUMN, *extra, *_SUM_COLUMNS]
    ].copy()
    leaf_rows[LEVEL_COLUMN] = leaf_rows[LOCATION_ID_COLUMN].map(level_of)
    collected = [leaf_rows]
    for level in range(int(tree[LEVEL_COLUMN].max()), root_level, -1):
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
