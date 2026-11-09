"""The conventions module holds fragments and names only, never an absolute path."""

from __future__ import annotations

import inspect
import re

from idd_spatial_analysis import constants

# Assembled from fragments so this file never contains a literal cluster path itself (the
# pre-commit guard greps committed files for the same roots).
_ROOTS = ("ihme", "snfs1", "home/j", "mnt/share", "mnt/team")
ABSOLUTE_PATH = re.compile("(" + "|".join(f"/{root}/" for root in _ROOTS) + r"|^[A-Z]:\\)")


def test_no_absolute_paths_in_constants() -> None:
    source = inspect.getsource(constants)
    assert ABSOLUTE_PATH.search(source) is None


def test_filename_templates_format() -> None:
    assert constants.LBD_ADMIN_FILENAME.format(level=2, suffix="") == "lbd_standard_admin_2.shp"
    assert (
        constants.LBD_ADMIN_FILENAME.format(level=0, suffix=constants.LBD_SIMPLIFIED_SUFFIX)
        == "lbd_standard_admin_0_simplified.shp"
    )
    assert constants.ADM_CODE_COLUMN.format(level=1) == "ADM1_CODE"
    assert constants.TILE_INDEX_RESOLUTION_DIR.format(resolution=100) == "100m"


def test_zone_nodata_is_outside_every_zone() -> None:
    assert constants.ZONE_NODATA == 0
    assert constants.ZONE_DTYPE == "int32"
