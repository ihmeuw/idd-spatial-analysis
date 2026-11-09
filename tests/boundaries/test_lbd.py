"""LBD releases: pinning, subsetting, dtypes, copying."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from idd_spatial_analysis.boundaries import AdminRelease, copy_release, load_admin
from idd_spatial_analysis.constants import WORLD_CYLINDRICAL_EQUAL_AREA

if TYPE_CHECKING:
    from pathlib import Path


def test_release_paths(release: AdminRelease) -> None:
    assert release.path(2).name == "lbd_standard_admin_2.shp"
    simplified = AdminRelease(release.root, release.release, simplified=True)
    assert simplified.path(0).name == "lbd_standard_admin_0_simplified.shp"
    assert simplified.disputed_path().name == "lbd_disputed_mask_simplified.shp"
    with pytest.raises(ValueError, match="admin level"):
        release.path(3)


def test_missing_release_names_the_directory(tmp_path: Path) -> None:
    missing = AdminRelease(tmp_path, "2099_01_01")
    with pytest.raises(FileNotFoundError, match="2099_01_01"):
        load_admin(missing, 0)


def test_load_admin_levels_codes_and_loc_id(release: AdminRelease) -> None:
    adm2 = load_admin(release, 2)
    assert len(adm2) == 4
    assert str(adm2["ADM0_CODE"].dtype) == "int32"
    assert str(adm2["ADM2_CODE"].dtype) == "int32"
    assert set(adm2.columns) >= {"ADM0_CODE", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME", "geometry"}
    adm0 = load_admin(release, 0, columns=["loc_id"])
    assert str(adm0["loc_id"].dtype) == "Int32"
    assert adm0["loc_id"].isna().sum() == 1


def test_country_subset_happens_at_read_time(release: AdminRelease) -> None:
    only_a = load_admin(release, 2, countries=[1])
    assert sorted(only_a["ADM2_CODE"]) == [111, 112, 121]
    with pytest.raises(ValueError, match="no rows"):
        load_admin(release, 1, countries=[99])
    with pytest.raises(ValueError, match="empty"):
        load_admin(release, 1, countries=[])


def test_reprojection_on_request(release: AdminRelease) -> None:
    adm1 = load_admin(release, 1, crs=WORLD_CYLINDRICAL_EQUAL_AREA)
    assert adm1.crs is not None
    assert adm1.crs.to_string().endswith("54034")


def test_copy_release_is_loadable_and_refuses_overwrite(
    release: AdminRelease, tmp_path: Path
) -> None:
    copied = copy_release(release, tmp_path / "project_copy")
    assert copied.directory == tmp_path / "project_copy" / "2024_07_29"
    assert sorted(load_admin(copied, 2)["ADM2_CODE"]) == [111, 112, 121, 211]
    simplified = AdminRelease(copied.root, copied.release, simplified=True)
    assert load_admin(simplified, 0).shape[0] == 2
    with pytest.raises(FileExistsError, match="refusing"):
        copy_release(release, tmp_path / "project_copy")
