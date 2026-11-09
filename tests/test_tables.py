"""Table checks."""

from __future__ import annotations

import pandas as pd
import pytest

from idd_spatial_analysis.tables import require_columns


def test_require_columns() -> None:
    table = pd.DataFrame({"a": [1]})
    require_columns(table, ["a"], "t")
    with pytest.raises(KeyError, match="t lacks columns"):
        require_columns(table, ["a", "b"], "t")
