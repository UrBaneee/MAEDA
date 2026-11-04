"""Low-cardinality text columns: the planner sees every value, and a filter
naming a value that does not exist fails instead of matching nothing
(附录 FO: "西区" reached the filter verbatim and returned 0 rows)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.agents.analysis_agent import _column_manifest
from src.tools.data_connector import CATEGORY_LIMIT, extract_schema
from src.tools.stats_tool import pandas_filter


def _df():
    return pd.DataFrame({
        "date": ["2022-06-01", "2022-06-02", "2023-01-05", "2023-01-06"],
        "region": ["West", "East", "West", "North"],
        "rep_id": [f"R{i}" for i in range(4)],
        "revenue": [1.0, 2.0, 3.0, 4.0],
    })


def test_schema_lists_categories_for_small_text_columns_only():
    cols = {c.name: c for c in extract_schema(_df(), "csv", "x.csv").columns}
    assert cols["region"].categories == ["East", "North", "West"]
    assert cols["revenue"].categories is None
    assert cols["date"].categories is None
    assert cols["region"].to_dict()["categories"] == ["East", "North", "West"]
    assert "categories" not in cols["revenue"].to_dict()


def test_schema_omits_categories_above_limit():
    df = pd.DataFrame({"sku": [f"S{i}" for i in range(CATEGORY_LIMIT + 1)]})
    assert extract_schema(df, "csv", "x.csv").columns[0].categories is None


def test_manifest_shows_values():
    source = extract_schema(_df(), "csv", "x.csv").to_source_dict()
    manifest = _column_manifest(source)
    assert "- region (object) values: East, North, West" in manifest
    assert "- revenue (float64)\n" in manifest + "\n"


@pytest.mark.parametrize("op,val", [("==", "西区"), ("!=", "Nishi District"), ("in", ["West", "华东"])])
def test_filter_unknown_category_value_raises_with_values(op, val):
    with pytest.raises(ValueError, match=r"not found in column 'region'.*'West'"):
        pandas_filter(_df(), [{"column": "region", "op": op, "value": val}])


def test_filter_known_value_and_other_columns_unchanged():
    assert pandas_filter(_df(), [{"column": "region", "op": "==", "value": "West"}])["row_count"] == 2
    assert pandas_filter(_df(), [{"column": "revenue", "op": ">", "value": 2}])["row_count"] == 2
    assert pandas_filter(_df(), [{"column": "region", "op": "contains", "value": "We"}])["row_count"] == 2


def test_filter_value_valid_in_full_data_but_absent_after_earlier_filter_is_allowed():
    out = pandas_filter(_df(), [
        {"column": "date", "op": ">=", "value": "2023-01-01"},
        {"column": "region", "op": "==", "value": "East"},
    ])
    assert out["row_count"] == 0
