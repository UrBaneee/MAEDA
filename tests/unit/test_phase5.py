"""
Phase 5 tests — Analysis Agent.
All LLM calls and heavy computation are deterministic / mocked.
Run with: pytest tests/unit/test_phase5.py -v
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agents.analysis_agent import (
    AnalysisAgent,
    AnalysisPlan,
    AnalysisStep,
    TOOL_REGISTRY,
    _aggregate,
    _column_manifest,
    _execution_order,
    _related_tables_manifest,
    _simplify_step,
)
from src.state.graph_state import initial_state
from src.tools.stats_tool import (
    analyze_time_series,
    anomaly_tool,
    compare_segments,
    compute_correlation,
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    pandas_derive,
    pandas_filter,
    pandas_groupby,
    pandas_pivot,
    pandas_tool,
    run_linear_regression,
    run_ttest,
    statistical_tool,
)
from src.tools.sql_tool import execute_sql, sql_tool


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "region":   ["North", "South", "East", "West", "North", "South", "East", "West"],
        "quarter":  ["Q1",    "Q1",    "Q1",   "Q1",   "Q2",    "Q2",    "Q2",   "Q2"],
        "revenue":  [1200.0,  850.0,   990.0,  1100.0, 1350.0,  920.0,  1050.0, 1200.0],
        "units":    [120,     85,      99,     110,    135,     92,     105,    120],
    })


@pytest.fixture
def time_df():
    return pd.DataFrame({
        "month":    ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        "revenue":  [1000.0,    1050.0,    1120.0,    1080.0,    1200.0,    1250.0],
    })


@pytest.fixture
def anomaly_df():
    np.random.seed(42)
    normal = np.random.normal(100, 10, 50).tolist()
    return pd.DataFrame({"value": normal + [500.0, -200.0]})  # 2 obvious outliers


@pytest.fixture
def multi_table_sqlite_db(tmp_path):
    db_path = tmp_path / "multi.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, revenue REAL)")
    conn.execute("CREATE TABLE customers (customer_id INTEGER, country TEXT)")
    conn.execute("INSERT INTO orders VALUES (1, 1, 100.0)")
    conn.execute("INSERT INTO customers VALUES (1, 'Canada')")
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


def _mock_llm(plan_data) -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content=json.dumps(plan_data),
        usage_metadata={"input_tokens": 100, "output_tokens": 80},
    ))
    return mock


# ─── 5.2 SQL query tool ───────────────────────────────────────────────────────

class TestSQLTool:
    def test_execute_sql_on_dataframe(self, sales_df):
        result = execute_sql("SELECT region, SUM(revenue) as total FROM data GROUP BY region",
                             df=sales_df)
        assert result["row_count"] == 4
        assert "total" in result["columns"]

    def test_execute_sql_filter(self, sales_df):
        result = execute_sql("SELECT * FROM data WHERE quarter = 'Q1'", df=sales_df)
        assert result["row_count"] == 4

    def test_execute_sql_order(self, sales_df):
        result = execute_sql("SELECT region, revenue FROM data ORDER BY revenue DESC LIMIT 3",
                             df=sales_df)
        assert result["row_count"] == 3
        revenues = [r["revenue"] for r in result["rows"]]
        assert revenues == sorted(revenues, reverse=True)

    def test_sql_tool_dispatcher(self, sales_df):
        result = sql_tool(sales_df,
                          {"query": "SELECT COUNT(*) as cnt FROM data"},
                          prior_results={})
        assert result["result"][0]["cnt"] == 8
        assert "result_summary" in result

    def test_sql_tool_missing_query_falls_back(self, sales_df):
        # Without a query, sql_tool constructs a basic SELECT and returns results
        result = sql_tool(sales_df, {}, prior_results={})
        assert "result" in result
        assert result.get("failed") is not True

    def test_sql_tool_null_limit_defaults_instead_of_breaking_query(self, sales_df):
        # {"limit": null} in JSON becomes Python None; must not literally
        # render as "LIMIT None" in the generated SQL.
        result = sql_tool(sales_df, {"limit": None}, prior_results={})
        assert "result" in result

    def test_sql_tool_group_by_without_select_columns_raises(self, sales_df):
        # A GROUP BY with nothing explicit to select can't safely default to
        # SELECT * — that lets SQLite return an arbitrary row per group
        # instead of an actual aggregate.
        with pytest.raises(ValueError, match="group_by.*without.*select_columns"):
            sql_tool(sales_df, {"group_by": "region"}, prior_results={})

    def test_sql_tool_group_by_with_select_columns_succeeds(self, sales_df):
        result = sql_tool(
            sales_df,
            {"select_columns": ["region", "SUM(revenue) AS total"], "group_by": "region"},
            prior_results={},
        )
        assert result.get("failed") is not True
        assert len(result["result"]) == 4  # 4 distinct regions in sales_df fixture

    def test_sql_tool_order_by_select_alias_does_not_raise(self, sales_df):
        # ORDER BY referencing an alias defined in select_columns (a normal,
        # valid SQL pattern) must not be rejected as a missing column — the
        # alias only exists within this query, not on the original dataframe.
        result = sql_tool(
            sales_df,
            {
                "select_columns": ["region", "SUM(revenue) AS total_revenue"],
                "group_by": "region",
                "order_by": "total_revenue",
            },
            prior_results={},
        )
        assert result.get("failed") is not True
        assert len(result["result"]) == 4


# ─── 5.3 Pandas transform tool ───────────────────────────────────────────────

class TestPandasTools:
    def test_groupby_sum(self, sales_df):
        result = pandas_groupby(sales_df, ["region"], "revenue", "sum")
        assert result["n_groups"] == 4
        rows = {r["region"]: r["revenue"] for r in result["result"]}
        assert abs(rows["North"] - 2550.0) < 0.1

    def test_groupby_mean(self, sales_df):
        result = pandas_groupby(sales_df, ["quarter"], "revenue", "mean")
        assert result["n_groups"] == 2

    def test_groupby_multi_key(self, sales_df):
        result = pandas_groupby(sales_df, ["region", "quarter"], "revenue", "sum")
        assert result["n_groups"] == 8

    def test_groupby_sort_desc(self, sales_df):
        result = pandas_groupby(sales_df, ["region"], "revenue", "sum", sort_desc=True)
        revenues = [r["revenue"] for r in result["result"]]
        assert revenues == sorted(revenues, reverse=True)

    def test_pivot(self, sales_df):
        result = pandas_pivot(sales_df, index="region", columns="quarter",
                               values="revenue", agg_func="sum")
        assert result["shape"][0] == 4   # 4 regions
        assert result["shape"][1] == 2   # 2 quarters

    def test_filter_equals(self, sales_df):
        result = pandas_filter(sales_df, [{"column": "region", "op": "==", "value": "North"}])
        assert result["row_count"] == 2
        assert all(r["region"] == "North" for r in result["result"])

    def test_filter_greater_than(self, sales_df):
        result = pandas_filter(sales_df, [{"column": "revenue", "op": ">", "value": 1100.0}])
        assert result["row_count"] > 0
        assert all(r["revenue"] > 1100.0 for r in result["result"])

    def test_filter_in(self, sales_df):
        result = pandas_filter(sales_df, [{"column": "region", "op": "in",
                                            "value": ["North", "South"]}])
        assert result["row_count"] == 4

    def test_filter_unknown_column_skipped(self, sales_df):
        result = pandas_filter(sales_df, [{"column": "nonexistent", "op": "==", "value": "x"}])
        assert result["row_count"] == len(sales_df)  # no-op

    def test_pandas_tool_dispatcher_groupby(self, sales_df):
        result = pandas_tool(sales_df,
                             {"operation": "groupby", "group_by": ["region"],
                              "agg_col": "revenue", "agg_func": "sum"},
                             prior_results={})
        assert "result_summary" in result
        assert "groupby" in result["result_summary"]


# ─── Derived columns (arithmetic + date-part) ─────────────────────────────────

class TestDerive:
    def test_derive_arithmetic_subtraction(self):
        df = pd.DataFrame({"unit_price": [100.0, 50.0], "cost": [60.0, 20.0]})
        result = pandas_derive(df, "margin", "unit_price", "-", "cost")
        assert result["result_df"]["margin"].tolist() == [40.0, 30.0]

    def test_derive_arithmetic_constant(self):
        df = pd.DataFrame({"revenue": [100.0, 200.0]})
        result = pandas_derive(df, "revenue_x2", "revenue", "*", 2)
        assert result["result_df"]["revenue_x2"].tolist() == [200.0, 400.0]

    def test_derive_quarter_from_date(self):
        df = pd.DataFrame({"date": ["2023-01-15", "2023-04-01", "2023-07-20", "2023-10-05"]})
        result = pandas_derive(df, "quarter", "date", "quarter")
        assert result["result_df"]["quarter"].tolist() == [1, 2, 3, 4]

    def test_derive_year_and_month_from_date(self):
        df = pd.DataFrame({"date": ["2022-03-10", "2024-11-01"]})
        year = pandas_derive(df, "year", "date", "year")
        month = pandas_derive(df, "month", "date", "month")
        assert year["result_df"]["year"].tolist() == [2022, 2024]
        assert month["result_df"]["month"].tolist() == [3, 11]

    def test_derive_unsupported_op_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="unsupported op"):
            pandas_derive(df, "b", "a", "modulo", 2)

    def test_derive_date_part_unparseable_column_raises(self):
        df = pd.DataFrame({"a": ["not", "a", "date"]})
        with pytest.raises(ValueError, match="could not be parsed as dates"):
            pandas_derive(df, "quarter", "a", "quarter")

    def test_pandas_tool_derive_quarter_dispatcher(self):
        df = pd.DataFrame({"date": ["2023-01-15", "2023-07-20"]})
        result = pandas_tool(
            df, {"operation": "derive", "new_column": "quarter", "left": "date", "op": "quarter"},
            prior_results={},
        )
        assert result["result_df"]["quarter"].tolist() == [1, 3]

    def test_pandas_tool_derive_quarter_does_not_require_right(self):
        # Date-part ops are unary — "right" must not be a required parameter.
        df = pd.DataFrame({"date": ["2023-01-15"]})
        result = pandas_tool(
            df, {"new_column": "quarter", "left": "date", "op": "quarter"},  # no "operation", no "right"
            prior_results={},
        )
        assert result["result_df"]["quarter"].tolist() == [1]

    def test_pandas_tool_derive_arithmetic_still_requires_right(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required parameter"):
            pandas_tool(df, {"operation": "derive", "new_column": "c", "left": "a", "op": "+"}, {})

    def test_derive_quarter_then_groupby_chain(self):
        # The realistic two-step flow: derive quarter, then group revenue by it.
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01", "2023-04-01", "2023-04-15"],
            "revenue": [100.0, 200.0, 300.0, 400.0],
        })
        derived = pandas_tool(
            df, {"operation": "derive", "new_column": "quarter", "left": "date", "op": "quarter"}, {},
        )
        grouped = pandas_tool(
            derived["result_df"],
            {"operation": "groupby", "group_by": ["quarter"], "agg_col": "revenue", "agg_func": "sum"},
            {},
        )
        totals = {r["quarter"]: r["revenue"] for r in grouped["result"]}
        assert totals[1] == 300.0
        assert totals[2] == 700.0


# ─── 5.4 Statistical tool ─────────────────────────────────────────────────────

class TestStatisticalTools:
    def test_correlation_pearson(self, sales_df):
        result = compute_correlation(sales_df, ["revenue", "units"])
        assert "matrix" in result
        assert "revenue" in result["matrix"]
        # revenue and units should be strongly correlated
        corr_val = result["matrix"]["revenue"]["units"]
        assert abs(corr_val) > 0.9

    def test_correlation_no_columns_uses_all_numeric(self, sales_df):
        result = compute_correlation(sales_df)
        assert set(result["columns_used"]) == {"revenue", "units"}

    def test_regression(self, sales_df):
        result = run_linear_regression(sales_df, target="revenue", features=["units"])
        assert "r_squared" in result
        assert result["r_squared"] > 0.9
        assert "coefficients" in result
        assert "units" in result["coefficients"]

    def test_ttest_significant(self):
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "value": np.random.normal(10, 1, 20).tolist() + np.random.normal(20, 1, 20).tolist()
        })
        result = run_ttest(df, "group", "value")
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_ttest_not_significant(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 30,
            "value": np.random.normal(10, 5, 30).tolist() + np.random.normal(10.1, 5, 30).tolist()
        })
        result = run_ttest(df, "group", "value")
        assert "p_value" in result

    def test_ttest_insufficient_groups(self):
        df = pd.DataFrame({"group": ["A"] * 5, "value": [1, 2, 3, 4, 5]})
        result = run_ttest(df, "group", "value")
        assert "error" in result

    def test_statistical_tool_dispatcher(self, sales_df):
        result = statistical_tool(sales_df,
                                  {"test": "correlation", "method": "pearson"},
                                  prior_results={})
        assert "result" in result

    def test_correlation_raises_on_nonexistent_column(self, sales_df):
        with pytest.raises(ValueError, match="not found among numeric columns"):
            compute_correlation(sales_df, ["revenue", "customer_lifetime_value"])

    def test_statistical_tool_infers_ttest_from_keys_without_explicit_test(self, sales_df):
        # No "test" key given — must infer "ttest" from group_col/value_col,
        # not silently default to "correlation" the way it used to.
        result = statistical_tool(
            sales_df, {"group_col": "region", "value_col": "revenue"}, prior_results={},
        )
        assert result["result_summary"] == "stats/ttest complete"

    def test_statistical_tool_raises_when_test_undeterminable(self, sales_df):
        with pytest.raises(ValueError, match="cannot determine a test"):
            statistical_tool(sales_df, {"metric": "revenue", "grouping_variable": "region"}, prior_results={})

    def test_statistical_tool_ttest_insufficient_groups_raises(self):
        # run_ttest() itself returns {"error": ...} (see
        # test_ttest_insufficient_groups above) — but through the dispatcher
        # that must become a real failure, not a silently "successful" step.
        df = pd.DataFrame({"group": ["A"] * 5, "value": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="Need"):
            statistical_tool(df, {"test": "ttest", "group_col": "group", "value_col": "value"}, prior_results={})


# ─── 5.5 Anomaly detection ────────────────────────────────────────────────────

class TestAnomalyDetection:
    def test_zscore_detects_outliers(self, anomaly_df):
        result = detect_anomalies_zscore(anomaly_df, "value", threshold=3.0)
        assert result["n_outliers"] == 2
        assert result["method"] == "zscore"
        assert 500.0 in result["outlier_values"] or any(v > 300 for v in result["outlier_values"])

    def test_iqr_detects_outliers(self, anomaly_df):
        result = detect_anomalies_iqr(anomaly_df, "value")
        assert result["n_outliers"] >= 1
        assert result["method"] == "iqr"
        assert "lower_fence" in result
        assert "upper_fence" in result

    def test_zscore_no_outliers_clean_data(self):
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = detect_anomalies_zscore(df, "x", threshold=4.0)
        assert result["n_outliers"] == 0

    def test_iqr_fields_present(self, anomaly_df):
        result = detect_anomalies_iqr(anomaly_df, "value")
        for key in ["q1", "q3", "iqr", "lower_fence", "upper_fence", "outlier_pct"]:
            assert key in result

    def test_anomaly_tool_dispatcher_iqr(self, anomaly_df):
        result = anomaly_tool(anomaly_df, {"method": "iqr", "column": "value"}, {})
        assert result["result"]["n_outliers"] >= 1

    def test_anomaly_tool_dispatcher_zscore(self, anomaly_df):
        result = anomaly_tool(anomaly_df, {"method": "zscore", "column": "value",
                                            "threshold": 3.0}, {})
        assert result["result"]["n_outliers"] == 2

    def test_isolation_forest(self, anomaly_df):
        from src.tools.stats_tool import detect_anomalies_isolation_forest
        result = detect_anomalies_isolation_forest(anomaly_df, ["value"], contamination=0.05)
        assert result["method"] == "isolation_forest"
        assert result["n_anomalies"] > 0

    def test_anomaly_tool_requires_column(self, anomaly_df):
        with pytest.raises(ValueError, match="requires a 'column'"):
            anomaly_tool(anomaly_df, {"method": "iqr"}, {})

    def test_anomaly_tool_raises_on_nonexistent_column(self, anomaly_df):
        with pytest.raises(ValueError, match="not found in dataframe"):
            anomaly_tool(anomaly_df, {"method": "iqr", "column": "nonexistent"}, {})


# ─── Time-series and comparison ───────────────────────────────────────────────

class TestTimeSeriesAndComparison:
    def test_time_series_trend(self, time_df):
        result = analyze_time_series(time_df, "month", "revenue")
        assert result["trend"]["direction"] == "increasing"
        assert result["trend"]["significant"] is True
        assert result["n_periods"] == 6

    def test_time_series_summary_stats(self, time_df):
        result = analyze_time_series(time_df, "month", "revenue")
        assert result["summary_stats"]["min"] == 1000.0
        assert result["summary_stats"]["max"] == 1250.0

    def test_time_series_too_few_points(self):
        df = pd.DataFrame({"date": ["2024-01", "2024-02"], "val": [1.0, 2.0]})
        result = analyze_time_series(df, "date", "val")
        assert "error" in result

    def test_timeseries_tool_too_few_points_raises(self):
        # analyze_time_series() itself returns {"error": ...} (see above) —
        # through the dispatcher that must become a real failure.
        from src.tools.stats_tool import timeseries_tool
        df = pd.DataFrame({"date": ["2024-01", "2024-02"], "val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 3 data points"):
            timeseries_tool(df, {"date_col": "date", "value_col": "val"}, {})

    def test_compare_segments(self, sales_df):
        result = compare_segments(sales_df, "region", "revenue")
        assert len(result["segments"]) == 4
        assert "top_segment" in result

    def test_compare_segments_top_segment_respects_agg(self):
        # A segment with the highest *sum* isn't necessarily the one with
        # the highest *mean* — top_segment must rank by whichever agg was
        # actually requested, not be hardcoded to "mean". A: 100 rows of 10
        # (sum=1000, mean=10). B: 2 rows of 20 (sum=40, mean=20).
        df = pd.DataFrame({
            "segment": ["A"] * 100 + ["B"] * 2,
            "value": [10.0] * 100 + [20.0] * 2,
        })
        by_mean = compare_segments(df, "segment", "value", agg="mean")
        by_sum = compare_segments(df, "segment", "value", agg="sum")
        assert by_mean["top_segment"] == "B"   # highest mean
        assert by_sum["top_segment"] == "A"    # highest sum

    def test_compare_segments_rejects_unknown_agg(self, sales_df):
        with pytest.raises(ValueError, match="unsupported agg"):
            compare_segments(sales_df, "region", "revenue", agg="avg")

    def test_compare_segments_two_groups_ttest(self):
        df = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10,
            "value": np.random.normal(10, 1, 10).tolist() + np.random.normal(20, 1, 10).tolist()
        })
        result = compare_segments(df, "group", "value")
        assert result["significance_test"]["test"] == "t-test"

    def test_compare_segments_three_groups_anova(self, sales_df):
        result = compare_segments(sales_df, "region", "revenue")
        assert result["significance_test"]["test"] == "one-way ANOVA"


# ─── 5.1 Plan generator ───────────────────────────────────────────────────────

class TestPlanGenerator:
    def test_plan_returns_steps(self):
        plan_data = [
            {"step_number": 1, "method": "groupby_aggregate", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["region"],
                             "agg_col": "revenue", "agg_func": "sum"},
             "depends_on": [], "expected_output": "revenue by region", "rationale": "test"},
            {"step_number": 2, "method": "correlation", "tool": "statistical_test",
             "parameters": {"test": "correlation"},
             "depends_on": [], "expected_output": "correlation matrix", "rationale": "test"},
        ]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("Revenue by region")
        state["parsed_intent"] = {"query_type": "descriptive", "target_metrics": ["revenue"]}
        state["schema_summary"] = "Sales data with revenue, region columns"
        result = asyncio.run(agent.plan(state))
        assert len(result["analysis_plan"]) == 2
        assert result["analysis_plan"][0]["tool"] == "pandas_transform"

    def test_plan_logs_decision_trace(self):
        plan_data = [{"step_number": 1, "method": "groupby", "tool": "pandas_transform",
                       "parameters": {}, "depends_on": [],
                       "expected_output": "", "rationale": ""}]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("test")
        result = asyncio.run(agent.plan(state))
        assert any(t["action"] == "plan_analysis" for t in result["decision_trace"])

    def test_plan_llm_failure_yields_empty_plan(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("test")
        result = asyncio.run(agent.plan(state))
        assert result["analysis_plan"] == []

    def test_plan_tracks_token_usage(self):
        agent = AnalysisAgent(llm=_mock_llm([]))
        state = initial_state("test")
        result = asyncio.run(agent.plan(state))
        assert "analysis_agent" in result["token_usage"]

    def test_plan_prompt_includes_related_tables_for_sql_source(self, multi_table_sqlite_db):
        mock_llm = _mock_llm([])
        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("test")
        state["active_source"] = {
            "type": "sql", "path": multi_table_sqlite_db, "table_name": "orders",
            "schema": {"columns": [{"name": "order_id", "dtype": "INTEGER"}]},
        }
        asyncio.run(agent.plan(state))
        prompt = mock_llm.ainvoke.call_args[0][0][1].content
        assert "### Related Tables" in prompt
        assert "customers" in prompt
        assert multi_table_sqlite_db in prompt

    def test_plan_prompt_omits_related_tables_for_csv_source(self):
        mock_llm = _mock_llm([])
        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("test")
        state["active_source"] = {
            "type": "csv", "path": "data/demo/sales_data.csv",
            "schema": {"columns": [{"name": "revenue", "dtype": "float64"}]},
        }
        asyncio.run(agent.plan(state))
        prompt = mock_llm.ainvoke.call_args[0][0][1].content
        assert "### Related Tables" not in prompt

    @pytest.mark.parametrize("query_type,expected_tool", [
        ("descriptive",  "pandas_transform"),
        ("diagnostic",   "statistical_test"),
        ("predictive",   "time_series"),
        ("comparative",  "comparison"),
        ("exploratory",  "statistical_test"),
    ])
    def test_plan_for_5_query_types(self, query_type, expected_tool):
        """5.1 acceptance: plan generated for all 5 query types."""
        plan_data = [{"step_number": 1, "method": "auto", "tool": expected_tool,
                       "parameters": {}, "depends_on": [],
                       "expected_output": "result", "rationale": "auto"}]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("test")
        state["parsed_intent"] = {"query_type": query_type}
        result = asyncio.run(agent.plan(state))
        assert result["analysis_plan"][0]["tool"] == expected_tool


# ─── Related tables manifest (cross-table join support) ───────────────────────

class TestRelatedTablesManifest:
    def test_lists_other_tables_for_sql_source(self, multi_table_sqlite_db):
        active = {"type": "sql", "path": multi_table_sqlite_db, "table_name": "orders"}
        manifest = _related_tables_manifest(active)
        assert "customers" in manifest
        assert "orders" not in manifest.split("Other tables")[1].split("Table:")[0]  # excluded, only mentioned in the intro sentence
        assert multi_table_sqlite_db in manifest

    def test_empty_for_non_sql_source(self):
        active = {"type": "csv", "path": "data/demo/sales_data.csv"}
        assert _related_tables_manifest(active) == ""

    def test_empty_for_none_source(self):
        assert _related_tables_manifest(None) == ""

    def test_empty_on_bad_connection_string(self):
        active = {"type": "sql", "path": "sqlite:///nonexistent/path/does/not/exist.db"}
        # Must degrade gracefully — a broken connection string must not
        # block planning against the single active table.
        assert _related_tables_manifest(active) == ""


# ─── 5.6 Step executor ────────────────────────────────────────────────────────

class TestStepExecutor:
    def test_execute_single_step(self, sales_df, tmp_path):
        csv = tmp_path / "sales.csv"
        sales_df.to_csv(str(csv), index=False)

        plan_data = [{"step_number": 1, "method": "groupby", "tool": "pandas_transform",
                       "parameters": {"operation": "groupby", "group_by": ["region"],
                                       "agg_col": "revenue", "agg_func": "sum"},
                       "depends_on": [], "expected_output": "grouped", "rationale": ""}]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert len(result["analysis_results"]) == 1
        assert result["analysis_results"][0]["failed"] is False

    def test_execute_respects_dependencies(self, sales_df, tmp_path):
        csv = tmp_path / "sales.csv"
        sales_df.to_csv(str(csv), index=False)

        plan_data = [
            {"step_number": 1, "method": "filter", "tool": "pandas_transform",
             "parameters": {"operation": "filter",
                             "filters": [{"column": "region", "op": "==", "value": "North"}]},
             "depends_on": [], "expected_output": "filtered df", "rationale": ""},
            {"step_number": 2, "method": "groupby", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["quarter"],
                             "agg_col": "revenue", "agg_func": "sum"},
             "depends_on": [1], "expected_output": "grouped", "rationale": ""},
        ]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert len(result["analysis_results"]) == 2
        assert all(not r["failed"] for r in result["analysis_results"])

    def test_execute_independent_steps_each_get_original_dataframe(self, sales_df, tmp_path):
        """
        Two independent steps (depends_on=[]) must each see the full
        original dataset, not whatever the immediately-preceding step left
        behind. Step 1 groups by region+revenue, collapsing away 'quarter'
        and 'units'; step 2 is unrelated (depends_on=[]) and needs 'quarter'
        and 'units' — it must succeed against the original data, not fail
        with a 'column not found' error caused by step 1's leftover df.
        """
        csv = tmp_path / "sales.csv"
        sales_df.to_csv(str(csv), index=False)

        plan_data = [
            {"step_number": 1, "method": "groupby_region", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["region"],
                             "agg_col": "revenue", "agg_func": "sum"},
             "depends_on": [], "expected_output": "revenue by region", "rationale": ""},
            {"step_number": 2, "method": "groupby_quarter", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["quarter"],
                             "agg_col": "units", "agg_func": "sum"},
             "depends_on": [], "expected_output": "units by quarter", "rationale": ""},
        ]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert len(result["analysis_results"]) == 2
        assert all(not r["failed"] for r in result["analysis_results"]), result["analysis_results"]

    def test_execute_unknown_tool_marks_failed(self, sales_df, tmp_path):
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)
        plan_data = [{"step_number": 1, "method": "mystery", "tool": "unknown_tool",
                       "parameters": {}, "depends_on": [],
                       "expected_output": "", "rationale": ""}]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert result["analysis_results"][0]["failed"] is True


# ─── 5.7 Error recovery ───────────────────────────────────────────────────────

class TestErrorRecovery:
    def test_failed_step_does_not_abort_execution(self, sales_df, tmp_path):
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)

        plan_data = [
            {"step_number": 1, "method": "bad", "tool": "unknown_tool",
             "parameters": {}, "depends_on": [], "expected_output": "", "rationale": ""},
            {"step_number": 2, "method": "groupby", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["region"],
                             "agg_col": "revenue", "agg_func": "sum"},
             "depends_on": [], "expected_output": "", "rationale": ""},
        ]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert len(result["analysis_results"]) == 2
        assert result["analysis_results"][0]["failed"] is True
        assert result["analysis_results"][1]["failed"] is False

    def test_simplify_step_strips_optional_params(self):
        step = AnalysisStep(
            step_number=1, method="groupby", tool="pandas_transform",
            parameters={"operation": "groupby", "group_by": ["x"], "agg_col": "y",
                        "agg_func": "sum", "sort_desc": True, "extra_param": "drop_me"},
            depends_on=[], expected_output="", rationale="",
        )
        simplified = _simplify_step(step)
        assert "extra_param" not in simplified.parameters
        assert "operation" in simplified.parameters

    def test_simplify_step_changes_agg_to_count(self):
        step = AnalysisStep(
            step_number=1, method="groupby", tool="pandas_transform",
            parameters={"operation": "groupby", "group_by": ["x"],
                        "agg_col": "y", "agg_func": "sum"},
            depends_on=[], expected_output="", rationale="",
        )
        simplified = _simplify_step(step)
        assert simplified.parameters["agg_func"] == "count"

    def test_repair_step_fixes_wrong_column_name(self, sales_df, tmp_path):
        """The retry should feed the real error message to the LLM and use
        its targeted fix, not just blindly strip parameters."""
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)

        # "Revenue" doesn't exist (real column is lowercase "revenue") --
        # this raises _require_columns' error with the real column list.
        plan_data = [
            {"step_number": 1, "method": "groupby", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["region"],
                             "agg_col": "Revenue", "agg_func": "sum"},
             "depends_on": [], "expected_output": "", "rationale": ""},
        ]
        repair_response = MagicMock(
            content=json.dumps({
                "parameters": {"operation": "groupby", "group_by": ["region"],
                                "agg_col": "revenue", "agg_func": "sum"},
                "reasoning": "Fixed casing: Revenue -> revenue",
            }),
            usage_metadata={"input_tokens": 50, "output_tokens": 30},
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=repair_response)

        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))

        mock_llm.ainvoke.assert_awaited_once()
        assert result["analysis_results"][0]["failed"] is False
        assert any("LLM-repaired" in w for w in result["analysis_results"][0]["warnings"])

    def test_repair_step_falls_back_to_simplify_when_llm_returns_null(self, sales_df, tmp_path):
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)
        # agg_func="mean" on a string column raises -- _simplify_step forces
        # agg_func="count" (works on any dtype), so the retry succeeds.
        plan_data = [
            {"step_number": 1, "method": "groupby", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["quarter"],
                             "agg_col": "region", "agg_func": "mean"},
             "depends_on": [], "expected_output": "", "rationale": ""},
        ]
        repair_response = MagicMock(
            content=json.dumps({"parameters": None, "reasoning": "not confident"}),
            usage_metadata={"input_tokens": 50, "output_tokens": 20},
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=repair_response)

        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))

        assert result["analysis_results"][0]["failed"] is False
        assert any("generic simplify" in w for w in result["analysis_results"][0]["warnings"])

    def test_repair_step_falls_back_to_simplify_when_llm_call_fails(self, sales_df, tmp_path):
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)
        plan_data = [
            {"step_number": 1, "method": "groupby", "tool": "pandas_transform",
             "parameters": {"operation": "groupby", "group_by": ["quarter"],
                             "agg_col": "region", "agg_func": "mean"},
             "depends_on": [], "expected_output": "", "rationale": ""},
        ]
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        agent = AnalysisAgent(llm=mock_llm)
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))

        assert result["analysis_results"][0]["failed"] is False
        assert any("generic simplify" in w for w in result["analysis_results"][0]["warnings"])

    def test_repair_step_returns_none_on_missing_parameters_key(self):
        from src.agents.analysis_agent import AnalysisStep
        step = AnalysisStep(
            step_number=1, method="groupby", tool="pandas_transform",
            parameters={"agg_col": "Revenue"}, depends_on=[],
            expected_output="", rationale="",
        )
        bad_response = MagicMock(
            content=json.dumps({"reasoning": "oops, forgot the key"}),
            usage_metadata={},
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=bad_response)
        agent = AnalysisAgent(llm=mock_llm)
        repaired = asyncio.run(agent._repair_step(step, ValueError("bad column"), "- revenue (float64)"))
        assert repaired is None


# ─── JSON parsing ─────────────────────────────────────────────────────────────

class TestParseJson:
    def test_parses_plain_array(self):
        from src.agents.analysis_agent import _parse_json
        assert _parse_json('[{"a": 1}]') == [{"a": 1}]

    def test_parses_plain_object(self):
        from src.agents.analysis_agent import _parse_json
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_parses_markdown_fenced_array(self):
        from src.agents.analysis_agent import _parse_json
        text = '```json\n[{"a": 1}]\n```'
        assert _parse_json(text) == [{"a": 1}]

    def test_parses_object_containing_a_nested_array(self):
        """Regression: the old bracket-scanning logic tried "[" ... "]"
        first and unconditionally, so an object whose only array is
        *nested* (e.g. {"parameters": {"group_by": [...]}}) got truncated
        down to just that inner array, discarding the object around it."""
        from src.agents.analysis_agent import _parse_json
        data = {"parameters": {"group_by": ["region"], "agg_col": "revenue"},
                "reasoning": "fixed it"}
        assert _parse_json(json.dumps(data)) == data

    def test_raises_on_garbage(self):
        from src.agents.analysis_agent import _parse_json
        with pytest.raises(ValueError):
            _parse_json("not json at all")


# ─── 5.8 Result aggregator ────────────────────────────────────────────────────

class TestResultAggregator:
    def test_aggregate_counts_steps(self):
        results = [
            {"step": 1, "method": "groupby", "result": {}, "result_summary": "ok", "failed": False},
            {"step": 2, "method": "stats",   "result": {}, "result_summary": "ok", "failed": False},
            {"step": 3, "method": "bad",     "result": None, "result_summary": "fail", "failed": True},
        ]
        agg = _aggregate(results)
        assert agg["n_steps_total"] == 3
        assert agg["n_steps_successful"] == 2

    def test_aggregate_key_findings(self):
        results = [
            {"step": 1, "method": "groupby", "result": {"rows": []},
             "result_summary": "4 groups", "failed": False},
        ]
        agg = _aggregate(results)
        assert len(agg["key_findings"]) == 1
        assert agg["key_findings"][0]["step"] == 1

    def test_intermediate_data_in_state(self, sales_df, tmp_path):
        """After execute(), state['intermediate_data'] is populated."""
        csv = tmp_path / "d.csv"
        sales_df.to_csv(str(csv), index=False)
        plan_data = [{"step_number": 1, "method": "groupby", "tool": "pandas_transform",
                       "parameters": {"operation": "groupby", "group_by": ["region"],
                                       "agg_col": "revenue", "agg_func": "sum"},
                       "depends_on": [], "expected_output": "", "rationale": ""}]
        agent = AnalysisAgent(llm=_mock_llm(plan_data))
        state = initial_state("q", data_sources=[{"type": "csv", "path": str(csv)}])
        state["active_source"] = {"type": "csv", "path": str(csv)}
        state["analysis_plan"] = plan_data
        result = asyncio.run(agent.execute(state))
        assert result["intermediate_data"] is not None
        assert "n_steps_total" in result["intermediate_data"]


# ─── Dependency ordering ──────────────────────────────────────────────────────

class TestExecutionOrder:
    def _make_step(self, num, deps):
        return AnalysisStep(step_number=num, method="", tool="pandas_transform",
                            parameters={}, depends_on=deps,
                            expected_output="", rationale="")

    def test_independent_steps_in_order(self):
        steps = [self._make_step(1, []), self._make_step(2, []), self._make_step(3, [])]
        ordered = _execution_order(steps)
        assert [s.step_number for s in ordered] == [1, 2, 3]

    def test_dependent_step_after_prerequisite(self):
        steps = [self._make_step(1, []), self._make_step(2, [1]), self._make_step(3, [2])]
        ordered = _execution_order(steps)
        nums = [s.step_number for s in ordered]
        assert nums.index(1) < nums.index(2) < nums.index(3)

    def test_diamond_dependency(self):
        steps = [
            self._make_step(1, []),
            self._make_step(2, [1]),
            self._make_step(3, [1]),
            self._make_step(4, [2, 3]),
        ]
        ordered = _execution_order(steps)
        nums = [s.step_number for s in ordered]
        assert nums.index(1) < nums.index(4)
        assert nums.index(2) < nums.index(4)
        assert nums.index(3) < nums.index(4)


# ─── Tool registry completeness ───────────────────────────────────────────────

def test_tool_registry_has_all_required_tools():
    required = {"sql_query", "pandas_transform", "statistical_test",
                "anomaly_detection", "time_series", "comparison"}
    assert required.issubset(set(TOOL_REGISTRY.keys()))


# ─── AnalysisStep dataclass ───────────────────────────────────────────────────

def test_analysis_step_from_dict_roundtrip():
    d = {"step_number": 1, "method": "groupby", "tool": "pandas_transform",
         "parameters": {"op": "sum"}, "depends_on": [0],
         "expected_output": "df", "rationale": "because"}
    step = AnalysisStep.from_dict(d)
    assert step.step_number == 1
    assert step.depends_on == [0]
    rd = step.to_dict()
    assert rd["method"] == "groupby"


def test_analysis_plan_from_list():
    data = [
        {"step_number": 1, "method": "a", "tool": "sql_query",
         "parameters": {}, "depends_on": [], "expected_output": "", "rationale": ""}
    ]
    plan = AnalysisPlan.from_llm_response(data)
    assert len(plan.steps) == 1
    assert plan.to_state_list()[0]["tool"] == "sql_query"


def test_analysis_plan_from_dict_with_metadata():
    data = {
        "estimated_complexity": "complex",
        "rationale": "multi-step",
        "steps": [{"step_number": 1, "method": "x", "tool": "comparison",
                    "parameters": {}, "depends_on": [],
                    "expected_output": "", "rationale": ""}]
    }
    plan = AnalysisPlan.from_llm_response(data)
    assert plan.estimated_complexity == "complex"
    assert plan.rationale == "multi-step"
