"""
Phase 4 tests — Data Source Connection.
All file I/O uses pytest tmp_path; LLM calls are mocked.
Run with: pytest tests/unit/test_phase4.py -v
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.tools.data_connector import (
    DataConnector,
    SchemaInfo,
    connect_csv,
    connect_excel,
    connect_json,
    connect_sql,
    extract_schema,
    generate_nl_summary,
    list_related_tables,
)
from src.tools.source_registry import SourceRegistry, _infer_name


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "sales.csv"
    p.write_text(
        "product,region,revenue,quarter\n"
        "Widget,North,1200.0,Q1\n"
        "Gadget,South,850.5,Q1\n"
        "Widget,East,,Q2\n"          # null revenue
        "Gadget,North,990.0,Q2\n"
        "Widget,South,1100.0,Q3\n"
    )
    return str(p)


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
    return str(p)


@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "orders.json"
    records = [
        {"order_id": 1, "amount": 100.0, "status": "shipped"},
        {"order_id": 2, "amount": 250.0, "status": "pending"},
        {"order_id": 3, "amount": 75.0,  "status": "shipped"},
    ]
    p.write_text(json.dumps(records))
    return str(p)


@pytest.fixture
def jsonl_file(tmp_path):
    p = tmp_path / "events.jsonl"
    lines = [
        json.dumps({"event": "click", "user": "u1"}),
        json.dumps({"event": "view",  "user": "u2"}),
    ]
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def excel_file(tmp_path):
    p = tmp_path / "report.xlsx"
    df = pd.DataFrame({"month": ["Jan", "Feb"], "sales": [1000, 1200]})
    df.to_excel(str(p), index=False, sheet_name="Summary")
    return str(p)


@pytest.fixture
def sqlite_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE customers (id INTEGER, name TEXT, spend REAL)")
    conn.execute("INSERT INTO customers VALUES (1,'Alice',500.0)")
    conn.execute("INSERT INTO customers VALUES (2,'Bob',300.0)")
    conn.commit()
    conn.close()
    return str(db_path), f"sqlite:///{db_path}"


@pytest.fixture
def multi_table_sqlite_db(tmp_path):
    db_path = tmp_path / "multi.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, revenue REAL)")
    conn.execute("INSERT INTO customers VALUES (1, 'Alice')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 100.0)")
    conn.commit()
    conn.close()
    return str(db_path), f"sqlite:///{db_path}"


# ─── 4.1 CSV connector ────────────────────────────────────────────────────────

class TestCSVConnector:
    def test_loads_csv(self, csv_file):
        df = connect_csv(csv_file)
        assert len(df) == 5
        assert "revenue" in df.columns
        assert "region" in df.columns

    def test_loads_tsv(self, tsv_file):
        df = connect_csv(tsv_file)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_handles_nulls(self, csv_file):
        df = connect_csv(csv_file)
        assert df["revenue"].isna().sum() == 1

    def test_schema_extraction(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        assert schema.row_count == 5
        assert schema.source_type == "csv"
        col_names = schema.column_names()
        assert "product" in col_names
        assert "revenue" in col_names

    def test_schema_null_pct(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        rev_col = next(c for c in schema.columns if c.name == "revenue")
        assert abs(rev_col.null_pct - 0.2) < 0.01  # 1 out of 5

    def test_schema_detects_numeric(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        assert schema.numeric_columns() == ["revenue"]

    def test_schema_preview(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        assert len(schema.preview) == 5
        assert "product" in schema.preview[0]

    def test_to_source_dict(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        d = schema.to_source_dict()
        assert d["type"] == "csv"
        assert d["path"] == csv_file
        assert "schema" in d
        assert "preview" in d


# ─── 4.2 SQL connector ───────────────────────────────────────────────────────

class TestSQLConnector:
    def test_loads_sqlite_auto_detect_table(self, sqlite_db):
        _, conn_str = sqlite_db
        df, table_name = connect_sql(conn_str)
        assert table_name == "customers"
        assert len(df) == 2
        assert "name" in df.columns

    def test_loads_sqlite_explicit_table(self, sqlite_db):
        _, conn_str = sqlite_db
        df, table_name = connect_sql(conn_str, table_name="customers")
        assert table_name == "customers"
        assert len(df) == 2

    def test_loads_sqlite_raw_query(self, sqlite_db):
        _, conn_str = sqlite_db
        df, table_name = connect_sql(conn_str, query="SELECT * FROM customers WHERE spend > 400")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Alice"
        assert table_name == "custom_query"

    def test_schema_extraction_sql(self, sqlite_db):
        _, conn_str = sqlite_db
        df, table_name = connect_sql(conn_str)
        schema = extract_schema(df, "sql", conn_str, table_name=table_name)
        assert schema.source_type == "sql"
        assert schema.table_name == "customers"
        assert schema.row_count == 2

    def test_raises_on_missing_table(self, sqlite_db):
        _, conn_str = sqlite_db
        with pytest.raises(Exception):
            connect_sql(conn_str, table_name="nonexistent_table")

    def test_list_related_tables(self, multi_table_sqlite_db):
        _, conn_str = multi_table_sqlite_db
        tables = list_related_tables(conn_str)
        assert set(tables.keys()) == {"customers", "orders"}
        assert any(c.startswith("customer_id") for c in tables["customers"])
        assert any(c.startswith("revenue") for c in tables["orders"])

    def test_list_related_tables_excludes_active_table(self, multi_table_sqlite_db):
        _, conn_str = multi_table_sqlite_db
        tables = list_related_tables(conn_str, exclude_table="orders")
        assert set(tables.keys()) == {"customers"}

    def test_list_related_tables_no_data_loaded(self, multi_table_sqlite_db):
        # Schema-only introspection — must not pull any rows into memory.
        _, conn_str = multi_table_sqlite_db
        tables = list_related_tables(conn_str)
        assert isinstance(tables["orders"], list)
        assert all(isinstance(c, str) for c in tables["orders"])


# ─── 4.5 JSON / Excel connectors ─────────────────────────────────────────────

class TestJSONConnector:
    def test_loads_json_array(self, json_file):
        df = connect_json(json_file)
        assert len(df) == 3
        assert "order_id" in df.columns

    def test_loads_jsonl(self, jsonl_file):
        df = connect_json(jsonl_file)
        assert len(df) == 2
        assert "event" in df.columns

    def test_schema_extraction_json(self, json_file):
        df = connect_json(json_file)
        schema = extract_schema(df, "json", json_file)
        assert schema.source_type == "json"
        assert schema.row_count == 3


class TestExcelConnector:
    def test_loads_excel_first_sheet(self, excel_file):
        df, sheet = connect_excel(excel_file)
        assert sheet == "Summary"
        assert len(df) == 2
        assert "sales" in df.columns

    def test_loads_excel_explicit_sheet(self, excel_file):
        df, sheet = connect_excel(excel_file, sheet_name="Summary")
        assert sheet == "Summary"
        assert len(df) == 2

    def test_schema_extraction_excel(self, excel_file):
        df, sheet = connect_excel(excel_file)
        schema = extract_schema(df, "excel", excel_file, table_name=sheet)
        assert schema.source_type == "excel"
        assert schema.table_name == "Summary"


# ─── 4.3 NL Schema Summary ────────────────────────────────────────────────────

class TestNLSchemaSummary:
    def test_llm_summary_called(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Sales dataset with 5 rows tracking product revenue by region and quarter."
        ))
        summary = asyncio.run(generate_nl_summary(schema, llm=mock_llm))
        assert "Sales" in summary or "revenue" in summary.lower()
        mock_llm.ainvoke.assert_called_once()

    def test_llm_failure_falls_back_to_rule_based(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        summary = asyncio.run(generate_nl_summary(schema, llm=mock_llm))
        # Rule-based fallback includes row count and column names
        assert "5" in summary
        assert "rows" in summary.lower()

    def test_rule_based_summary_contains_key_facts(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        summary = schema.rule_based_summary()
        assert "5" in summary
        assert any(col in summary for col in ["product", "region", "revenue"])

    def test_rule_based_summary_with_table_name(self, sqlite_db):
        _, conn_str = sqlite_db
        df, table_name = connect_sql(conn_str)
        schema = extract_schema(df, "sql", conn_str, table_name=table_name)
        summary = schema.rule_based_summary()
        assert "customers" in summary

    def test_connector_with_summary_5_datasets(self, tmp_path):
        """4.3 acceptance: NL summaries generated for 5 different dataset shapes."""
        datasets = [
            ("revenue.csv",  "store,revenue,month\nA,100,Jan\nB,200,Feb\n"),
            ("users.csv",    "user_id,age,country\n1,25,US\n2,32,UK\n"),
            ("events.csv",   "event,ts,user\nclick,2024-01,u1\nview,2024-01,u2\n"),
            ("products.csv", "sku,price,category\nABC,9.99,electronics\n"),
            ("logs.csv",     "level,msg,service\nERROR,crash,api\nINFO,ok,api\n"),
        ]
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=lambda msgs: MagicMock(
                content=f"Dataset with {msgs[1].content[:20]}"
            )
        )
        connector = DataConnector(llm=mock_llm)
        for filename, content in datasets:
            p = tmp_path / filename
            p.write_text(content)
            schema, summary = asyncio.run(connector.connect_with_summary(
                {"type": "csv", "path": str(p)}
            ))
            assert isinstance(schema, SchemaInfo)
            assert len(summary) > 0


# ─── 4.4 MCP quality delegation ──────────────────────────────────────────────

class TestMCPQualityDelegation:
    def test_connect_and_profile_calls_mcp(self, csv_file):
        """connect_with_summary should work; MCP profiling is tested via nodes."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Sales data with 5 rows."
        ))
        connector = DataConnector(llm=mock_llm)
        schema, summary = asyncio.run(connector.connect_with_summary(
            {"type": "csv", "path": csv_file}
        ))
        assert schema.row_count == 5
        assert "Sales" in summary or len(summary) > 0

    def test_connect_and_profile_node_uses_connector_and_mcp(self, csv_file):
        """
        Full node integration: DataConnector + MCP fallback (both mocked).
        Verifies state fields are populated correctly.
        """
        import src.graph.nodes as nodes
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import DataQualityReport
        from src.state.graph_state import initial_state

        # Mock LLM for schema summary
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Sales dataset with product revenue by region."
        ))

        # Mock MCP client — no critical issues
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_report = DataQualityReport(
            row_count=5, columns=[], quality_issues=[], has_critical_issues=False
        )
        mock_mcp.profile_dataset = AsyncMock(
            return_value=(mock_report, {"system": "data_cleaner", "tool": "profile_dataset",
                                        "mode": "mcp", "args": {}, "duration_ms": 5.0})
        )

        # Inject mocks
        old_connector = nodes._data_connector
        old_mcp = nodes._subsystem_client
        nodes._data_connector = DataConnector(llm=mock_llm)
        nodes._subsystem_client = mock_mcp

        try:
            state = initial_state("Show revenue", data_sources=[{"type": "csv", "path": csv_file}])
            result = asyncio.run(nodes.connect_and_profile_node(state))
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

        assert result["schema_summary"] == "Sales dataset with product revenue by region."
        assert result["active_source"] is not None
        assert result["data_quality_report"]["has_critical_issues"] is False
        assert len(result["mcp_call_log"]) == 1

    def test_critical_issues_triggers_cleaning(self, csv_file, tmp_path, monkeypatch):
        """When Data Cleaner reports critical issues, cleaning is invoked."""
        import src.graph.nodes as nodes
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import (
            CleaningResult,
            DataQualityReport,
            QualityIssue,
        )
        from src.state.graph_state import initial_state

        # M8: clean_dataset's output must resolve inside
        # settings.maeda_artifact_root/<run_id> to pass path validation.
        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Data summary."))

        critical_report = DataQualityReport(
            row_count=5, columns=[], has_critical_issues=True,
            quality_issues=[QualityIssue(code="dup_pk", severity="critical")],
        )

        state = initial_state("q", data_sources=[{"type": "csv", "path": csv_file}])

        # Byte-identical to csv_file but at a *distinct* path under this
        # round's expected artifact directory -- exercises the no_diff stop
        # condition (content unchanged) separately from path validation
        # (which a same-as-input path would fail on its own terms).
        clean_dir = tmp_path / f"{state['run_id']}_clean1"
        clean_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = clean_dir / "sales.csv"
        cleaned_path.write_text(Path(csv_file).read_text())

        # Real shape (agentic-data-cleaner-v2 mcp_app.py clean_dataset) is a
        # dict, not a string — see M3.
        clean_result = CleaningResult(
            cleaned_path=str(cleaned_path),
            changes_summary={"total_rounds": 1, "plan_steps": 2, "needs_review_count": 0},
            rows_affected=2,
        )

        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 3.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(critical_report, _log("profile_dataset")))
        # 定案 #3: get_cleaning_plan is no longer called from the node —
        # left unconfigured here; calling it would raise on the mock.
        mock_mcp.clean_dataset = AsyncMock(return_value=(clean_result, _log("clean_dataset")))

        old_connector = nodes._data_connector
        old_mcp = nodes._subsystem_client
        nodes._data_connector = DataConnector(llm=mock_llm)
        nodes._subsystem_client = mock_mcp

        try:
            result = asyncio.run(nodes.connect_and_profile_node(state))
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

        assert result["cleaning_applied"] is True
        assert result["cleaning_summary"]["plan_steps"] == 2
        assert result["cleaning_stop_reason"] == "no_diff"
        assert len(result["mcp_call_log"]) == 2  # profile + clean
        mock_mcp.get_cleaning_plan.assert_not_called()


# ─── M8: _validate_cleaned_path (path/hash protection, 定案 #16) ──────────────

class TestValidateClearedPathHelper:
    """Direct tests of nodes.py::_validate_cleaned_path, isolated from the
    full node -- each failure mode gets its own case rather than relying on
    the end-to-end cleaning loop to exercise all of them indirectly."""

    @staticmethod
    def _setup(tmp_path):
        from src.graph.nodes import _validate_cleaned_path
        run_id = "run123"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        return _validate_cleaned_path, str(tmp_path), run_id, run_dir

    def test_valid_path_passes(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        out = run_dir / "cleaned.csv"
        out.write_text("a,b\n1,2\n")
        assert validate(str(out), "/some/other/input.csv", artifact_root, run_id) is None

    def test_relative_path_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        reason = validate("relative/cleaned.csv", "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "absolute" in reason

    def test_non_normalized_path_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        out = run_dir / "cleaned.csv"
        out.write_text("a,b\n1,2\n")
        traversal_path = str(run_dir / ".." / run_id / "cleaned.csv")
        reason = validate(traversal_path, "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "normalized" in reason

    def test_outside_artifact_root_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        sibling = tmp_path / f"{run_id}_extra" / "cleaned.csv"  # shares a string prefix, not a real subdir
        sibling.parent.mkdir(parents=True)
        sibling.write_text("a,b\n1,2\n")
        reason = validate(str(sibling), "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "outside" in reason

    def test_identical_to_input_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        out = run_dir / "cleaned.csv"
        out.write_text("a,b\n1,2\n")
        reason = validate(str(out), str(out), artifact_root, run_id)
        assert reason is not None and "identical" in reason

    def test_unreadable_output_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        missing = run_dir / "does_not_exist.csv"
        reason = validate(str(missing), "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "readable" in reason

    def test_unsupported_format_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        out = run_dir / "cleaned.parquet"
        out.write_text("not really parquet but readable")
        reason = validate(str(out), "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "format" in reason

    def test_empty_path_rejected(self, tmp_path):
        validate, artifact_root, run_id, run_dir = self._setup(tmp_path)
        reason = validate("", "/some/input.csv", artifact_root, run_id)
        assert reason is not None and "empty" in reason


# ─── M7: cleaning loop (TB0.5 v1, ECOSYSTEM_INTEGRATION_PLAN.md 附录 B) ───────
#
# connect_and_profile_node's cleaning loop, rewritten in M7. Each test
# drives the node directly -- twice in a row for the multi-round cases,
# threading `state` through by hand the way the LangGraph cyclic edge
# ("clean" -> connect_and_profile_data) would -- against a mocked
# SubSystemWithFallback, since the loop's convergence logic is entirely
# about how state flows between rounds.

class TestCleaningLoop:
    @staticmethod
    def _mock_mcp(**method_side_effects):
        from src.mcp_client.fallback import SubSystemWithFallback
        mock = MagicMock(spec=SubSystemWithFallback)
        for name, effect in method_side_effects.items():
            setattr(mock, name, AsyncMock(side_effect=effect))
        return mock

    @staticmethod
    def _log(tool):
        return {"system": "data_cleaner", "tool": tool,
                "mode": "mcp", "args": {}, "duration_ms": 1.0}

    @staticmethod
    def _run(state, mock_mcp):
        import src.graph.nodes as nodes
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Summary."))
        old_connector = nodes._data_connector
        old_mcp = nodes._subsystem_client
        nodes._data_connector = DataConnector(llm=mock_llm)
        nodes._subsystem_client = mock_mcp
        try:
            return asyncio.run(nodes.connect_and_profile_node(state))
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

    @pytest.fixture(autouse=True)
    def _artifact_root_in_tmp(self, tmp_path, monkeypatch):
        """M8: clean_dataset's output must resolve inside
        settings.maeda_artifact_root/<run_id> to pass path validation --
        point that root at tmp_path so fixture files built under it validate."""
        from src.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "maeda_artifact_root", str(tmp_path))

    @staticmethod
    def _clean_output(tmp_path, pipeline_run_id, round_num, filename, content):
        """Builds a cleaned-output file at the path M8 validation expects
        for round `round_num` of a pipeline run -- mirrors nodes.py's
        `_round_run_id("clean", round_index + 1)` naming. round_num is
        1-indexed (the first clean call in a run is round 1)."""
        d = tmp_path / f"{pipeline_run_id}_clean{round_num}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / filename
        p.write_text(content)
        return str(p)

    def test_cleaned_path_adopted_and_report_refreshed(self, tmp_path):
        """The confirmed M7 bug: cleaned_path used to never be written back
        into data_sources/active_source, and data_quality_report stayed
        frozen at the pre-clean state -- so later rounds kept re-profiling
        and re-cleaning the *original* dirty file forever. Proves the fix
        end to end in a single round that converges."""
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("a,b\n1,\n2,\n3,\n4,\n5,1\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        cleaned = self._clean_output(
            tmp_path, state["run_id"], 1, "cleaned.csv", "a,b\n1,0\n2,0\n3,0\n4,0\n5,1\n",
        )

        dirty_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=True)
        clean_result = CleaningResult(cleaned_path=cleaned, changes_summary={"total_rounds": 1}, rows_affected=0)
        clean_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)
        validation = QualityValidation(passed=True, score=1.0, issues=[], details={})

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset")),
                              (clean_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
            validate_quality=[(validation, self._log("validate_quality"))],
        )

        result = self._run(state, mock_mcp)

        assert result["data_sources"][0]["path"] == cleaned
        assert result["active_source"]["path"] == cleaned
        assert result["data_quality_report"]["has_critical_issues"] is False
        assert result["cleaning_applied"] is True
        assert result["cleaning_stop_reason"] == "passed"
        assert result["iteration_count"] == 1

    def test_run_ids_distinct_and_execution_plan_traced(self, tmp_path):
        """M4: each MCP call in the round gets its own run_id (reusing one
        id across separate clean_dataset calls would make cleaner overwrite
        round 1's artifacts with round 2's -- see the module docstring on
        _round_run_id), all sharing the pipeline's run_id as a prefix; and
        the execution_plan cleaner returns lands in decision_trace."""
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("a,b\n1,\n2,\n3,\n4,\n5,1\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        pipeline_run_id = state["run_id"]
        cleaned = self._clean_output(
            tmp_path, pipeline_run_id, 1, "cleaned.csv", "a,b\n1,0\n2,0\n3,0\n4,0\n5,1\n",
        )

        dirty_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=True)
        clean_result = CleaningResult(
            cleaned_path=cleaned, changes_summary={}, rows_affected=0,
            execution_plan={
                "plan_id": "p1", "planner_mode_requested": "rule",
                "planner_mode_used": "rule", "planner_fallback_reason": None,
                "steps": [{"step_id": "s1"}, {"step_id": "s2"}],
            },
        )
        clean_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)
        validation = QualityValidation(passed=True, score=1.0, issues=[], details={})

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset")),
                              (clean_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
            validate_quality=[(validation, self._log("validate_quality"))],
        )

        result = self._run(state, mock_mcp)

        profile_run_ids = [c.kwargs.get("run_id") for c in mock_mcp.profile_dataset.call_args_list]
        clean_run_id = mock_mcp.clean_dataset.call_args.kwargs.get("run_id")
        validate_run_id = mock_mcp.validate_quality.call_args.kwargs.get("run_id")

        all_ids = [*profile_run_ids, clean_run_id, validate_run_id]
        assert all(rid.startswith(pipeline_run_id) for rid in all_ids), all_ids
        assert len(set(all_ids)) == len(all_ids), f"expected all distinct, got {all_ids}"

        plan_traces = [t for t in result["decision_trace"] if t["action"] == "clean_dataset_execution_plan"]
        assert len(plan_traces) == 1
        assert "planner_mode_used='rule'" in plan_traces[0]["reasoning"]
        assert "steps=2" in plan_traces[0]["reasoning"]

    def test_iteration_count_zero_when_nothing_to_clean(self, csv_file):
        """附录 B.5: the counter tracks *completed clean_dataset calls*, not
        node entries -- must stay 0 when cleaning was never triggered."""
        from src.mcp_client.models import DataQualityReport
        from src.state.graph_state import initial_state

        clean_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)
        mock_mcp = self._mock_mcp(profile_dataset=[(clean_report, self._log("profile_dataset"))])

        state = initial_state("q", data_sources=[{"type": "csv", "path": csv_file}])
        result = self._run(state, mock_mcp)

        assert result["iteration_count"] == 0
        assert result["cleaning_applied"] is False
        assert result.get("cleaning_stop_reason") is None

    def test_prefers_server_provided_hashes_over_self_hashing(self, tmp_path):
        """M8: when the server sends input_sha256/output_sha256 (live since
        cleaner's C3), those are authoritative -- even if MAEDA's own
        self-computed hash of the raw input would say otherwise (e.g. the
        input was Excel and got converted to CSV server-side first, so
        self-hashing the raw input isn't even measuring the same thing)."""
        from src.mcp_client.models import CleaningResult, DataQualityReport
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("a,b\n1,\n2,\n")  # self-hash of this will NOT match cleaned's self-hash
        dirty_report = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        cleaned = self._clean_output(tmp_path, state["run_id"], 1, "cleaned.csv", "a,b\n1,0\n2,0\n")
        clean_result = CleaningResult(
            cleaned_path=cleaned, changes_summary={}, rows_affected=0,
            input_sha256="same_hash", output_sha256="same_hash",  # server says: no diff
        )

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
        )

        result = self._run(state, mock_mcp)

        assert result["cleaning_stop_reason"] == "no_diff"
        mock_mcp.validate_quality.assert_not_awaited()

    def test_no_diff_stop_when_output_byte_identical(self, tmp_path):
        """附录 B.4 #2: cleaner "succeeding" but pointing right back at the
        same content must stop the loop rather than spin on a no-op round.
        Also proves validate_quality/re-profile are skipped in this case --
        an unconfigured side_effect on either would raise StopIteration."""
        from src.mcp_client.models import CleaningResult, DataQualityReport
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("a,b\n1,\n2,\n")
        dirty_report = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)
        clean_result = CleaningResult(cleaned_path=str(dirty), changes_summary={}, rows_affected=0)

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
        )

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        result = self._run(state, mock_mcp)

        assert result["cleaning_stop_reason"] == "no_diff"
        assert result["iteration_count"] == 1
        mock_mcp.validate_quality.assert_not_awaited()

    def test_cleaning_not_applied_when_output_unreadable(self, tmp_path):
        """附录 B: cleaning_applied=true only when the output file exists,
        is readable, and was actually adopted -- a cleaner response
        pointing at a missing file must not be treated as success."""
        from src.mcp_client.models import CleaningResult, DataQualityReport
        from src.state.graph_state import initial_state

        dirty = tmp_path / "dirty.csv"
        dirty.write_text("a,b\n1,\n2,\n")
        dirty_report = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)
        missing_output = str(tmp_path / "does_not_exist.csv")
        clean_result = CleaningResult(cleaned_path=missing_output, changes_summary={}, rows_affected=0)

        mock_mcp = self._mock_mcp(
            profile_dataset=[(dirty_report, self._log("profile_dataset"))],
            clean_dataset=[(clean_result, self._log("clean_dataset"))],
        )

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(dirty)}])
        result = self._run(state, mock_mcp)

        assert result["cleaning_applied"] is False
        assert result["cleaning_stop_reason"] == "no_diff"
        assert result["data_sources"][0]["path"] == str(dirty)  # unusable output never adopted

    def test_router_routes_ready_when_stop_reason_set_even_if_still_critical(self):
        """附录 B.3: a terminal stop_reason must win over has_critical_issues
        still being true -- "not silently ready" means attaching a caveat,
        not refusing to proceed to analysis."""
        from src.graph.router import route_after_profiling
        from src.state.graph_state import initial_state

        state = initial_state("q")
        state["data_quality_report"] = {"has_critical_issues": True}
        state["iteration_count"] = 1
        state["cleaning_stop_reason"] = "max_rounds"
        assert route_after_profiling(state) == "ready"

    def test_two_rounds_then_passes(self, tmp_path):
        """Drives the node twice, threading state through by hand the way
        the graph's cyclic edge would, proving round 2 correctly picks up
        round 1's adopted cleaned file rather than re-reading the original."""
        from src.graph.router import route_after_profiling
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        original = tmp_path / "v0.csv"
        original.write_text("a,b\n1,\n2,\n3,\n4,\n5,1\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(original)}])
        pipeline_run_id = state["run_id"]
        v1 = self._clean_output(tmp_path, pipeline_run_id, 1, "v1.csv", "a,b\n1,0\n2,\n3,\n4,\n5,1\n")
        v2 = self._clean_output(tmp_path, pipeline_run_id, 2, "v2.csv", "a,b\n1,0\n2,0\n3,0\n4,0\n5,1\n")

        still_critical = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=True)
        now_clean = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)

        validation1 = QualityValidation(
            passed=False, score=0.5, issues=["Missing values"],
            details={"mean_null_ratio": 0.2, "duplicate_row_ratio": 0.0, "schema_score": 1.0},
        )
        validation2 = QualityValidation(passed=True, score=1.0, issues=[], details={})

        mock_mcp = self._mock_mcp(
            profile_dataset=[
                (still_critical, self._log("profile_dataset")),  # round 1 entry
                (still_critical, self._log("profile_dataset")),  # round 1 re-profile
                (still_critical, self._log("profile_dataset")),  # round 2 entry (re-reads v1)
                (now_clean, self._log("profile_dataset")),        # round 2 re-profile
            ],
            clean_dataset=[
                (CleaningResult(cleaned_path=v1, changes_summary={"round": 1}, rows_affected=0),
                 self._log("clean_dataset")),
                (CleaningResult(cleaned_path=v2, changes_summary={"round": 2}, rows_affected=0),
                 self._log("clean_dataset")),
            ],
            validate_quality=[
                (validation1, self._log("validate_quality")),
                (validation2, self._log("validate_quality")),
            ],
        )

        state = self._run(state, mock_mcp)
        assert state.get("cleaning_stop_reason") is None  # not passed, nothing to compare yet -- keep looping
        assert state["iteration_count"] == 1
        assert state["data_sources"][0]["path"] == v1
        assert route_after_profiling(state) == "clean"

        state = self._run(state, mock_mcp)
        assert state["cleaning_stop_reason"] == "passed"
        assert state["iteration_count"] == 2
        assert state["data_sources"][0]["path"] == v2
        assert route_after_profiling(state) == "ready"

    def test_no_improvement_stops_when_signature_repeats(self, tmp_path):
        """附录 B.4 #3: an identical three-dimension verdict two rounds
        running means the loop is spinning, not converging -- must stop
        even though has_critical_issues is still true."""
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        v0 = tmp_path / "v0.csv"; v0.write_text("a\n1\n2\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(v0)}])
        pipeline_run_id = state["run_id"]
        v1 = self._clean_output(tmp_path, pipeline_run_id, 1, "v1.csv", "a\n1\n3\n")
        v2 = self._clean_output(tmp_path, pipeline_run_id, 2, "v2.csv", "a\n1\n4\n")

        still_critical = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)
        same_signature = {"mean_null_ratio": 0.2, "duplicate_row_ratio": 0.0, "schema_score": 1.0}

        mock_mcp = self._mock_mcp(
            profile_dataset=[(still_critical, self._log("profile_dataset"))] * 4,
            clean_dataset=[
                (CleaningResult(cleaned_path=v1, changes_summary={}, rows_affected=0), self._log("clean_dataset")),
                (CleaningResult(cleaned_path=v2, changes_summary={}, rows_affected=0), self._log("clean_dataset")),
            ],
            validate_quality=[
                (QualityValidation(passed=False, score=0.5, issues=["x"], details=same_signature),
                 self._log("validate_quality")),
                (QualityValidation(passed=False, score=0.5, issues=["x"], details=same_signature),
                 self._log("validate_quality")),
            ],
        )

        state = self._run(state, mock_mcp)
        assert state.get("cleaning_stop_reason") is None

        state = self._run(state, mock_mcp)
        assert state["cleaning_stop_reason"] == "no_improvement"
        assert state["iteration_count"] == 2

    def test_regressed_stops_and_flags_for_review(self, tmp_path):
        """附录 B.4 #4: quality getting worse round-over-round must stop
        the loop, not just plow ahead hoping the next round helps."""
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        v0 = tmp_path / "v0.csv"; v0.write_text("a\n1\n2\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(v0)}])
        pipeline_run_id = state["run_id"]
        v1 = self._clean_output(tmp_path, pipeline_run_id, 1, "v1.csv", "a\n1\n3\n")
        v2 = self._clean_output(tmp_path, pipeline_run_id, 2, "v2.csv", "a\n1\n4\n")

        still_critical = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)

        mock_mcp = self._mock_mcp(
            profile_dataset=[(still_critical, self._log("profile_dataset"))] * 4,
            clean_dataset=[
                (CleaningResult(cleaned_path=v1, changes_summary={}, rows_affected=0), self._log("clean_dataset")),
                (CleaningResult(cleaned_path=v2, changes_summary={}, rows_affected=0), self._log("clean_dataset")),
            ],
            validate_quality=[
                (QualityValidation(passed=False, score=0.6, issues=["x"],
                                    details={"mean_null_ratio": 0.2, "duplicate_row_ratio": 0.0, "schema_score": 1.0}),
                 self._log("validate_quality")),
                # Different signature (schema now flagged too) so
                # no_improvement doesn't preempt the regression check.
                (QualityValidation(passed=False, score=0.3, issues=["x", "y"],
                                    details={"mean_null_ratio": 0.2, "duplicate_row_ratio": 0.0, "schema_score": 0.5}),
                 self._log("validate_quality")),
            ],
        )

        state = self._run(state, mock_mcp)
        state = self._run(state, mock_mcp)

        assert state["cleaning_stop_reason"] == "regressed"
        assert "0.6" in state["cleaning_caveat"] and "0.3" in state["cleaning_caveat"]

    def test_max_rounds_reached_after_configured_cap(self, tmp_path):
        """Regression guard for the pre-M7 off-by-one: iteration_count now
        means 'completed clean_dataset calls' (附录 B.5), so the loop must
        run exactly _MAX_CLEAN_ITERATIONS clean rounds, not one fewer."""
        from src.graph.router import _MAX_CLEAN_ITERATIONS, route_after_profiling
        from src.mcp_client.models import (
            CleaningResult, DataQualityReport, QualityValidation,
        )
        from src.state.graph_state import initial_state

        original = tmp_path / "v0.csv"
        original.write_text("a,b\n1,\n2,0\n")

        state = initial_state("q", data_sources=[{"type": "csv", "path": str(original)}])
        pipeline_run_id = state["run_id"]
        # cleaned_files[i] is the output of round i+1 (1-indexed, matching
        # _round_run_id's "clean{round}" naming).
        cleaned_files = [
            self._clean_output(tmp_path, pipeline_run_id, i + 1, f"v{i + 1}.csv", f"a,b\n1,\n2,{i + 1}\n")
            for i in range(_MAX_CLEAN_ITERATIONS)
        ]

        still_critical = DataQualityReport(row_count=2, columns=[], quality_issues=[], has_critical_issues=True)
        profile_responses = [(still_critical, self._log("profile_dataset"))
                              for _ in range(2 * _MAX_CLEAN_ITERATIONS)]
        clean_responses = [
            (CleaningResult(cleaned_path=cleaned_files[i], changes_summary={}, rows_affected=0),
             self._log("clean_dataset"))
            for i in range(_MAX_CLEAN_ITERATIONS)
        ]
        # Alternate the "duplicate" dimension each round so consecutive
        # signatures differ (avoids tripping no_improvement); keep score
        # flat so it never decreases (avoids tripping regressed) --
        # isolates the max_rounds path specifically.
        validate_responses = [
            (QualityValidation(
                passed=False, score=0.5, issues=["x"],
                details={"mean_null_ratio": 0.2, "schema_score": 1.0,
                          "duplicate_row_ratio": 0.06 if i % 2 == 0 else 0.0},
             ), self._log("validate_quality"))
            for i in range(_MAX_CLEAN_ITERATIONS)
        ]

        mock_mcp = self._mock_mcp(
            profile_dataset=profile_responses,
            clean_dataset=clean_responses,
            validate_quality=validate_responses,
        )

        for _ in range(_MAX_CLEAN_ITERATIONS):
            state = self._run(state, mock_mcp)

        assert state["iteration_count"] == _MAX_CLEAN_ITERATIONS
        assert state["cleaning_stop_reason"] == "max_rounds"
        assert "maximum" in state["cleaning_caveat"]
        assert route_after_profiling(state) == "ready"


# ─── 4.6 Source Registry ─────────────────────────────────────────────────────

class TestSourceRegistry:
    def test_register_and_retrieve(self):
        reg = SourceRegistry()
        reg.register({"name": "sales", "type": "csv", "path": "/data/sales.csv"})
        assert "sales" in reg
        assert reg.get("sales")["path"] == "/data/sales.csv"

    def test_auto_name_from_path(self):
        reg = SourceRegistry()
        name = reg.register({"type": "csv", "path": "/data/customers.csv"})
        assert name == "customers"

    def test_first_registered_becomes_active(self):
        reg = SourceRegistry()
        reg.register({"name": "a", "type": "csv", "path": "/a.csv"})
        reg.register({"name": "b", "type": "csv", "path": "/b.csv"})
        assert reg.active["name"] == "a"

    def test_set_active(self):
        reg = SourceRegistry()
        reg.register({"name": "a", "type": "csv", "path": "/a.csv"})
        reg.register({"name": "b", "type": "csv", "path": "/b.csv"})
        reg.set_active("b")
        assert reg.active["name"] == "b"

    def test_set_active_unknown_raises(self):
        reg = SourceRegistry()
        with pytest.raises(KeyError):
            reg.set_active("nonexistent")

    def test_update_source(self):
        reg = SourceRegistry()
        reg.register({"name": "ds", "type": "csv", "path": "/x.csv"})
        reg.update("ds", {"schema": {"row_count": 100}})
        assert reg.get("ds")["schema"]["row_count"] == 100

    def test_register_many(self):
        reg = SourceRegistry()
        sources = [
            {"name": "a", "type": "csv", "path": "/a.csv"},
            {"name": "b", "type": "json", "path": "/b.json"},
        ]
        names = reg.register_many(sources)
        assert names == ["a", "b"]
        assert len(reg) == 2

    def test_to_state_list(self):
        reg = SourceRegistry()
        reg.register({"name": "x", "type": "csv", "path": "/x.csv"})
        lst = reg.to_state_list()
        assert len(lst) == 1
        assert lst[0]["name"] == "x"

    def test_infer_name_from_path(self):
        assert _infer_name("/data/sales_2024.csv") == "sales_2024"
        assert _infer_name("postgresql://host/mydb") == "mydb"
        assert _infer_name("") == "source"

    def test_multi_source_queries(self):
        """Registry supports registering and switching between multiple sources."""
        reg = SourceRegistry()
        reg.register_many([
            {"name": "orders",    "type": "csv",  "path": "/orders.csv"},
            {"name": "customers", "type": "csv",  "path": "/customers.csv"},
            {"name": "products",  "type": "sql",  "path": "sqlite:///prod.db"},
        ])
        assert len(reg) == 3
        reg.set_active("products")
        assert reg.active["type"] == "sql"
        all_s = reg.all_sources()
        assert len(all_s) == 3


# ─── SchemaInfo helpers ───────────────────────────────────────────────────────

class TestSchemaInfo:
    def test_to_dict_has_all_keys(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        d = schema.to_dict()
        for key in ["source_type", "path", "row_count", "columns", "preview"]:
            assert key in d

    def test_column_info_to_dict(self, csv_file):
        df = connect_csv(csv_file)
        schema = extract_schema(df, "csv", csv_file)
        for col in schema.columns:
            d = col.to_dict()
            assert "name" in d
            assert "dtype" in d
            assert "null_pct" in d
            assert "sample_values" in d
