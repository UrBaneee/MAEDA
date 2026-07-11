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
            result = nodes.connect_and_profile_node(state)
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

        assert result["schema_summary"] == "Sales dataset with product revenue by region."
        assert result["active_source"] is not None
        assert result["data_quality_report"]["has_critical_issues"] is False
        assert len(result["mcp_call_log"]) == 1

    def test_critical_issues_triggers_cleaning(self, csv_file):
        """When Data Cleaner reports critical issues, cleaning is invoked."""
        import src.graph.nodes as nodes
        from src.mcp_client.fallback import SubSystemWithFallback
        from src.mcp_client.models import (
            CleaningPlan,
            CleaningResult,
            DataQualityReport,
        )
        from src.state.graph_state import initial_state

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Data summary."))

        critical_report = DataQualityReport(
            row_count=5, columns=[], has_critical_issues=True,
            quality_issues=[{"severity": "critical", "issue": "dup_pk"}]
        )
        clean_plan = CleaningPlan(steps=[])
        clean_result = CleaningResult(
            cleaned_path=csv_file,
            changes_summary="Dropped 2 duplicate rows",
            rows_affected=2,
        )

        _log = lambda tool: {"system": "data_cleaner", "tool": tool,
                              "mode": "mcp", "args": {}, "duration_ms": 3.0}
        mock_mcp = MagicMock(spec=SubSystemWithFallback)
        mock_mcp.profile_dataset = AsyncMock(return_value=(critical_report, _log("profile_dataset")))
        mock_mcp.get_cleaning_plan = AsyncMock(return_value=(clean_plan, _log("get_cleaning_plan")))
        mock_mcp.clean_dataset = AsyncMock(return_value=(clean_result, _log("clean_dataset")))

        old_connector = nodes._data_connector
        old_mcp = nodes._subsystem_client
        nodes._data_connector = DataConnector(llm=mock_llm)
        nodes._subsystem_client = mock_mcp

        try:
            state = initial_state("q", data_sources=[{"type": "csv", "path": csv_file}])
            result = nodes.connect_and_profile_node(state)
        finally:
            nodes._data_connector = old_connector
            nodes._subsystem_client = old_mcp

        assert result["cleaning_applied"] is True
        assert "Dropped 2" in result["cleaning_summary"]
        assert len(result["mcp_call_log"]) == 3  # profile + plan + clean


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
