"""
Phase 3 tests — MCP Integration Layer.
All HTTP calls are mocked; no live servers required.
Run with: pytest tests/unit/test_phase3.py -v
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_client.client import MCPClient, MCPConnectionError, MCPToolError
from src.mcp_client.data_cleaner import DataCleanerClient
from src.mcp_client.fallback import SubSystemWithFallback, _basic_pandas_profile
from src.mcp_client.models import (
    CleaningPlan,
    CleaningResult,
    CleaningStep,
    Collection,
    ColumnProfile,
    DataQualityReport,
    QualityValidation,
    RAGChunk,
    SubSystemHealth,
)
from src.mcp_client.rag_server import RAGServerClient


# ─── 3.6 Response model parsing ───────────────────────────────────────────────

class TestDataQualityReport:
    def test_from_mcp_response_full(self):
        raw = {
            "row_count": 1000,
            "columns": [
                {"name": "revenue", "dtype": "float64", "null_pct": 0.05,
                 "unique_count": 800, "sample_values": [100.0, 200.0]},
            ],
            "quality_issues": [{"severity": "warning", "issue": "skew"}],
        }
        report = DataQualityReport.from_mcp_response(raw)
        assert report.row_count == 1000
        assert len(report.columns) == 1
        assert report.columns[0].name == "revenue"
        assert report.has_critical_issues is False

    def test_critical_issue_detection(self):
        raw = {
            "row_count": 500,
            "columns": [],
            "quality_issues": [{"severity": "critical", "issue": "duplicate_pk"}],
        }
        report = DataQualityReport.from_mcp_response(raw)
        assert report.has_critical_issues is True

    def test_to_dict_roundtrip(self):
        raw = {
            "row_count": 100,
            "columns": [{"name": "x", "dtype": "int64", "null_pct": 0.0,
                          "unique_count": 10, "sample_values": [1, 2]}],
            "quality_issues": [],
        }
        report = DataQualityReport.from_mcp_response(raw)
        d = report.to_dict()
        assert d["row_count"] == 100
        assert d["columns"][0]["name"] == "x"

    def test_empty_response_defaults(self):
        report = DataQualityReport.from_mcp_response({})
        assert report.row_count == 0
        assert report.columns == []
        assert report.has_critical_issues is False


class TestRAGChunk:
    def test_from_mcp_response(self):
        raw = {"content": "Domain knowledge.", "score": 0.92,
               "source_file": "guide.pdf", "page": 3, "chunk_id": "c_001"}
        chunk = RAGChunk.from_mcp_response(raw)
        assert chunk.content == "Domain knowledge."
        assert chunk.score == 0.92
        assert chunk.source_file == "guide.pdf"

    def test_to_dict(self):
        chunk = RAGChunk(content="text", score=0.8, source_file="doc.pdf", page=1, chunk_id="x")
        d = chunk.to_dict()
        assert d["content"] == "text"
        assert d["score"] == 0.8


class TestCleaningPlan:
    def test_from_mcp_response(self):
        raw = {"steps": [{"operation": "fill_null", "target_column": "age",
                           "rationale": "50% null", "estimated_impact": "high"}]}
        plan = CleaningPlan.from_mcp_response(raw)
        assert len(plan.steps) == 1
        assert plan.steps[0].operation == "fill_null"

    def test_empty_plan(self):
        plan = CleaningPlan.from_mcp_response({})
        assert plan.steps == []


# ─── 3.1 MCPClient transport ─────────────────────────────────────────────────

class TestMCPClient:
    def test_call_tool_success(self):
        client = MCPClient("http://localhost:8001")
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {"jsonrpc": "2.0", "id": 1, "result": {"row_count": 500}},
        ))
        client._client = mock_http

        result = asyncio.run(client.call_tool("profile_dataset", {"path": "/data/sales.csv"}))
        assert result["row_count"] == 500

    def test_call_tool_mcp_content_block(self):
        """MCPClient unwraps content-block responses correctly."""
        client = MCPClient("http://localhost:8001")
        mock_http = AsyncMock()
        mock_http.is_closed = False
        inner = json.dumps({"row_count": 200})
        mock_http.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text", "text": inner}]}
            },
        ))
        client._client = mock_http

        result = asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))
        assert result["row_count"] == 200

    def test_call_tool_connection_error(self):
        import httpx
        client = MCPClient("http://localhost:8001", max_retries=1)
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client._client = mock_http

        with pytest.raises(MCPConnectionError):
            asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))

    def test_call_tool_server_error_response(self):
        client = MCPClient("http://localhost:8001", max_retries=1)
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(return_value=MagicMock(
            status_code=500,
            text="Internal server error",
        ))
        client._client = mock_http

        with pytest.raises(MCPConnectionError):
            asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))

    def test_call_tool_rpc_error(self):
        client = MCPClient("http://localhost:8001", max_retries=1)
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {"jsonrpc": "2.0", "id": 1,
                          "error": {"code": -32601, "message": "Method not found"}},
        ))
        client._client = mock_http

        with pytest.raises(MCPToolError):
            asyncio.run(client.call_tool("nonexistent_tool", {}))

    def test_health_check_ok(self):
        client = MCPClient("http://localhost:8001")
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
        client._client = mock_http

        ok, latency = asyncio.run(client.health_check())
        assert ok is True
        assert latency is not None and latency >= 0

    def test_health_check_failure(self):
        import httpx
        client = MCPClient("http://localhost:8001")
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client._client = mock_http

        ok, latency = asyncio.run(client.health_check())
        assert ok is False
        assert latency is None


# ─── 3.2 DataCleanerClient ────────────────────────────────────────────────────

def _mock_dc_transport(tool_responses: dict) -> MCPClient:
    transport = MagicMock(spec=MCPClient)
    async def call_tool(name, args):
        return tool_responses[name]
    transport.call_tool = AsyncMock(side_effect=call_tool)
    return transport


class TestDataCleanerClient:
    def test_profile_dataset(self):
        transport = _mock_dc_transport({
            "profile_dataset": {
                "row_count": 300,
                "columns": [{"name": "col1", "dtype": "str", "null_pct": 0.0,
                              "unique_count": 50, "sample_values": ["a"]}],
                "quality_issues": [],
            }
        })
        client = DataCleanerClient(transport)
        report = asyncio.run(client.profile_dataset("/data/test.csv"))
        assert isinstance(report, DataQualityReport)
        assert report.row_count == 300

    def test_get_cleaning_plan(self):
        transport = _mock_dc_transport({
            "get_cleaning_plan": {
                "steps": [{"operation": "drop_duplicates", "target_column": "id",
                            "rationale": "duplicates found", "estimated_impact": "medium"}]
            }
        })
        client = DataCleanerClient(transport)
        plan = asyncio.run(client.get_cleaning_plan("/data/test.csv"))
        assert isinstance(plan, CleaningPlan)
        assert plan.steps[0].operation == "drop_duplicates"

    def test_clean_dataset(self):
        transport = _mock_dc_transport({
            "clean_dataset": {
                "cleaned_path": "/data/test_clean.csv",
                "changes_summary": "Dropped 10 duplicates",
                "rows_affected": 10,
            }
        })
        client = DataCleanerClient(transport)
        result = asyncio.run(client.clean_dataset("/data/test.csv"))
        assert isinstance(result, CleaningResult)
        assert result.rows_affected == 10

    def test_validate_quality(self):
        transport = _mock_dc_transport({
            "validate_quality": {"passed": True, "score": 0.97, "issues": []}
        })
        client = DataCleanerClient(transport)
        validation = asyncio.run(client.validate_quality("/data/test_clean.csv"))
        assert isinstance(validation, QualityValidation)
        assert validation.passed is True
        assert validation.score == 0.97


# ─── 3.3 RAGServerClient ─────────────────────────────────────────────────────

def _mock_rag_transport(tool_responses: dict) -> MCPClient:
    transport = MagicMock(spec=MCPClient)
    async def call_tool(name, args):
        return tool_responses[name]
    transport.call_tool = AsyncMock(side_effect=call_tool)
    return transport


class TestRAGServerClient:
    def test_retrieve(self):
        transport = _mock_rag_transport({
            "retrieve": {
                "chunks": [{"content": "Domain context.", "score": 0.88}]
            }
        })
        client = RAGServerClient(transport)
        chunks = asyncio.run(client.retrieve("revenue trends", top_k=3))
        assert len(chunks) == 1
        assert chunks[0].content == "Domain context."

    def test_retrieve_with_metadata(self):
        transport = _mock_rag_transport({
            "retrieve_with_metadata": {
                "chunks": [
                    {"content": "Enterprise insight.", "score": 0.95,
                     "source_file": "guide.pdf", "page": 5, "chunk_id": "c_42"}
                ]
            }
        })
        client = RAGServerClient(transport)
        chunks = asyncio.run(client.retrieve_with_metadata("market analysis", top_k=5))
        assert chunks[0].source_file == "guide.pdf"
        assert chunks[0].page == 5

    def test_list_collections(self):
        transport = _mock_rag_transport({
            "list_collections": {
                "collections": [
                    {"name": "industry_reports", "doc_count": 42, "description": "Annual reports"}
                ]
            }
        })
        client = RAGServerClient(transport)
        collections = asyncio.run(client.list_collections())
        assert len(collections) == 1
        assert isinstance(collections[0], Collection)
        assert collections[0].name == "industry_reports"


# ─── 3.4 Fallback layer ───────────────────────────────────────────────────────

def _build_fallback(
    dc_side_effect=None, rag_side_effect=None,
    dc_responses=None, rag_responses=None,
) -> SubSystemWithFallback:
    dc_transport = MagicMock(spec=MCPClient)
    rag_transport = MagicMock(spec=MCPClient)

    if dc_side_effect:
        dc_transport.call_tool = AsyncMock(side_effect=dc_side_effect)
    elif dc_responses:
        async def dc_call(name, args):
            return dc_responses[name]
        dc_transport.call_tool = AsyncMock(side_effect=dc_call)

    if rag_side_effect:
        rag_transport.call_tool = AsyncMock(side_effect=rag_side_effect)
    elif rag_responses:
        async def rag_call(name, args):
            return rag_responses[name]
        rag_transport.call_tool = AsyncMock(side_effect=rag_call)

    dc_transport.health_check = AsyncMock(return_value=(True, 5.0))
    rag_transport.health_check = AsyncMock(return_value=(True, 8.0))

    dc = DataCleanerClient(dc_transport)
    rag = RAGServerClient(rag_transport)
    dc._transport = dc_transport
    rag._transport = rag_transport
    return SubSystemWithFallback(dc, rag)


class TestSubSystemWithFallback:
    def test_profile_dataset_success(self):
        fb = _build_fallback(dc_responses={
            "profile_dataset": {"row_count": 500, "columns": [], "quality_issues": []}
        })
        report, log = asyncio.run(fb.profile_dataset("/data/test.csv"))
        assert report.row_count == 500
        assert log["mode"] == "mcp"
        assert log["tool"] == "profile_dataset"

    def test_profile_dataset_fallback(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        report, log = asyncio.run(fb.profile_dataset(str(csv)))
        assert log["mode"] == "fallback"
        assert log["error"] is not None
        assert report.row_count == 2

    def test_retrieve_knowledge_success(self):
        fb = _build_fallback(rag_responses={
            "retrieve_with_metadata": {
                "chunks": [{"content": "ctx", "score": 0.9,
                             "source_file": "doc.pdf", "page": 1, "chunk_id": "c1"}]
            }
        })
        chunks, log = asyncio.run(fb.retrieve_knowledge("query"))
        assert len(chunks) == 1
        assert log["mode"] == "mcp"

    def test_retrieve_knowledge_fallback(self):
        fb = _build_fallback(rag_side_effect=MCPConnectionError("offline"))
        chunks, log = asyncio.run(fb.retrieve_knowledge("query"))
        assert chunks == []
        assert log["mode"] == "fallback"

    def test_get_cleaning_plan_fallback(self):
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        plan, log = asyncio.run(fb.get_cleaning_plan("/data/x.csv"))
        assert isinstance(plan, CleaningPlan)
        assert plan.steps == []
        assert log["mode"] == "fallback"

    def test_clean_dataset_fallback_returns_original_path(self):
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        result, log = asyncio.run(fb.clean_dataset("/data/original.csv"))
        assert result.cleaned_path == "/data/original.csv"
        assert log["mode"] == "fallback"

    def test_validate_quality_fallback_passes(self):
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        validation, log = asyncio.run(fb.validate_quality("/data/x.csv"))
        assert validation.passed is True
        assert log["mode"] == "fallback"


# ─── 3.5 MCP call logging ────────────────────────────────────────────────────

class TestMCPCallLogging:
    def test_log_has_required_fields(self):
        fb = _build_fallback(dc_responses={
            "profile_dataset": {"row_count": 10, "columns": [], "quality_issues": []}
        })
        _, log = asyncio.run(fb.profile_dataset("/data/x.csv"))
        for field in ["system", "tool", "args", "duration_ms", "mode"]:
            assert field in log, f"Missing log field: {field}"

    def test_log_records_timing(self):
        fb = _build_fallback(dc_responses={
            "profile_dataset": {"row_count": 1, "columns": [], "quality_issues": []}
        })
        _, log = asyncio.run(fb.profile_dataset("/data/x.csv"))
        assert isinstance(log["duration_ms"], float)
        assert log["duration_ms"] >= 0

    def test_fallback_log_contains_error(self):
        fb = _build_fallback(rag_side_effect=MCPConnectionError("timed out"))
        _, log = asyncio.run(fb.retrieve_knowledge("test"))
        assert log["error"] is not None
        assert "timed out" in log["error"]


# ─── 3.7 Health check ────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_check_both_available(self):
        dc_transport = MagicMock(spec=MCPClient)
        rag_transport = MagicMock(spec=MCPClient)
        dc_transport.health_check = AsyncMock(return_value=(True, 12.5))
        rag_transport.health_check = AsyncMock(return_value=(True, 8.0))
        dc = DataCleanerClient(dc_transport)
        rag = RAGServerClient(rag_transport)
        dc._transport = dc_transport
        rag._transport = rag_transport
        fb = SubSystemWithFallback(dc, rag)

        health = asyncio.run(fb.health_check())
        assert isinstance(health, SubSystemHealth)
        assert health.data_cleaner_available is True
        assert health.rag_server_available is True
        assert health.data_cleaner_latency_ms == 12.5

    def test_health_check_both_down(self):
        import httpx
        dc_transport = MagicMock(spec=MCPClient)
        rag_transport = MagicMock(spec=MCPClient)
        dc_transport.health_check = AsyncMock(side_effect=httpx.ConnectError("refused"))
        rag_transport.health_check = AsyncMock(side_effect=httpx.ConnectError("refused"))
        dc = DataCleanerClient(dc_transport)
        rag = RAGServerClient(rag_transport)
        dc._transport = dc_transport
        rag._transport = rag_transport
        fb = SubSystemWithFallback(dc, rag)

        health = asyncio.run(fb.health_check())
        assert health.data_cleaner_available is False
        assert health.rag_server_available is False
        assert health.any_available is False

    def test_sub_system_health_to_dict(self):
        health = SubSystemHealth(
            data_cleaner_available=True, rag_server_available=False,
            data_cleaner_latency_ms=10.0, rag_server_latency_ms=None
        )
        d = health.to_dict()
        assert d["data_cleaner_available"] is True
        assert d["rag_server_available"] is False


# ─── Pandas fallback profiler ────────────────────────────────────────────────

class TestBasicPandasProfile:
    def test_profiles_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("name,age,score\nAlice,30,88.5\nBob,,92.0\nCarol,25,\n")
        report = _basic_pandas_profile(str(csv))
        assert report.row_count == 3
        col_names = {c.name for c in report.columns}
        assert "name" in col_names
        assert "age" in col_names

    def test_null_warning_on_high_null_rate(self, tmp_path):
        csv = tmp_path / "nulls.csv"
        # 80% nulls in column b
        rows = "a,b\n1,\n2,\n3,\n4,\n5,1\n"
        csv.write_text(rows)
        report = _basic_pandas_profile(str(csv))
        issues = [i for i in report.quality_issues if i["column"] == "b"]
        assert any(i["issue"] == "high_null_rate" for i in issues)

    def test_unreadable_file_returns_empty_report(self):
        report = _basic_pandas_profile("/nonexistent/path/data.csv")
        assert report.row_count == 0
        assert report.columns == []
        assert report.has_critical_issues is False


# ─── 3.8 Integration: three-service chain ────────────────────────────────────

class TestMCPIntegration:
    """
    Simulates the full MAEDA ↔ Data Cleaner ↔ RAG chain using mocked transports.
    Validates that state["mcp_call_log"] is correctly populated end-to-end.
    """

    def test_full_chain_populates_mcp_call_log(self):
        from src.state.graph_state import initial_state

        dc_responses = {
            "profile_dataset": {
                "row_count": 200, "columns": [
                    {"name": "revenue", "dtype": "float64", "null_pct": 0.0,
                     "unique_count": 150, "sample_values": [1000.0]}
                ],
                "quality_issues": [],
            }
        }
        rag_responses = {
            "retrieve_with_metadata": {
                "chunks": [{"content": "Market context.", "score": 0.87,
                             "source_file": "report.pdf", "page": 2, "chunk_id": "r1"}]
            }
        }
        fb = _build_fallback(dc_responses=dc_responses, rag_responses=rag_responses)

        state = initial_state("Show revenue trend", data_sources=[{"path": "/data/revenue.csv"}])

        # Simulate data profiling
        report, prof_log = asyncio.run(fb.profile_dataset("/data/revenue.csv"))
        state["mcp_call_log"] = [prof_log]
        state["data_quality_report"] = report.to_dict()

        # Simulate RAG retrieval
        chunks, rag_log = asyncio.run(fb.retrieve_knowledge("revenue trend"))
        state["mcp_call_log"].append(rag_log)
        state["rag_context"] = [c.to_dict() for c in chunks]

        assert len(state["mcp_call_log"]) == 2
        assert state["mcp_call_log"][0]["system"] == "data_cleaner"
        assert state["mcp_call_log"][1]["system"] == "rag_server"
        assert state["data_quality_report"]["row_count"] == 200
        assert len(state["rag_context"]) == 1

    def test_full_chain_works_with_both_subsystems_down(self, tmp_path):
        """MAEDA runs standalone when both sub-systems are offline."""
        from src.state.graph_state import initial_state

        csv = tmp_path / "data.csv"
        csv.write_text("revenue,region\n100,North\n200,South\n")

        fb = _build_fallback(
            dc_side_effect=MCPConnectionError("offline"),
            rag_side_effect=MCPConnectionError("offline"),
        )
        state = initial_state("Show revenue", data_sources=[{"path": str(csv)}])

        report, prof_log = asyncio.run(fb.profile_dataset(str(csv)))
        state["mcp_call_log"] = [prof_log]
        state["data_quality_report"] = report.to_dict()

        chunks, rag_log = asyncio.run(fb.retrieve_knowledge("revenue"))
        state["mcp_call_log"].append(rag_log)
        state["rag_context"] = [c.to_dict() for c in chunks]

        # Profiling fell back to pandas
        assert state["data_quality_report"]["row_count"] == 2
        assert state["mcp_call_log"][0]["mode"] == "fallback"
        # RAG returned empty
        assert state["rag_context"] == []
        assert state["mcp_call_log"][1]["mode"] == "fallback"
        # No exceptions — MAEDA ran standalone
