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

    def test_critical_issue_detection_reads_field_directly(self):
        """has_critical_issues comes straight from the cleaner's own verdict
        (附录 B.1, live since C1) — not derived from quality_issues, which
        is why it must not crash on the real string-list shape below."""
        raw = {
            "row_count": 500,
            "columns": [],
            "quality_issues": [],
            "has_critical_issues": True,
        }
        report = DataQualityReport.from_mcp_response(raw)
        assert report.has_critical_issues is True

    def test_quality_issues_real_shape_is_string_list(self):
        """The real cleaner (agentic-data-cleaner-v2 mcp_app.py profile_dataset)
        returns quality_issues as list[str] issue codes, not list[dict] with
        a "severity" key. Parsing this used to crash on `i.get("severity")`
        (M2) -- this is the regression test for that fix."""
        raw = {
            "row_count": 500,
            "columns": [],
            "quality_issues": ["schema_inconsistency", "type_inconsistency"],
            "has_critical_issues": True,
            "quality_contract_version": "1",
        }
        report = DataQualityReport.from_mcp_response(raw)
        assert [i.code for i in report.quality_issues] == [
            "schema_inconsistency", "type_inconsistency",
        ]
        assert all(i.source == "cleaner" for i in report.quality_issues)
        assert report.quality_contract_version == "1"

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
#
# MCPClient speaks the real MCP streamable-http protocol via the official
# `mcp` SDK (initialize handshake + session-scoped tool calls), so these
# tests mock at the SDK boundary (streamable_http_client + ClientSession)
# rather than at the old raw-httpx-POST level.

from contextlib import asynccontextmanager

from mcp.types import CallToolResult, TextContent


class _FakeSession:
    """Stands in for mcp.ClientSession — supports `async with`."""

    def __init__(self, call_tool_result=None, call_tool_side_effect=None,
                 list_tools_side_effect=None, initialize_side_effect=None):
        self.initialize = AsyncMock(side_effect=initialize_side_effect)
        self.call_tool = AsyncMock(
            return_value=call_tool_result, side_effect=call_tool_side_effect,
        )
        self.list_tools = AsyncMock(side_effect=list_tools_side_effect)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _patch_mcp_sdk(session: _FakeSession):
    """
    Patches src.mcp_client.client.streamable_http_client (an async context
    manager yielding (read, write, get_session_id)) and .ClientSession
    (constructed with (read, write), used as an async context manager) so
    call_tool()/health_check() drive `session` without a real network call.
    """
    @asynccontextmanager
    async def fake_streamable_http(*args, **kwargs):
        yield (MagicMock(), MagicMock(), MagicMock())

    return (
        patch("src.mcp_client.client.streamable_http_client", fake_streamable_http),
        patch("src.mcp_client.client.ClientSession", return_value=session),
    )


class TestMCPClient:
    def test_call_tool_success(self):
        client = MCPClient("http://localhost:8001")
        session = _FakeSession(call_tool_result=CallToolResult(
            content=[], structuredContent={"row_count": 500}, isError=False,
        ))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            result = asyncio.run(client.call_tool("profile_dataset", {"path": "/data/sales.csv"}))
        assert result["row_count"] == 500
        session.initialize.assert_awaited_once()

    def test_call_tool_text_content_block(self):
        """MCPClient falls back to parsing a text content block as JSON when
        structuredContent isn't populated."""
        client = MCPClient("http://localhost:8001")
        inner = json.dumps({"row_count": 200})
        session = _FakeSession(call_tool_result=CallToolResult(
            content=[TextContent(type="text", text=inner)],
            structuredContent=None, isError=False,
        ))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            result = asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))
        assert result["row_count"] == 200

    def test_call_tool_connection_error(self):
        client = MCPClient("http://localhost:8001", max_retries=1)
        session = _FakeSession(call_tool_side_effect=ConnectionRefusedError("refused"))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPConnectionError):
                asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))

    def test_call_tool_handshake_failure(self):
        """A failed initialize() handshake (e.g. protocol/session negotiation
        error) must surface as MCPConnectionError, not propagate raw."""
        client = MCPClient("http://localhost:8001", max_retries=1)
        session = _FakeSession(initialize_side_effect=RuntimeError("400 Missing session ID"))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPConnectionError):
                asyncio.run(client.call_tool("profile_dataset", {"path": "x"}))

    def test_call_tool_rpc_error(self):
        client = MCPClient("http://localhost:8001", max_retries=1)
        session = _FakeSession(call_tool_result=CallToolResult(
            content=[TextContent(type="text", text="Method not found")],
            structuredContent=None, isError=True,
        ))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPToolError):
                asyncio.run(client.call_tool("nonexistent_tool", {}))

    def test_health_check_ok(self):
        client = MCPClient("http://localhost:8001")
        session = _FakeSession()
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            ok, latency = asyncio.run(client.health_check())
        assert ok is True
        assert latency is not None and latency >= 0
        session.list_tools.assert_awaited_once()

    def test_health_check_failure(self):
        client = MCPClient("http://localhost:8001")
        session = _FakeSession(initialize_side_effect=ConnectionRefusedError("refused"))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            ok, latency = asyncio.run(client.health_check())
        assert ok is False
        assert latency is None

    # ── M6: retry predicate + deadline ──────────────────────────────────────

    def test_tool_error_is_not_retried(self):
        """错误处理矩阵: schema/contract/auth/data-input errors are not
        transient — retrying a tool-level error just wastes the deadline
        arriving at the same failure. Only MCPConnectionError is retryable."""
        client = MCPClient("http://localhost:8001", max_retries=2)
        session = _FakeSession(call_tool_result=CallToolResult(
            content=[TextContent(type="text", text="bad request")],
            structuredContent=None, isError=True,
        ))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPToolError):
                asyncio.run(client.call_tool("profile_dataset", {}))
        assert session.call_tool.call_count == 1  # no retry

    def test_connection_error_retried_up_to_max_retries(self):
        client = MCPClient("http://localhost:8001", max_retries=2)
        session = _FakeSession(call_tool_side_effect=ConnectionRefusedError("refused"))
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPConnectionError):
                asyncio.run(client.call_tool("profile_dataset", {}))
        assert session.call_tool.call_count == 3  # 1 initial + 2 retries

    def test_deadline_exceeded_raises_connection_error(self):
        async def _slow_call_tool(*args, **kwargs):
            await asyncio.sleep(1.0)
            return CallToolResult(content=[], structuredContent={}, isError=False)

        client = MCPClient("http://localhost:8001", max_retries=0)
        session = _FakeSession(call_tool_side_effect=_slow_call_tool)
        p1, p2 = _patch_mcp_sdk(session)
        with p1, p2:
            with pytest.raises(MCPConnectionError, match="deadline"):
                asyncio.run(client.call_tool("profile_dataset", {}, timeout=5.0, deadline=0.05))


# ─── 3.2 DataCleanerClient ────────────────────────────────────────────────────

def _mock_dc_transport(tool_responses: dict) -> MCPClient:
    transport = MagicMock(spec=MCPClient)
    async def call_tool(name, args, **kwargs):
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
                "quality_contract_version": "1",
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
                "changes_summary": {"total_rounds": 1, "plan_steps": 2},
                "rows_affected": 10,
            }
        })
        client = DataCleanerClient(transport)
        result = asyncio.run(client.clean_dataset("/data/test.csv"))
        assert isinstance(result, CleaningResult)
        assert result.rows_affected == 10

    def test_validate_quality(self):
        transport = _mock_dc_transport({
            "validate_quality": {"passed": True, "score": 0.97, "issues": [], "quality_contract_version": "1"}
        })
        client = DataCleanerClient(transport)
        validation = asyncio.run(client.validate_quality("/data/test_clean.csv"))
        assert isinstance(validation, QualityValidation)
        assert validation.passed is True
        assert validation.score == 0.97


# ─── 3.3 RAGServerClient ─────────────────────────────────────────────────────

def _mock_rag_transport(tool_responses: dict) -> MCPClient:
    transport = MagicMock(spec=MCPClient)
    async def call_tool(name, args, **kwargs):
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

    def test_retrieve_with_metadata_forwards_collection(self):
        """collection must reach the wire — this is what scopes retrieval to
        one document set instead of the whole shared knowledge base."""
        transport = _mock_rag_transport({
            "retrieve_with_metadata": {"chunks": []}
        })
        client = RAGServerClient(transport)
        asyncio.run(client.retrieve_with_metadata(
            "market analysis", top_k=5, collection="wake_apparel"
        ))
        transport.call_tool.assert_awaited_once_with(
            "retrieve_with_metadata",
            {"input": {"query": "market analysis", "top_k": 5, "collection": "wake_apparel"}},
            timeout=15.0, deadline=30.0,
        )

    def test_retrieve_with_metadata_omits_collection_when_unset(self):
        """Unscoped calls must not send a "collection" key at all, so a
        RAG-MCP-Server without collection support (or where it defaults to
        None) sees exactly the same request shape as before this feature."""
        transport = _mock_rag_transport({
            "retrieve_with_metadata": {"chunks": []}
        })
        client = RAGServerClient(transport)
        asyncio.run(client.retrieve_with_metadata("market analysis", top_k=5))
        transport.call_tool.assert_awaited_once_with(
            "retrieve_with_metadata",
            {"input": {"query": "market analysis", "top_k": 5}},
            timeout=15.0, deadline=30.0,
        )

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
        async def dc_call(name, args, **kwargs):
            return dc_responses[name]
        dc_transport.call_tool = AsyncMock(side_effect=dc_call)

    if rag_side_effect:
        rag_transport.call_tool = AsyncMock(side_effect=rag_side_effect)
    elif rag_responses:
        async def rag_call(name, args, **kwargs):
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
            "profile_dataset": {"row_count": 500, "columns": [], "quality_issues": [], "quality_contract_version": "1"}
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

    def test_retrieve_knowledge_forwards_collection_end_to_end(self):
        """collection must survive the full path: fallback layer ->
        RAGServerClient -> transport.call_tool's wire-level args."""
        fb = _build_fallback(rag_responses={
            "retrieve_with_metadata": {"chunks": []}
        })
        asyncio.run(fb.retrieve_knowledge("query", collection="wake_apparel"))
        rag_transport = fb._rag._transport
        rag_transport.call_tool.assert_awaited_once_with(
            "retrieve_with_metadata",
            {"input": {"query": "query", "top_k": 5, "collection": "wake_apparel"}},
            timeout=15.0, deadline=30.0,
        )

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
            "profile_dataset": {"row_count": 10, "columns": [], "quality_issues": [], "quality_contract_version": "1"}
        })
        _, log = asyncio.run(fb.profile_dataset("/data/x.csv"))
        for field in ["system", "tool", "args", "duration_ms", "mode"]:
            assert field in log, f"Missing log field: {field}"

    def test_log_records_timing(self):
        fb = _build_fallback(dc_responses={
            "profile_dataset": {"row_count": 1, "columns": [], "quality_issues": [], "quality_contract_version": "1"}
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
        issues = [i for i in report.quality_issues if i.column == "b"]
        assert any(i.code == "high_null_rate" for i in issues)

    def test_unreadable_file_returns_empty_report(self):
        report = _basic_pandas_profile("/nonexistent/path/data.csv")
        assert report.row_count == 0
        assert report.columns == []
        assert report.has_critical_issues is False

    def test_detects_duplicate_rows(self, tmp_path):
        csv = tmp_path / "dups.csv"
        csv.write_text("a,b\n1,x\n1,x\n2,y\n1,x\n")
        report = _basic_pandas_profile(str(csv))
        dup = [i for i in report.quality_issues if i.code == "duplicate_rows"]
        assert len(dup) == 1
        assert "2 fully duplicated rows" in dup[0].detail

    def test_detects_constant_column(self, tmp_path):
        csv = tmp_path / "const.csv"
        csv.write_text("region,value\nNorth,1\nNorth,2\nNorth,3\n")
        report = _basic_pandas_profile(str(csv))
        issues = [i for i in report.quality_issues if i.code == "constant_column"]
        assert [i.column for i in issues] == ["region"]

    def test_detects_mixed_type_column(self, tmp_path):
        csv = tmp_path / "mixed.csv"
        csv.write_text("amount\n100\n200\nN/A\n300\nunknown\n400\n")
        report = _basic_pandas_profile(str(csv))
        issues = [i for i in report.quality_issues if i.code == "mixed_types"]
        assert [i.column for i in issues] == ["amount"]

    def test_all_text_column_not_flagged_as_mixed(self, tmp_path):
        csv = tmp_path / "text.csv"
        csv.write_text("city\nParis\nLondon\nBerlin\n")
        report = _basic_pandas_profile(str(csv))
        assert not [i for i in report.quality_issues if i.code == "mixed_types"]

    def test_detects_high_outlier_share(self, tmp_path):
        csv = tmp_path / "outliers.csv"
        # 10 tightly clustered values + 2 extreme ones (>5% outlier share)
        values = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100, 10000, -10000]
        csv.write_text("v\n" + "\n".join(str(v) for v in values) + "\n")
        report = _basic_pandas_profile(str(csv))
        issues = [i for i in report.quality_issues if i.code == "high_outlier_share"]
        assert [i.column for i in issues] == ["v"]

    def test_clean_data_produces_no_issues(self, tmp_path):
        csv = tmp_path / "clean.csv"
        csv.write_text(
            "id,city\n" + "\n".join(f"{i},city{i}" for i in range(20)) + "\n"
        )
        report = _basic_pandas_profile(str(csv))
        assert report.quality_issues == []

    def test_never_sets_critical_issues(self, tmp_path):
        """Fallback mode has no cleaner to delegate to — critical=True would
        trigger a cleaning path that no-ops but claims cleaning was applied."""
        csv = tmp_path / "bad.csv"
        csv.write_text("a,b\n1,\n1,\n1,\n1,\n1,\n")  # constant + 100% nulls + dups
        report = _basic_pandas_profile(str(csv))
        assert report.quality_issues  # plenty found
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
                "quality_contract_version": "1",
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


# ─── M1: error-envelope detection at the typed-client boundary ────────────────
#
# Before this, DataCleanerClient/RAGServerClient silently parsed the
# migration-era error envelope ({"error": true, ...} / {"error": "..."}) as
# if it were a real successful response -- confirmed live via
# scripts/check_ecosystem.py. These are the regression tests for that fix.

class TestErrorEnvelopeDetection:
    def test_cleaner_error_envelope_raises_tool_error(self):
        transport = _mock_dc_transport({
            "profile_dataset": {
                "error": True, "error_type": "FileNotFoundError",
                "message": "[Errno 2] No such file or directory: 'x.csv'",
            }
        })
        client = DataCleanerClient(transport)
        with pytest.raises(MCPToolError) as exc_info:
            asyncio.run(client.profile_dataset("x.csv"))
        assert exc_info.value.error_type == "FileNotFoundError"
        assert exc_info.value.raw["error"] is True

    def test_cleaner_success_envelope_does_not_raise(self):
        transport = _mock_dc_transport({
            "profile_dataset": {
                "row_count": 5, "columns": [], "quality_issues": [],
                "quality_contract_version": "1",
            }
        })
        client = DataCleanerClient(transport)
        report = asyncio.run(client.profile_dataset("x.csv"))
        assert report.row_count == 5

    def test_rag_error_field_raises_tool_error(self):
        transport = _mock_rag_transport({
            "retrieve": {"error": "INDEX_NOT_FOUND: no such collection"},
        })
        client = RAGServerClient(transport)
        with pytest.raises(MCPToolError) as exc_info:
            asyncio.run(client.retrieve("query"))
        assert "INDEX_NOT_FOUND" in str(exc_info.value)

    def test_rag_empty_chunks_is_not_an_error(self):
        """A legitimate zero-hit query must not be confused with a failure."""
        transport = _mock_rag_transport({"retrieve": {"chunks": []}})
        client = RAGServerClient(transport)
        chunks = asyncio.run(client.retrieve("query"))
        assert chunks == []


class TestContractVersionCheck:
    def test_mismatched_version_raises_contract_error(self):
        from src.mcp_client.client import MCPContractError

        transport = _mock_dc_transport({
            "profile_dataset": {
                "row_count": 5, "columns": [], "quality_issues": [],
                "quality_contract_version": "2",  # Settings expects "1"
            }
        })
        client = DataCleanerClient(transport)
        with pytest.raises(MCPContractError):
            asyncio.run(client.profile_dataset("x.csv"))

    def test_missing_version_raises_contract_error(self):
        """A response with no quality_contract_version at all must not be
        silently treated as version "1" by default."""
        from src.mcp_client.client import MCPContractError

        transport = _mock_dc_transport({
            "profile_dataset": {"row_count": 5, "columns": [], "quality_issues": []}
        })
        client = DataCleanerClient(transport)
        with pytest.raises(MCPContractError):
            asyncio.run(client.profile_dataset("x.csv"))

    def test_clean_dataset_has_no_version_field_to_check(self):
        """定案 #4b / 附录 B.6: only profile_dataset and validate_quality
        carry quality_contract_version -- clean_dataset must not require it."""
        transport = _mock_dc_transport({
            "clean_dataset": {"cleaned_path": "x.csv", "changes_summary": {}, "rows_affected": 0}
        })
        client = DataCleanerClient(transport)
        result = asyncio.run(client.clean_dataset("x.csv"))
        assert result.cleaned_path == "x.csv"


class TestFieldPresenceCheck:
    """
    附录 B.3 "字段缺失" row: a response genuinely missing a field the
    contract guarantees must not be silently treated as the safest-looking
    default (e.g. has_critical_issues absent -> silently "False" -> a
    dirty dataset walks right past cleaning). strict fails outright;
    degraded proceeds with the default but must not do so silently.
    """

    @pytest.fixture(autouse=True)
    def _reset_strict_mode(self):
        from src.config.settings import settings as _settings
        original = _settings.mcp_strict_mode
        yield
        _settings.mcp_strict_mode = original

    def test_missing_has_critical_issues_fails_in_strict_mode(self):
        from src.config.settings import settings as _settings
        from src.mcp_client.client import MCPContractError

        _settings.mcp_strict_mode = "strict"
        transport = _mock_dc_transport({
            "profile_dataset": {
                "row_count": 5, "columns": [], "quality_issues": [],
                "quality_contract_version": "1",
            }
        })
        client = DataCleanerClient(transport)
        with pytest.raises(MCPContractError):
            asyncio.run(client.profile_dataset("x.csv"))

    def test_missing_has_critical_issues_defaults_false_in_degraded_mode(self):
        from src.config.settings import settings as _settings

        _settings.mcp_strict_mode = "degraded"
        transport = _mock_dc_transport({
            "profile_dataset": {
                "row_count": 5, "columns": [], "quality_issues": [],
                "quality_contract_version": "1",
            }
        })
        client = DataCleanerClient(transport)
        report = asyncio.run(client.profile_dataset("x.csv"))
        assert report.has_critical_issues is False

    def test_missing_passed_fails_in_strict_mode(self):
        from src.config.settings import settings as _settings
        from src.mcp_client.client import MCPContractError

        _settings.mcp_strict_mode = "strict"
        transport = _mock_dc_transport({
            "validate_quality": {"score": 1.0, "issues": [], "quality_contract_version": "1"}
        })
        client = DataCleanerClient(transport)
        with pytest.raises(MCPContractError):
            asyncio.run(client.validate_quality("x.csv"))

    def test_missing_passed_defaults_true_in_degraded_mode(self):
        from src.config.settings import settings as _settings

        _settings.mcp_strict_mode = "degraded"
        transport = _mock_dc_transport({
            "validate_quality": {"score": 1.0, "issues": [], "quality_contract_version": "1"}
        })
        client = DataCleanerClient(transport)
        validation = asyncio.run(client.validate_quality("x.csv"))
        assert validation.passed is True


# ─── M4: clean_dataset explicit args + planner_mode degradation check ─────────

class TestCleanDatasetExplicitArgs:
    def test_sends_planner_mode_max_rounds_run_id(self):
        transport = _mock_dc_transport({
            "clean_dataset": {
                "cleaned_path": "x_clean.csv", "changes_summary": {}, "rows_affected": 0,
                "execution_plan": {
                    "plan_id": "p1", "planner_mode_requested": "rule",
                    "planner_mode_used": "rule", "planner_fallback_reason": None, "steps": [],
                },
            }
        })
        client = DataCleanerClient(transport)
        asyncio.run(client.clean_dataset("x.csv", planner_mode="rule", max_rounds=1, run_id="r1"))
        sent_args = transport.call_tool.await_args.args[1]
        assert sent_args["planner_mode"] == "rule"
        assert sent_args["max_rounds"] == 1
        assert sent_args["run_id"] == "r1"

    def test_omits_run_id_when_not_given(self):
        """FastMCP silently ignores unknown fields (附录 E P4) -- but an
        empty run_id is a valid-looking value, not an unknown field, so it
        must not be sent at all when the caller didn't ask for one."""
        transport = _mock_dc_transport({
            "clean_dataset": {"cleaned_path": "x.csv", "changes_summary": {}, "rows_affected": 0}
        })
        client = DataCleanerClient(transport)
        asyncio.run(client.clean_dataset("x.csv"))
        sent_args = transport.call_tool.await_args.args[1]
        assert "run_id" not in sent_args

    def test_execution_plan_parsed_onto_result(self):
        transport = _mock_dc_transport({
            "clean_dataset": {
                "cleaned_path": "x.csv", "changes_summary": {}, "rows_affected": 0,
                "execution_plan": {
                    "plan_id": "p1", "planner_mode_requested": "rule",
                    "planner_mode_used": "rule", "planner_fallback_reason": None,
                    "steps": [{"step_id": "s1", "mcp_tool": "handle_missing"}],
                },
            }
        })
        client = DataCleanerClient(transport)
        result = asyncio.run(client.clean_dataset("x.csv"))
        assert result.execution_plan["plan_id"] == "p1"
        assert len(result.execution_plan["steps"]) == 1


class TestPlannerModeDegradation:
    """定案 #6: the cleaner's own LLMPlanner already degrades from llm to
    rule internally and reports it via execution_plan -- strict mode must
    not silently accept that; degraded mode may, but must log it."""

    @pytest.fixture(autouse=True)
    def _reset_strict_mode(self):
        from src.config.settings import settings as _settings
        original = _settings.mcp_strict_mode
        yield
        _settings.mcp_strict_mode = original

    @staticmethod
    def _degraded_response():
        return {
            "cleaned_path": "x.csv", "changes_summary": {}, "rows_affected": 0,
            "execution_plan": {
                "plan_id": "p1", "planner_mode_requested": "llm",
                "planner_mode_used": "rule",
                "planner_fallback_reason": "litellm.AuthenticationError: missing key",
                "steps": [],
            },
        }

    def test_strict_mode_raises_on_silent_llm_to_rule_fallback(self):
        from src.config.settings import settings as _settings
        from src.mcp_client.client import MCPContractError

        _settings.mcp_strict_mode = "strict"
        transport = _mock_dc_transport({"clean_dataset": self._degraded_response()})
        client = DataCleanerClient(transport)
        with pytest.raises(MCPContractError):
            asyncio.run(client.clean_dataset("x.csv", planner_mode="llm"))

    def test_degraded_mode_accepts_fallback_with_warning(self, caplog):
        from src.config.settings import settings as _settings

        _settings.mcp_strict_mode = "degraded"
        transport = _mock_dc_transport({"clean_dataset": self._degraded_response()})
        client = DataCleanerClient(transport)
        result = asyncio.run(client.clean_dataset("x.csv", planner_mode="llm"))
        assert result.execution_plan["planner_mode_used"] == "rule"

    def test_rule_requested_never_triggers_the_check(self):
        """rule has no external dependency -- must never raise regardless
        of what execution_plan happens to say."""
        from src.config.settings import settings as _settings
        from src.mcp_client.client import MCPContractError

        _settings.mcp_strict_mode = "strict"
        transport = _mock_dc_transport({"clean_dataset": self._degraded_response()})
        client = DataCleanerClient(transport)
        # Even though the canned response claims a "llm -> rule" fallback,
        # this call only ever requested "rule" -- the check is a no-op.
        try:
            result = asyncio.run(client.clean_dataset("x.csv", planner_mode="rule"))
        except MCPContractError:
            pytest.fail("planner_mode='rule' must never trigger the degradation check")
        assert result.cleaned_path == "x.csv"


# ─── M1: error-matrix classification in SubSystemWithFallback ─────────────────

class TestErrorMatrix:
    """
    ECOSYSTEM_INTEGRATION_PLAN.md 错误处理矩阵: only connection/
    internal_unknown errors may ever produce a fallback result, and only in
    degraded mode. data_input/contract errors fail in *both* modes.
    """

    @pytest.fixture(autouse=True)
    def _reset_strict_mode(self):
        from src.config.settings import settings as _settings
        original = _settings.mcp_strict_mode
        yield
        _settings.mcp_strict_mode = original

    def test_data_input_error_fails_in_degraded_mode_too(self):
        """A nonexistent file must not be papered over as an empty/fallback
        profile in degraded mode — it's a real error, not an availability
        problem the standalone guarantee is meant to cover."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure

        _settings.mcp_strict_mode = "degraded"
        fb = _build_fallback(dc_side_effect=MCPToolError(
            "profile_dataset reported an internal error: not found",
            error_type="FileNotFoundError",
        ))
        with pytest.raises(SubSystemHardFailure) as exc_info:
            asyncio.run(fb.profile_dataset("/data/missing.csv"))
        assert exc_info.value.error_class == "data_input"
        assert exc_info.value.log["mode"] == "mcp"  # never "fallback"

    def test_connection_error_fails_in_strict_mode(self):
        """The same connection failure that falls back in degraded mode
        must hard-fail in strict — strict exists precisely so problems
        aren't hidden during CI/联调."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure

        _settings.mcp_strict_mode = "strict"
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        with pytest.raises(SubSystemHardFailure) as exc_info:
            asyncio.run(fb.profile_dataset("/data/x.csv"))
        assert exc_info.value.error_class == "connection"

    def test_connection_error_still_falls_back_in_degraded_mode(self):
        """Regression guard: strict-mode behavior must not leak into the
        default degraded mode other tests rely on."""
        from src.config.settings import settings as _settings

        _settings.mcp_strict_mode = "degraded"
        fb = _build_fallback(dc_side_effect=MCPConnectionError("offline"))
        report, log = asyncio.run(fb.profile_dataset("/data/x.csv"))
        assert log["mode"] == "fallback"
        assert log["error_class"] == "connection"
        assert log["recoverable"] is True
        assert log["service_reachable"] is False

    def test_contract_error_fails_in_both_modes(self):
        from src.config.settings import settings as _settings
        from src.mcp_client.client import MCPContractError
        from src.mcp_client.fallback import SubSystemHardFailure

        for mode in ("strict", "degraded"):
            _settings.mcp_strict_mode = mode
            fb = _build_fallback(dc_side_effect=MCPContractError(
                "version mismatch", error_type="ContractVersionMismatch",
            ))
            with pytest.raises(SubSystemHardFailure) as exc_info:
                asyncio.run(fb.profile_dataset("/data/x.csv"))
            assert exc_info.value.error_class == "contract", f"failed for mode={mode}"

    def test_bad_zip_file_is_data_input_not_internal_unknown(self):
        """Live TB2 regression: feeding cleaner a garbage-bytes .zip path
        raises {"error": true, "error_type": "BadZipFile", ...} (confirmed
        against the real service, not guessed) -- matrix row "格式不支持".
        Before BadZipFile was added to _DATA_INPUT_ERROR_TYPES this fell
        into internal_unknown and, in degraded mode, silently invoked the
        pandas fallback profiler on the same unreadable file, which fails
        identically and returns a fabricated empty-but-successful report
        (row_count=0, has_critical_issues=False) -- the G.1-shaped bug,
        recurring under a different exception name."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure

        for mode in ("strict", "degraded"):
            _settings.mcp_strict_mode = mode
            fb = _build_fallback(dc_side_effect=MCPToolError(
                "profile_dataset reported an internal error: File is not a zip file",
                error_type="BadZipFile",
            ))
            with pytest.raises(SubSystemHardFailure) as exc_info:
                asyncio.run(fb.profile_dataset("/data/bad.zip"))
            assert exc_info.value.error_class == "data_input", f"failed for mode={mode}"
            assert exc_info.value.log["mode"] == "mcp"  # never "fallback"

    def test_rag_invalid_collection_is_data_input_not_internal_unknown(self):
        """rag-framework R3 gives protocol-level tool errors a stable
        "<CODE>: <detail>" text prefix (rag/core/errors.py) but
        MCPToolError.error_type stays None for that shape (only the
        migration-era envelope populates it) -- so before this
        classification, INVALID_COLLECTION fell through to
        internal_unknown and was fallback-eligible in degraded mode,
        indistinguishable from a real service bug. A bad collection name
        is a caller mistake, same family as a bad file path, and must
        hard-fail in both modes like any other data_input error."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure

        for mode in ("strict", "degraded"):
            _settings.mcp_strict_mode = mode
            fb = _build_fallback(rag_side_effect=MCPToolError(
                "MCP tool error: Error executing tool retrieve_with_metadata: "
                "INVALID_COLLECTION: Collection 'ghost' does not exist. "
                "Known collections: ['wake_apparel']."
            ))
            with pytest.raises(SubSystemHardFailure) as exc_info:
                asyncio.run(fb.retrieve_knowledge("q"))
            assert exc_info.value.error_class == "data_input", f"failed for mode={mode}"

    def test_rag_index_config_mismatch_is_data_input(self):
        """Same family as INVALID_COLLECTION: a query configuration that
        doesn't match what the corpus was actually indexed with is the
        "假 hybrid" case R2 exists to prevent -- degraded mode must not
        silently fallback past it."""
        from src.config.settings import settings as _settings
        from src.mcp_client.fallback import SubSystemHardFailure

        for mode in ("strict", "degraded"):
            _settings.mcp_strict_mode = mode
            fb = _build_fallback(rag_side_effect=MCPToolError(
                "MCP tool error: Error executing tool retrieve: "
                "INDEX_CONFIG_MISMATCH: requested embedding_provider=openai "
                "but index was built with provider=none."
            ))
            with pytest.raises(SubSystemHardFailure) as exc_info:
                asyncio.run(fb.retrieve_knowledge("q"))
            assert exc_info.value.error_class == "data_input", f"failed for mode={mode}"

    def test_rag_retrieval_internal_error_stays_internal_unknown(self):
        """Unlike INVALID_COLLECTION/INDEX_CONFIG_MISMATCH,
        RETRIEVAL_INTERNAL_ERROR is a real service-side bug, not a caller
        mistake -- it must keep the internal_unknown classification
        (fallback-eligible in degraded, matching "服务内部未知错误")."""
        from src.config.settings import settings as _settings

        _settings.mcp_strict_mode = "degraded"
        fb = _build_fallback(rag_side_effect=MCPToolError(
            "MCP tool error: Error executing tool retrieve_with_metadata: "
            "RETRIEVAL_INTERNAL_ERROR: unexpected failure in query pipeline."
        ))
        chunks, log = asyncio.run(fb.retrieve_knowledge("q"))
        assert log["mode"] == "fallback"
        assert log["error_class"] == "internal_unknown"
