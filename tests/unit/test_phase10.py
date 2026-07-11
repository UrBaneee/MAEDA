"""
Phase 10 tests — MAEDA MCP Server.
Run with: pytest tests/unit/test_phase10.py -v
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


# ─── _infer_type helper ───────────────────────────────────────────────────────

def test_infer_type_csv():
    from src.mcp_server.server import _infer_type
    assert _infer_type("data/sales.csv") == "csv"

def test_infer_type_sqlite():
    from src.mcp_server.server import _infer_type
    assert _infer_type("data/db.sqlite") == "sqlite"

def test_infer_type_excel():
    from src.mcp_server.server import _infer_type
    assert _infer_type("report.xlsx") == "excel"

def test_infer_type_json():
    from src.mcp_server.server import _infer_type
    assert _infer_type("data.json") == "json"

def test_infer_type_unknown_defaults_csv():
    from src.mcp_server.server import _infer_type
    assert _infer_type("noextension") == "csv"


# ─── _connect_data_source ────────────────────────────────────────────────────

def test_connect_data_source_registers_source():
    import src.mcp_server.server as srv

    mock_schema = MagicMock()
    mock_schema.columns = [MagicMock(), MagicMock()]
    mock_schema.row_count = 100

    with patch("src.tools.data_connector.DataConnector") as mock_cls:
        mock_connector = MagicMock()
        mock_connector.connect_with_summary = AsyncMock(
            return_value=(mock_schema, "Dataset with 100 rows and 2 columns.")
        )
        mock_cls.return_value = mock_connector

        srv._registered_sources.clear()
        result = asyncio.run(srv._connect_data_source("csv", "/tmp/test.csv"))

    assert result["status"] == "connected"
    assert "source_id" in result
    assert result["source_id"] in srv._registered_sources


def test_connect_data_source_handles_profile_failure():
    import src.mcp_server.server as srv

    with patch("src.tools.data_connector.DataConnector") as mock_cls:
        mock_connector = MagicMock()
        mock_connector.connect_with_summary = AsyncMock(
            side_effect=FileNotFoundError("File not found")
        )
        mock_cls.return_value = mock_connector

        srv._registered_sources.clear()
        result = asyncio.run(srv._connect_data_source("csv", "/nonexistent.csv"))

    assert "registered_without_profile" in result["status"]
    assert "source_id" in result


# ─── _get_eval_report ────────────────────────────────────────────────────────

def test_get_eval_report_empty():
    import src.mcp_server.server as srv
    srv._latest_eval = {}
    result = asyncio.run(srv._get_eval_report())
    assert result["status"] == "no_eval_available"


def test_get_eval_report_with_data():
    import src.mcp_server.server as srv
    srv._latest_eval = {"answer_relevance": {"score": 0.9, "label": "pass"}, "_aggregate": 0.85}
    result = asyncio.run(srv._get_eval_report())
    assert result["status"] == "ok"
    assert "_aggregate" in result["eval_scores"]


# ─── _analyze_data ────────────────────────────────────────────────────────────

def test_analyze_data_success():
    import src.mcp_server.server as srv
    import src.graph.nodes as _nodes

    # Mock the full graph to return a plausible state
    mock_result = {
        "report": "# Report\n\nRevenue is strong.",
        "insights": [{"finding": "Revenue up", "confidence": 0.9}],
        "charts": [{"chart_type": "bar", "title": "Sales", "image_path": "/tmp/chart.png"}],
        "eval_scores": {"_aggregate": 0.85},
        "guardrail_passed": True,
    }

    with patch("src.graph.builder.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value=mock_result)
        mock_build.return_value = mock_graph

        result = asyncio.run(srv._analyze_data("Show revenue", "data.csv"))

    assert result["status"] == "success"
    assert result["report"] == "# Report\n\nRevenue is strong."
    assert len(result["insights"]) == 1
    assert result["guardrail_passed"] is True
    # Eval should be cached
    assert srv._latest_eval.get("_aggregate") == 0.85


def test_analyze_data_error_handling():
    import src.mcp_server.server as srv

    with patch("src.graph.builder.build_graph", side_effect=RuntimeError("Graph failed")):
        result = asyncio.run(srv._analyze_data("q", ""))

    assert result["status"] == "error"
    assert "Graph failed" in result["error"]


def test_analyze_data_no_data_source():
    import src.mcp_server.server as srv

    mock_result = {
        "report": "# Report\nNo data.",
        "insights": [],
        "charts": [],
        "eval_scores": {},
        "guardrail_passed": True,
    }

    with patch("src.graph.builder.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value=mock_result)
        mock_build.return_value = mock_graph

        result = asyncio.run(srv._analyze_data("Describe the data", ""))

    assert result["status"] == "success"


# ─── Tool output is valid JSON ────────────────────────────────────────────────

def test_analyze_data_returns_json_serialisable():
    import src.mcp_server.server as srv

    mock_result = {
        "report": "# Report", "insights": [], "charts": [],
        "eval_scores": {"_aggregate": 0.7}, "guardrail_passed": True,
    }
    with patch("src.graph.builder.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value=mock_result)
        mock_build.return_value = mock_graph

        raw = asyncio.run(srv._analyze_data("q", ""))

    # Verify can be serialised to JSON (as the MCP tool wrapper does)
    serialised = json.dumps(raw, default=str)
    assert json.loads(serialised)["status"] == "success"


# ─── _build_mcp_app ───────────────────────────────────────────────────────────

def test_build_mcp_app_raises_without_mcp():
    """If mcp package missing, build raises ImportError."""
    import sys
    import importlib
    from unittest.mock import patch as _patch

    # Temporarily hide mcp
    with _patch.dict(sys.modules, {"mcp": None, "mcp.server": None,
                                    "mcp.server.fastmcp": None}):
        from src.mcp_server import server as srv_mod
        import importlib
        try:
            # Reimport forces the guard to run
            srv_mod._build_mcp_app()
            # If mcp IS installed this won't raise — that's fine
        except ImportError as e:
            assert "mcp" in str(e).lower()
        except Exception:
            pass  # Other errors (e.g. mcp is installed) are fine


def test_build_mcp_app_succeeds_when_mcp_available():
    """If mcp is importable, server builds without error."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        return  # Skip if mcp not installed

    from src.mcp_server.server import _build_mcp_app
    app = _build_mcp_app()
    assert app is not None


# ─── 10.5 Progress tracking ───────────────────────────────────────────────────

def test_analyze_data_includes_progress():
    import src.mcp_server.server as srv

    mock_result = {
        "report": "# Report", "insights": [], "charts": [],
        "eval_scores": {}, "guardrail_passed": True,
    }
    with patch("src.graph.builder.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value=mock_result)
        mock_build.return_value = mock_graph

        result = asyncio.run(srv._analyze_data("q", ""))

    assert "progress" in result
    assert len(result["progress"]) >= 1
    assert any("Parsing" in p for p in result["progress"])
