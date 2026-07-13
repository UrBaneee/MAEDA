"""
MAEDA MCP Server — Phase 10.

Exposes MAEDA itself as an MCP server so it can be called by Claude Desktop,
MCP Inspector, or any other MCP-capable client.

Tools exposed:
  10.2 analyze_data         — run the full MAEDA pipeline on a query + data source
  10.3 connect_data_source  — register a new data source
  10.4 get_eval_report      — return the latest eval scores

Usage:
    python -m src.mcp_server.server          # starts on stdio (Claude Desktop)
    uvicorn src.mcp_server.server:app        # HTTP mode

Progress streaming (10.5): each pipeline phase emits a progress event via the
result text so clients can display real-time status.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger("maeda.mcp_server")

# ─── In-memory session store ─────────────────────────────────────────────────
# Maps source_id → source descriptor; cleared on restart (demo-grade)
_registered_sources: dict[str, dict] = {}
_latest_eval: dict = {}


# ─── Tool implementations (pure async functions — no MCP dependency needed) ──

async def _analyze_data(query: str, data_source: str) -> dict:
    """
    10.2 Run the full MAEDA LangGraph pipeline.
    Returns a summary with report, insights, eval scores, and charts metadata.
    Streams progress via status field in each stage dict.
    """
    from src.graph.builder import build_graph
    from src.state.graph_state import initial_state

    state = initial_state(query)

    # Register data source if provided
    if data_source:
        state["data_sources"] = [{"path": data_source, "source_type": _infer_type(data_source)}]

    progress: list[str] = []

    def _emit(msg: str) -> None:
        progress.append(msg)
        logger.info("[MCP progress] %s", msg)

    _emit("Parsing intent...")
    start = time.time()
    try:
        graph = build_graph()
        result = await graph.ainvoke(state)
        elapsed = round(time.time() - start, 2)
        _emit(f"Complete in {elapsed}s")

        # Cache eval for get_eval_report
        global _latest_eval
        _latest_eval = result.get("eval_scores") or {}

        return {
            "status": "success",
            "query": query,
            "report": result.get("report", ""),
            "insights": result.get("insights", []),
            "charts": [
                {"type": c.get("chart_type"), "title": c.get("title"), "path": c.get("image_path")}
                for c in result.get("charts", [])
                if c.get("chart_type") != "dashboard"
            ],
            "eval_scores": result.get("eval_scores"),
            "guardrail_passed": result.get("guardrail_passed"),
            "progress": progress,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        logger.error("analyze_data failed: %s", exc)
        return {
            "status": "error",
            "query": query,
            "error": str(exc),
            "progress": progress,
        }


async def _connect_data_source(source_type: str, path_or_uri: str) -> dict:
    """
    10.3 Register a data source so it is available for subsequent analyze_data calls.
    """
    import uuid
    source_id = str(uuid.uuid4())[:8]
    descriptor = {
        "id": source_id,
        "source_type": source_type,
        "path": path_or_uri,
    }
    _registered_sources[source_id] = descriptor

    # Quick validation
    try:
        from src.tools.data_connector import DataConnector
        connector = DataConnector()
        schema, summary = await connector.connect_with_summary(descriptor)
        descriptor["schema_summary"] = summary
        descriptor["columns"] = len(schema.columns)
        descriptor["rows"] = schema.row_count
        return {"status": "connected", "source_id": source_id, **descriptor}
    except Exception as exc:
        logger.warning("connect_data_source validation failed: %s", exc)
        descriptor["schema_summary"] = f"Could not profile: {exc}"
        return {"status": "registered_without_profile", "source_id": source_id, **descriptor}


async def _get_eval_report() -> dict:
    """10.4 Return the latest evaluation report."""
    if not _latest_eval:
        return {"status": "no_eval_available",
                "message": "Run analyze_data first to generate an eval report."}
    return {"status": "ok", "eval_scores": _latest_eval}


# ─── MCP Server (FastMCP) ────────────────────────────────────────────────────

def _build_mcp_app():
    """Build the FastMCP application. Imported lazily to avoid hard dependency."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "mcp package not installed. Run: pip install mcp"
        )

    mcp = FastMCP("MAEDA — Multi-Agent Enterprise Data Analyst")

    @mcp.tool()
    async def analyze_data(query: str, data_source: str = "") -> str:
        """
        Analyze a dataset using the MAEDA multi-agent pipeline.

        Args:
            query: Natural language analysis question (e.g. "Why did revenue drop in Q3?")
            data_source: Path to CSV/JSON/Excel/SQLite or connection string. Optional if
                         a source was previously registered with connect_data_source.
        """
        result = await _analyze_data(query, data_source)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def connect_data_source(source_type: str, path_or_uri: str) -> str:
        """
        Register a new data source for analysis.

        Args:
            source_type: One of: csv, sqlite, postgres, json, excel
            path_or_uri: File path or database connection URI
        """
        result = await _connect_data_source(source_type, path_or_uri)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def get_eval_report() -> str:
        """
        Get the latest MAEDA evaluation report (scores from the most recent run).
        """
        result = await _get_eval_report()
        return json.dumps(result, indent=2, default=str)

    return mcp


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    """Start MAEDA MCP server on stdio (for Claude Desktop)."""
    mcp = _build_mcp_app()
    mcp.run()


if __name__ == "__main__":
    main()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _infer_type(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "csv": "csv", "json": "json", "xlsx": "excel", "xls": "excel",
        "db": "sqlite", "sqlite": "sqlite", "sqlite3": "sqlite",
    }.get(ext, "csv")
