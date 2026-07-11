"""
Graceful degradation layer for MCP sub-systems.

SubSystemWithFallback wraps DataCleanerClient and RAGServerClient.
When a sub-system is offline (MCPConnectionError), it falls back to
built-in alternatives so MAEDA can run standalone:

  Data Cleaner unavailable → basic pandas profiling
  RAG Server unavailable   → empty context (no domain enrichment)

MCP call logging (task 3.5) is also handled here: every call is timed and
appended to state["mcp_call_log"] via the provided log_call callback.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pandas as pd

from src.mcp_client.client import MCPConnectionError
from src.mcp_client.data_cleaner import DataCleanerClient
from src.mcp_client.models import (
    CleaningPlan,
    CleaningResult,
    Collection,
    ColumnProfile,
    DataQualityReport,
    QualityValidation,
    RAGChunk,
    SubSystemHealth,
)
from src.mcp_client.rag_server import RAGServerClient
from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.fallback")


# ─── MCP call logger ─────────────────────────────────────────────────────────

def _make_call_record(
    system: str,
    tool: str,
    args: dict,
    result: Any,
    duration_ms: float,
    error: Optional[str] = None,
    mode: str = "mcp",  # "mcp" | "fallback"
) -> dict:
    return {
        "system": system,
        "tool": tool,
        "args": args,
        "result_summary": str(result)[:200] if result else None,
        "duration_ms": round(duration_ms, 1),
        "error": error,
        "mode": mode,
    }


# ─── SubSystemWithFallback ────────────────────────────────────────────────────

class SubSystemWithFallback:
    """
    Facade over DataCleanerClient and RAGServerClient with:
      1. Graceful degradation on MCPConnectionError
      2. Automatic MCP call logging (call log returned per-call for state append)

    Usage:
        client = SubSystemWithFallback(data_cleaner, rag_server)
        report, log = await client.profile_dataset("/data/sales.csv")
        state["mcp_call_log"] = [*state["mcp_call_log"], log]
    """

    def __init__(
        self,
        data_cleaner: DataCleanerClient,
        rag_server: RAGServerClient,
    ):
        self._dc = data_cleaner
        self._rag = rag_server

    # ── Data Cleaner delegation ───────────────────────────────────────────────

    async def profile_dataset(self, path: str) -> tuple[DataQualityReport, dict]:
        """Profile dataset via Data Cleaner MCP; fall back to pandas on failure."""
        args = {"path": path}
        start = time.monotonic()
        try:
            result = await self._dc.profile_dataset(path)
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record("data_cleaner", "profile_dataset", args, result, duration_ms)
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("Data Cleaner unavailable, using pandas fallback: %s", exc)
            result = _basic_pandas_profile(path)
            log = _make_call_record(
                "data_cleaner", "profile_dataset", args, result,
                duration_ms, error=str(exc), mode="fallback"
            )
            return result, log

    async def get_cleaning_plan(self, path: str) -> tuple[CleaningPlan, dict]:
        """Get cleaning plan from Data Cleaner; fall back to empty plan."""
        args = {"path": path}
        start = time.monotonic()
        try:
            result = await self._dc.get_cleaning_plan(path)
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record("data_cleaner", "get_cleaning_plan", args, result, duration_ms)
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("Data Cleaner unavailable for cleaning plan: %s", exc)
            result = CleaningPlan(steps=[])
            log = _make_call_record(
                "data_cleaner", "get_cleaning_plan", args, result,
                duration_ms, error=str(exc), mode="fallback"
            )
            return result, log

    async def clean_dataset(
        self, path: str, plan: Optional[CleaningPlan] = None
    ) -> tuple[CleaningResult, dict]:
        """Clean dataset via Data Cleaner; fall back to returning path as-is."""
        args = {"path": path}
        start = time.monotonic()
        try:
            result = await self._dc.clean_dataset(path, plan)
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record("data_cleaner", "clean_dataset", args, result, duration_ms)
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("Data Cleaner unavailable for cleaning: %s", exc)
            result = CleaningResult(
                cleaned_path=path,
                changes_summary="Data Cleaner unavailable; no cleaning applied",
                rows_affected=0,
            )
            log = _make_call_record(
                "data_cleaner", "clean_dataset", args, result,
                duration_ms, error=str(exc), mode="fallback"
            )
            return result, log

    async def validate_quality(self, path: str) -> tuple[QualityValidation, dict]:
        """Validate data quality; fall back to 'passed' if unavailable."""
        args = {"path": path}
        start = time.monotonic()
        try:
            result = await self._dc.validate_quality(path)
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record("data_cleaner", "validate_quality", args, result, duration_ms)
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("Data Cleaner unavailable for validation: %s", exc)
            result = QualityValidation(passed=True, score=1.0, issues=[])
            log = _make_call_record(
                "data_cleaner", "validate_quality", args, result,
                duration_ms, error=str(exc), mode="fallback"
            )
            return result, log

    # ── RAG Server delegation ─────────────────────────────────────────────────

    async def retrieve_knowledge(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RAGChunk], dict]:
        """Retrieve domain knowledge; return empty list if RAG is unavailable."""
        args = {"query": query, "top_k": top_k}
        start = time.monotonic()
        try:
            result = await self._rag.retrieve_with_metadata(query, top_k)
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record(
                "rag_server", "retrieve_with_metadata", args,
                f"{len(result)} chunks", duration_ms
            )
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("RAG Server unavailable, skipping domain enrichment: %s", exc)
            log = _make_call_record(
                "rag_server", "retrieve_with_metadata", args,
                "0 chunks", duration_ms, error=str(exc), mode="fallback"
            )
            return [], log

    async def list_collections(self) -> tuple[list[Collection], dict]:
        """List RAG collections; return empty list on failure."""
        args: dict = {}
        start = time.monotonic()
        try:
            result = await self._rag.list_collections()
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record("rag_server", "list_collections", args, result, duration_ms)
            return result, log
        except MCPConnectionError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            log = _make_call_record(
                "rag_server", "list_collections", args, [],
                duration_ms, error=str(exc), mode="fallback"
            )
            return [], log

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> SubSystemHealth:
        """
        Check availability of both sub-systems.
        Safe to call at any time — never raises.
        """
        dc_ok, dc_ms = await _safe_health_check(self._dc._transport)
        rag_ok, rag_ms = await _safe_health_check(self._rag._transport)
        return SubSystemHealth(
            data_cleaner_available=dc_ok,
            rag_server_available=rag_ok,
            data_cleaner_latency_ms=dc_ms,
            rag_server_latency_ms=rag_ms,
        )


# ─── Pandas fallback profiler ─────────────────────────────────────────────────

def _basic_pandas_profile(path: str) -> DataQualityReport:
    """
    Minimal local profiling using pandas.
    Used when the Data Cleaner MCP is unavailable.
    """
    try:
        if path.startswith("sqlite:///"):
            import sqlite3 as _sqlite3
            bare = path[len("sqlite:///"):]
            con = _sqlite3.connect(bare)
            tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table = tables[0][0] if tables else None
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1000", con) if table else pd.DataFrame()
            con.close()
        elif path.endswith((".csv", ".tsv")):
            df = pd.read_csv(path)
        elif path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, nrows=1000)
        else:
            df = pd.read_json(path)
    except Exception as exc:
        logger.error("Fallback profiler could not read %s: %s", path, exc)
        return DataQualityReport(row_count=0, columns=[], quality_issues=[], has_critical_issues=False)

    columns = []
    quality_issues = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        unique_count = int(df[col].nunique())
        sample = df[col].dropna().head(3).tolist()
        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(df[col].dtype),
                null_pct=round(null_pct, 4),
                unique_count=unique_count,
                sample_values=[str(v) for v in sample],
            )
        )
        if null_pct > 0.5:
            quality_issues.append({
                "column": col,
                "issue": "high_null_rate",
                "severity": "warning",
                "detail": f"{null_pct:.1%} nulls",
            })

    return DataQualityReport(
        row_count=len(df),
        columns=columns,
        quality_issues=quality_issues,
        has_critical_issues=False,  # Pandas fallback never blocks progress
    )


async def _safe_health_check(transport) -> tuple[bool, Optional[float]]:
    try:
        return await transport.health_check()
    except Exception:
        return False, None


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_subsystem_client(
    data_cleaner_url: Optional[str] = None,
    rag_server_url: Optional[str] = None,
) -> SubSystemWithFallback:
    """
    Build the canonical SubSystemWithFallback from settings (or overrides).
    Import this wherever you need to call sub-systems.
    """
    from src.config.settings import settings
    from src.mcp_client.client import MCPClient

    dc_url = data_cleaner_url or settings.data_cleaner_mcp_url
    rag_url = rag_server_url or settings.rag_server_mcp_url

    dc_client = DataCleanerClient(MCPClient(dc_url))
    rag_client = RAGServerClient(MCPClient(rag_url))
    return SubSystemWithFallback(dc_client, rag_client)
