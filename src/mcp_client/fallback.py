"""
Graceful degradation + error-matrix classification layer for MCP sub-systems.

SubSystemWithFallback wraps DataCleanerClient and RAGServerClient and routes
every call through the ECOSYSTEM_INTEGRATION_PLAN.md 错误处理矩阵 (M1):

  错误类型                        | strict   | degraded          | 继续分析
  连接失败 / 明确的瞬时超时        | 失败     | fallback           | degraded 下是
  参数/schema/契约版本错误         | 失败     | 失败(不 fallback)  | 否
  认证或权限错误                   | 失败     | 失败(不 fallback)  | 否
  文件不存在/不可读/格式不支持     | 受控失败 | 受控失败(不 fallback) | 否
  服务内部未知错误                 | 失败     | fallback(须留痕)   | degraded 下是

Only the "connection" and "internal_unknown" rows are ever allowed to
produce a fallback result, and only in degraded mode (settings.mcp_strict_mode,
定案 #14). Everything else — a bad file path, a contract-version mismatch —
must not be silently papered over in either mode: it raises
SubSystemHardFailure so the caller (a graph node) can transition to an
explicit error state instead of presenting a fabricated result as if
nothing was wrong.

  Data Cleaner unavailable (connection/internal_unknown, degraded) → basic pandas profiling
  RAG Server unavailable   (connection/internal_unknown, degraded) → empty context

MCP call logging is also handled here: every call is timed and appended to
state["mcp_call_log"] via the returned log dict, now carrying error_class/
recoverable/service_reachable alongside the pre-existing mode/error fields.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pandas as pd

from src.config.settings import settings
from src.mcp_client.client import MCPConnectionError, MCPContractError, MCPToolError
from src.mcp_client.data_cleaner import DataCleanerClient
from src.mcp_client.models import (
    CleaningPlan,
    CleaningResult,
    Collection,
    ColumnProfile,
    DataQualityReport,
    QualityIssue,
    QualityValidation,
    RAGChunk,
    SubSystemHealth,
)
from src.mcp_client.rag_server import RAGServerClient
from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.fallback")


# ─── Error-matrix classification ───────────────────────────────────────────

# error_type names (the Python exception class name cleaner puts in its
# `{"error": true, "error_type": ...}` envelope) known today to be
# data/input errors rather than internal bugs. There is no stable
# error_code yet (附录 D C4 not landed — see 附录 E.1), so this is the best
# classification available; revisit once cleaner ships one.
_DATA_INPUT_ERROR_TYPES = {
    "FileNotFoundError", "IsADirectoryError", "PermissionError",
    "UnicodeDecodeError", "EmptyDataError", "ParserError",
}


class SubSystemHardFailure(Exception):
    """
    Raised when the 错误处理矩阵 says a call must NOT be papered over with a
    fallback result, in either mode: param/schema/contract-version errors,
    auth errors, and data/input errors. Callers must catch this and
    transition to an explicit error state — letting it propagate uncaught
    would skip guardrails/eval (CLAUDE.md: both run on every execution).
    """

    def __init__(self, message: str, error_class: str, log: dict):
        super().__init__(message)
        self.error_class = error_class
        self.log = log


def _classify(exc: Exception) -> tuple[str, bool, bool]:
    """
    Maps a caught exception to (error_class, recoverable, service_reachable).
    error_class ∈ {"connection", "data_input", "contract", "internal_unknown"}.
    "认证或权限错误" is a matrix row but neither sub-system emits a
    distinguishable auth error today — it currently falls into
    "internal_unknown" until C4 ships stable error codes.
    """
    if isinstance(exc, MCPConnectionError):
        return "connection", True, False
    if isinstance(exc, MCPContractError):
        return "contract", False, True
    if isinstance(exc, MCPToolError):
        if exc.error_type in _DATA_INPUT_ERROR_TYPES:
            return "data_input", False, True
        return "internal_unknown", False, True
    return "internal_unknown", False, True


def _make_call_record(
    system: str,
    tool: str,
    args: dict,
    result: Any,
    duration_ms: float,
    error: Optional[str] = None,
    mode: str = "mcp",  # "mcp" | "fallback"
    error_class: Optional[str] = None,
    recoverable: Optional[bool] = None,
    service_reachable: Optional[bool] = None,
) -> dict:
    return {
        "system": system,
        "tool": tool,
        "args": args,
        "result_summary": str(result)[:200] if result else None,
        "duration_ms": round(duration_ms, 1),
        "error": error,
        "mode": mode,
        "error_class": error_class,
        "recoverable": recoverable,
        "service_reachable": service_reachable,
    }


async def _call_with_matrix(
    system: str,
    tool: str,
    args: dict,
    call: Callable[[], Any],
    fallback_factory: Callable[[], Any],
) -> tuple[Any, dict]:
    """
    Shared error-matrix execution for one MCP call.

    - Success: returns (result, log) with mode="mcp".
    - connection / internal_unknown, degraded mode: returns
      (fallback_factory(), log) with mode="fallback" — the only case that
      ever calls fallback_factory.
    - Everything else (connection/internal_unknown in strict; data_input
      and contract in *either* mode): raises SubSystemHardFailure. The
      matrix is explicit that data_input/contract errors fail in degraded
      too — a bad path or a version mismatch is a real error, not a
      "sub-system unavailable" situation the standalone guarantee covers.
    """
    start = time.monotonic()
    try:
        result = await call()
        duration_ms = (time.monotonic() - start) * 1000
        return result, _make_call_record(system, tool, args, result, duration_ms)
    except (MCPConnectionError, MCPToolError) as exc:
        duration_ms = (time.monotonic() - start) * 1000
        error_class, recoverable, service_reachable = _classify(exc)
        mode = settings.mcp_strict_mode
        can_fallback = error_class in ("connection", "internal_unknown") and mode == "degraded"

        log = _make_call_record(
            system, tool, args, None, duration_ms, error=str(exc),
            mode="fallback" if can_fallback else "mcp",
            error_class=error_class, recoverable=recoverable,
            service_reachable=service_reachable,
        )

        if can_fallback:
            logger.warning(
                "%s.%s degraded to fallback (error_class=%s): %s", system, tool, error_class, exc,
            )
            return fallback_factory(), log

        logger.error(
            "%s.%s hard failure (mode=%s, error_class=%s): %s", system, tool, mode, error_class, exc,
        )
        raise SubSystemHardFailure(str(exc), error_class, log) from exc


# ─── SubSystemWithFallback ────────────────────────────────────────────────────

class SubSystemWithFallback:
    """
    Facade over DataCleanerClient and RAGServerClient with:
      1. Error-matrix-governed degradation (M1)
      2. Automatic MCP call logging (call log returned per-call for state append)

    Usage:
        client = SubSystemWithFallback(data_cleaner, rag_server)
        report, log = await client.profile_dataset("/data/sales.csv")
        state["mcp_call_log"] = [*state["mcp_call_log"], log]

    profile_dataset/get_cleaning_plan/clean_dataset/validate_quality/
    retrieve_knowledge/list_collections may now raise SubSystemHardFailure
    per the matrix — callers must catch it (see src/graph/nodes.py).
    """

    def __init__(
        self,
        data_cleaner: DataCleanerClient,
        rag_server: RAGServerClient,
    ):
        self._dc = data_cleaner
        self._rag = rag_server

    # ── Data Cleaner delegation ───────────────────────────────────────────────

    async def profile_dataset(self, path: str, run_id: str = "") -> tuple[DataQualityReport, dict]:
        """Profile dataset via Data Cleaner MCP; fall back to pandas on failure."""
        return await _call_with_matrix(
            "data_cleaner", "profile_dataset", {"dataset_path": path, "run_id": run_id},
            call=lambda: self._dc.profile_dataset(path, run_id=run_id),
            fallback_factory=lambda: _basic_pandas_profile(path),
        )

    async def get_cleaning_plan(self, path: str) -> tuple[CleaningPlan, dict]:
        """Get cleaning plan from Data Cleaner; fall back to empty plan."""
        return await _call_with_matrix(
            "data_cleaner", "get_cleaning_plan", {"dataset_path": path},
            call=lambda: self._dc.get_cleaning_plan(path),
            fallback_factory=lambda: CleaningPlan(steps=[]),
        )

    async def clean_dataset(
        self,
        path: str,
        plan: Optional[CleaningPlan] = None,
        planner_mode: str = "rule",
        max_rounds: int = 1,
        run_id: str = "",
    ) -> tuple[CleaningResult, dict]:
        """Clean dataset via Data Cleaner; fall back to returning path as-is."""
        return await _call_with_matrix(
            "data_cleaner", "clean_dataset",
            {"dataset_path": path, "planner_mode": planner_mode, "max_rounds": max_rounds, "run_id": run_id},
            call=lambda: self._dc.clean_dataset(
                path, plan, planner_mode=planner_mode, max_rounds=max_rounds, run_id=run_id,
            ),
            fallback_factory=lambda: CleaningResult(
                cleaned_path=path,
                changes_summary={"fallback_reason": "Data Cleaner unavailable; no cleaning applied"},
                rows_affected=0,
            ),
        )

    async def validate_quality(self, path: str, run_id: str = "") -> tuple[QualityValidation, dict]:
        """Validate data quality; fall back to 'passed' if unavailable."""
        return await _call_with_matrix(
            "data_cleaner", "validate_quality", {"dataset_path": path, "run_id": run_id},
            call=lambda: self._dc.validate_quality(path, run_id=run_id),
            fallback_factory=lambda: QualityValidation(passed=True, score=1.0, issues=[]),
        )

    # ── RAG Server delegation ─────────────────────────────────────────────────

    async def retrieve_knowledge(
        self, query: str, top_k: int = 5, collection: Optional[str] = None
    ) -> tuple[list[RAGChunk], dict]:
        """Retrieve domain knowledge; return empty list if RAG is unavailable."""
        return await _call_with_matrix(
            "rag_server", "retrieve_with_metadata",
            {"query": query, "top_k": top_k, "collection": collection},
            call=lambda: self._rag.retrieve_with_metadata(query, top_k, collection=collection),
            fallback_factory=lambda: [],
        )

    async def list_collections(self) -> tuple[list[Collection], dict]:
        """List RAG collections; return empty list on failure."""
        return await _call_with_matrix(
            "rag_server", "list_collections", {},
            call=lambda: self._rag.list_collections(),
            fallback_factory=lambda: [],
        )

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
    Local profiling using pandas. Used when the Data Cleaner MCP is unavailable.

    Checks: high null rate, constant columns, mixed-type object columns,
    duplicate rows, numeric outlier share, and empty datasets. All issues are
    reported as warnings/info for the Insight Agent's quality caveat —
    has_critical_issues stays False because in fallback mode there is no
    cleaner to delegate to: setting it True would send the pipeline down the
    get_cleaning_plan → clean_dataset path, both of which also fall back to
    no-ops, ending with the state claiming "cleaning applied" while nothing
    was actually cleaned. Honest caveats beat a fake cleaning pass.
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
    quality_issues: list[QualityIssue] = []

    if df.empty:
        quality_issues.append(QualityIssue(
            code="empty_dataset", severity="warning", detail="Dataset has no rows", source="fallback",
        ))

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality_issues.append(QualityIssue(
            code="duplicate_rows", severity="warning",
            detail=f"{dup_count} fully duplicated rows ({dup_count / len(df):.1%})",
            source="fallback",
        ))

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
            quality_issues.append(QualityIssue(
                code="high_null_rate", severity="warning", column=col,
                detail=f"{null_pct:.1%} nulls", source="fallback",
            ))

        non_null = df[col].dropna()
        if len(df) > 1 and unique_count == 1 and len(non_null) == len(df):
            quality_issues.append(QualityIssue(
                code="constant_column", severity="info", column=col,
                detail=f"Single value {non_null.iloc[0]!r} in every row", source="fallback",
            ))

        if df[col].dtype == object and len(non_null) >= 2:
            numeric_pct = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
            # All-numeric-as-strings and all-text are both internally
            # consistent; only a genuine mix is worth flagging.
            if 0.05 < numeric_pct < 0.95:
                quality_issues.append(QualityIssue(
                    code="mixed_types", severity="warning", column=col,
                    detail=f"{numeric_pct:.0%} of values parse as numbers, the rest are text",
                    source="fallback",
                ))

        if pd.api.types.is_numeric_dtype(df[col]) and len(non_null) >= 10:
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_pct = float(
                    ((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).mean()
                )
                if outlier_pct > 0.05:
                    quality_issues.append(QualityIssue(
                        code="high_outlier_share", severity="info", column=col,
                        detail=f"{outlier_pct:.1%} of values outside 1.5×IQR", source="fallback",
                    ))

    return DataQualityReport(
        row_count=len(df),
        columns=columns,
        quality_issues=quality_issues,
        has_critical_issues=False,  # see docstring: fallback has no cleaner to delegate to
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
    from src.mcp_client.client import MCPClient

    dc_url = data_cleaner_url or settings.data_cleaner_mcp_url
    rag_url = rag_server_url or settings.rag_server_mcp_url

    dc_client = DataCleanerClient(MCPClient(dc_url))
    rag_client = RAGServerClient(MCPClient(rag_url))
    return SubSystemWithFallback(dc_client, rag_client)
