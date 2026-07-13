"""
Low-level async MCP transport client.

Uses the official `mcp` SDK's streamable-http client + ClientSession to
speak the real MCP protocol: an "initialize" handshake that negotiates a
session ID, followed by tool calls that carry that session ID on every
request. An earlier hand-rolled implementation POSTed bare JSON-RPC to
/mcp with no handshake at all — against a spec-compliant server (FastMCP's
streamable-http transport) that gets a 406 Not Acceptable (missing the
required Accept header), then a 400 Missing session ID once the header is
fixed. This is presumably why "Phase 10: MAEDA MCP Server ... never
exercised by a real client" was a standing known limitation — the
integration layer had never actually been protocol-tested end to end.

This client is intentionally thin — it handles session lifecycle, timeouts,
retries, and health checks. High-level semantics (including any
tool-specific argument shape, e.g. some tools wrap their arguments under an
"input" key depending on how the server's tool function is declared) live
in data_cleaner.py / rag_server.py.
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.client")


# ─── Exceptions ───────────────────────────────────────────────────────────────

class MCPError(Exception):
    """Base for all MCP client errors."""


class MCPConnectionError(MCPError):
    """Server is unreachable, the connection was refused, or the protocol
    handshake failed."""


class MCPToolError(MCPError):
    """Server returned an error result for a tool call."""


# ─── Low-level MCP client ─────────────────────────────────────────────────────

class MCPClient:
    """
    Async client for a single MCP server over the streamable-http transport.

    Opens a fresh session (initialize handshake included) per call rather
    than holding one open across the whole pipeline run. MAEDA's graph
    nodes now all run under a single shared event loop (roadmap #13 —
    previously each node wrapped its work in its own `asyncio.run()`,
    which is exactly what made a long-lived session unsafe to reuse
    across nodes; that specific constraint is gone now, but per-call
    sessions remain the simpler, still-correct choice). The per-call
    handshake costs a network round trip, which is a reasonable trade
    for correctness at MAEDA's call volume (at most a handful of MCP
    calls per pipeline run).

    Supports:
    - Tool calls (call_tool)
    - Health checks (health_check)
    - Auto-retry with exponential back-off on transient failures
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def _mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """No-op: sessions are opened and closed per call, nothing to hold open."""

    # ── Core tool call ─────────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool and return the parsed result dict.
        Raises MCPConnectionError on network/handshake failures, MCPToolError
        on a tool-level error response.
        """
        return await self._call_with_retry(tool_name, arguments)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def _call_with_retry(self, tool_name: str, arguments: dict):
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http_client, \
                    streamable_http_client(self._mcp_url, http_client=http_client) as (
                        read, write, _,
                    ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        except MCPError:
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"Cannot reach MCP server at {self.base_url}: {exc}"
            ) from exc

        if result.isError:
            message = result.content[0].text if result.content else "Unknown tool error"
            raise MCPToolError(f"MCP tool error: {message}")

        return _parse_tool_result(result)

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> tuple[bool, Optional[float]]:
        """
        Returns (is_available, latency_ms).
        Opens a session and lists tools to verify both reachability and that
        the protocol handshake itself succeeds — a server that accepts TCP
        connections but rejects the handshake is not actually usable.
        """
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client, \
                    streamable_http_client(self._mcp_url, http_client=http_client) as (
                        read, write, _,
                    ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.list_tools()
            return True, (time.monotonic() - start) * 1000
        except Exception:
            return False, None


def _parse_tool_result(result) -> dict:
    """
    FastMCP populates structuredContent with the tool's actual return value
    when it's a Pydantic model/dict (true for every tool this codebase
    calls); fall back to parsing the first text content block as JSON for
    servers that only populate the human-readable content list.
    """
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        if getattr(block, "type", None) == "text":
            import json
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, AttributeError):
                return {"text": block.text}
    return {}


# ─── Unified sub-system client ────────────────────────────────────────────────

class SubSystemClient:
    """
    Unified MCP client providing access to both sub-systems.
    Used directly when both servers are known to be available.
    In normal use, prefer SubSystemWithFallback (fallback.py).
    """

    def __init__(self, data_cleaner_url: str, rag_server_url: str):
        from src.mcp_client.data_cleaner import DataCleanerClient
        from src.mcp_client.rag_server import RAGServerClient

        self._dc_transport = MCPClient(data_cleaner_url)
        self._rag_transport = MCPClient(rag_server_url)
        self.data_cleaner = DataCleanerClient(self._dc_transport)
        self.rag_server = RAGServerClient(self._rag_transport)

    async def close(self) -> None:
        await self._dc_transport.close()
        await self._rag_transport.close()
