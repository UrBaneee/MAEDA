"""
Low-level async MCP transport client.

MCP over HTTP uses JSON-RPC 2.0. Each tool call is a POST to the server's
/mcp endpoint with method "tools/call" and params {name, arguments}.

This client is intentionally thin — it handles HTTP, timeouts, retries,
and health checks. High-level semantics live in data_cleaner.py / rag_server.py.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.client")


# ─── Exceptions ───────────────────────────────────────────────────────────────

class MCPError(Exception):
    """Base for all MCP client errors."""


class MCPConnectionError(MCPError):
    """Server is unreachable or the connection was refused."""


class MCPToolError(MCPError):
    """Server returned an error result for a tool call."""


# ─── Low-level MCP client ─────────────────────────────────────────────────────

class MCPClient:
    """
    Async JSON-RPC client for a single MCP server.

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
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Core tool call ─────────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool and return the parsed result dict.
        Raises MCPConnectionError on network failures, MCPToolError on server errors.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        return await self._post_with_retry(payload)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def _post_with_retry(self, payload: dict) -> dict:
        client = await self._get_client()
        try:
            response = await client.post("/mcp", json=payload)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise MCPConnectionError(
                f"Cannot reach MCP server at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise MCPConnectionError(
                f"MCP server returned HTTP {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        if "error" in body:
            raise MCPToolError(
                f"MCP tool error: {body['error'].get('message', body['error'])}"
            )

        result = body.get("result", {})
        # MCP spec: result may be wrapped in {"content": [...]} for tool responses
        if "content" in result and isinstance(result["content"], list):
            # Extract the first text content block
            for block in result["content"]:
                if block.get("type") == "text":
                    import json
                    try:
                        return json.loads(block["text"])
                    except (json.JSONDecodeError, KeyError):
                        return {"text": block.get("text", "")}
        return result

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> tuple[bool, Optional[float]]:
        """
        Returns (is_available, latency_ms).
        Sends a lightweight ping (tools/list) to test reachability.
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        client = await self._get_client()
        start = time.monotonic()
        try:
            response = await client.post("/mcp", json=payload, timeout=5.0)
            latency_ms = (time.monotonic() - start) * 1000
            return response.status_code == 200, latency_ms
        except Exception:
            return False, None


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
