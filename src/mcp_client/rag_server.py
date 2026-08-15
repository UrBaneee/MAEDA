"""
RAG-MCP-Server integration.

Wraps MCPClient to provide typed, high-level calls to the RAG-MCP-Server
sub-system. MAEDA does NOT build its own RAG — it delegates entirely here.

Tools exposed by the RAG-MCP-Server:
  retrieve               {query, top_k, collection?} → list[RAGChunk]
  retrieve_with_metadata {query, top_k, collection?} → list[RAGChunk]  (with source attribution)
  list_collections       {}                          → list[Collection]

retrieve/retrieve_with_metadata's arguments are sent wrapped as
{"input": {query, top_k}} rather than flat {query, top_k} — the reference
implementation (rag-framework, FastMCP-based) declares these tools as
taking a single Pydantic-model parameter literally named `input`, and
FastMCP maps MCP tool-call arguments onto Python parameter names, so the
argument dict's top-level key must match. A different RAG-MCP-Server
implementation using flat keyword parameters instead of one Pydantic model
would need this changed back to unwrapped {query, top_k}.
"""
from __future__ import annotations

from typing import Optional

from src.mcp_client.client import MCPClient, MCPToolError
from src.mcp_client.models import Collection, RAGChunk
from src.utils.logger import get_logger

logger = get_logger("maeda.mcp.rag_server")


def _raise_if_error_field(raw: dict, tool: str) -> None:
    """
    rag-framework signals an internal failure via a top-level `error`
    string field (定案 #13 migration-era shape), distinct from a
    legitimate zero-hit `{"chunks": []}`. Before this check existed,
    nothing here looked for it, so a failure would silently resolve to
    `raw.get("chunks", [])` == [] — indistinguishable from "the query
    legitimately matched nothing".
    """
    error = raw.get("error") if isinstance(raw, dict) else None
    if error:
        raise MCPToolError(f"{tool} reported an error: {error}", raw=raw)


class RAGServerClient:
    """High-level client for the RAG-MCP-Server."""

    def __init__(self, transport: MCPClient):
        self._transport = transport

    async def retrieve(
        self, query: str, top_k: int = 5, collection: Optional[str] = None
    ) -> list[RAGChunk]:
        """Basic retrieval — returns chunks without detailed source metadata."""
        logger.debug("retrieve | query=%s top_k=%d collection=%s", query[:60], top_k, collection)
        input_args: dict = {"query": query, "top_k": top_k}
        if collection:
            input_args["collection"] = collection
        raw = await self._transport.call_tool(
            "retrieve", {"input": input_args}, timeout=15.0, deadline=30.0,
        )
        _raise_if_error_field(raw, "retrieve")
        return [RAGChunk.from_mcp_response(c) for c in raw.get("chunks", [])]

    async def retrieve_with_metadata(
        self, query: str, top_k: int = 5, collection: Optional[str] = None
    ) -> list[RAGChunk]:
        """
        Retrieval with full source attribution (source_file, page, chunk_id).
        Preferred over plain retrieve() for insight generation.

        collection: Optional collection name to scope retrieval (e.g.
            "wake_apparel"). Left unset, RAG-MCP-Server searches its entire
            knowledge base — see settings.rag_collection.
        """
        logger.debug(
            "retrieve_with_metadata | query=%s top_k=%d collection=%s",
            query[:60], top_k, collection,
        )
        input_args: dict = {"query": query, "top_k": top_k}
        if collection:
            input_args["collection"] = collection
        raw = await self._transport.call_tool(
            "retrieve_with_metadata", {"input": input_args}, timeout=15.0, deadline=30.0,
        )
        _raise_if_error_field(raw, "retrieve_with_metadata")
        return [RAGChunk.from_mcp_response(c) for c in raw.get("chunks", [])]

    async def list_collections(self) -> list[Collection]:
        """List available knowledge collections."""
        logger.debug("list_collections")
        raw = await self._transport.call_tool("list_collections", {}, timeout=10.0, deadline=20.0)
        _raise_if_error_field(raw, "list_collections")
        return [Collection.from_mcp_response(c) for c in raw.get("collections", [])]
