"""
RAG-MCP-Server integration.

Wraps MCPClient to provide typed, high-level calls to the RAG-MCP-Server
sub-system. MAEDA does NOT build its own RAG — it delegates entirely here.

Tools exposed by the RAG-MCP-Server:
  retrieve               {query, top_k, collection?} → list[RAGChunk]
  retrieve_with_metadata {query, top_k, collection?} → list[RAGChunk]  (with source attribution)
  list_collections       {}                          → list[Collection]

Both retrieve tools also echo the tier the server actually ran at
(`retrieval_mode`/`embedding_provider`/`reranker_provider`/
`degraded_reason`) — parsed into RetrievalTier by the `*_tiered`
variants below, because rag picks that tier from its own environment and
MAEDA cannot infer it from the request it sent (附录 CH.2/CI.3/CK.3).

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
from src.mcp_client.models import Collection, RAGChunk, RetrievalTier
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
        chunks, _ = await self.retrieve_tiered(query, top_k, collection=collection)
        return chunks

    async def retrieve_tiered(
        self, query: str, top_k: int = 5, collection: Optional[str] = None
    ) -> tuple[list[RAGChunk], RetrievalTier]:
        """`retrieve` + the tier the server says it actually ran at (附录
        CI.3/CK.3 — see RetrievalTier). Both tools echo the same four
        fields, so both parse them; MAEDA's pipeline only uses
        retrieve_with_metadata today, but leaving `retrieve` unvalidated
        would just recreate the same blind spot the moment anything
        starts calling it."""
        logger.debug("retrieve | query=%s top_k=%d collection=%s", query[:60], top_k, collection)
        input_args: dict = {"query": query, "top_k": top_k}
        if collection:
            input_args["collection"] = collection
        raw = await self._transport.call_tool(
            "retrieve", {"input": input_args}, timeout=15.0, deadline=30.0,
        )
        _raise_if_error_field(raw, "retrieve")
        return (
            [RAGChunk.from_mcp_response(c) for c in raw.get("chunks", [])],
            RetrievalTier.from_mcp_response(raw),
        )

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
        chunks, _ = await self.retrieve_with_metadata_tiered(query, top_k, collection=collection)
        return chunks

    async def retrieve_with_metadata_tiered(
        self, query: str, top_k: int = 5, collection: Optional[str] = None
    ) -> tuple[list[RAGChunk], RetrievalTier]:
        """
        `retrieve_with_metadata` + the retrieval tier the server reports
        it actually ran at (附录 CI.3/CK.3). This is the variant
        SubSystemWithFallback.retrieve_knowledge uses, so every retrieval
        on MAEDA's real pipeline path carries its tier into
        state["mcp_call_log"].

        Kept as a separate method rather than changing
        retrieve_with_metadata's return type: the plain "just give me
        chunks" shape is what non-pipeline callers want, and a silently
        widened return type is the kind of change that breaks callers at
        runtime instead of at import.
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
        return (
            [RAGChunk.from_mcp_response(c) for c in raw.get("chunks", [])],
            RetrievalTier.from_mcp_response(raw),
        )

    async def list_collections(self) -> list[Collection]:
        """List available knowledge collections."""
        logger.debug("list_collections")
        raw = await self._transport.call_tool("list_collections", {}, timeout=10.0, deadline=20.0)
        _raise_if_error_field(raw, "list_collections")
        return [Collection.from_mcp_response(c) for c in raw.get("collections", [])]
