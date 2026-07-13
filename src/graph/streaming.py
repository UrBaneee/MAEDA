"""
Streaming helpers for driving the compiled graph with real per-node
progress, instead of guessing progress with a fixed timer.

Kept separate from ui/app.py: that module executes Streamlit page-rendering
calls (st.set_page_config(), sidebar, tabs) at import time, which makes it
hostile to unit testing without a running Streamlit session. This module
has no Streamlit dependency, so the actual streaming logic is independently
testable; ui/app.py only wires its output into placeholders.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Optional

from src.graph.builder import build_graph
from src.state.graph_state import MAEDAState, initial_state

# Human-readable label shown as each node starts/completes.
NODE_LABELS: dict[str, str] = {
    "parse_intent": "🔍 Parsing intent...",
    "ask_clarification": "❓ Clarification needed...",
    "connect_and_profile_data": "📊 Profiling data...",
    "plan_analysis": "🧠 Planning analysis...",
    "execute_analysis": "🧮 Running analysis...",
    "generate_viz": "🎨 Generating visualizations...",
    "retrieve_domain_knowledge": "📚 Retrieving domain knowledge...",
    "generate_insights": "💡 Generating insights...",
    "run_guardrails": "🛡️ Running guardrails...",
    "run_eval": "🎯 Evaluating output...",
    "handle_error": "⚠️ Handling error...",
}


def build_initial_state(query: str, data_source_path: Optional[str]) -> MAEDAState:
    """Build the initial MAEDAState for a query + optional data source path,
    inferring the source type from the file extension (SQL sources get a
    sqlite:/// connection string rather than a bare path)."""
    state = initial_state(query)
    if data_source_path:
        ext = data_source_path.rsplit(".", 1)[-1].lower() if "." in data_source_path else "csv"
        type_map = {"csv": "csv", "json": "json", "xlsx": "excel",
                    "xls": "excel", "db": "sql", "sqlite": "sql"}
        src_type = type_map.get(ext, "csv")
        src_path = f"sqlite:///{data_source_path}" if src_type == "sql" else data_source_path
        state["data_sources"] = [{"path": src_path, "type": src_type}]
    return state


async def astream_pipeline(state: MAEDAState) -> AsyncIterator[tuple[str, MAEDAState]]:
    """Yield (node_name, state_after_node) as each graph node completes.

    MAEDAState is a plain TypedDict with no reducer annotations, and every
    node function returns the full state object (not a partial patch) --
    so LangGraph's "updates" stream mode yields the complete, valid state
    so far after each node, not a diff. The last state yielded is
    equivalent to what graph.ainvoke(state) would have returned.
    """
    graph = build_graph()
    async for chunk in graph.astream(state, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            yield node_name, node_state


def run_pipeline_streaming(
    query: str,
    data_source_path: Optional[str],
    on_node: Optional[Callable[[str, MAEDAState], None]] = None,
) -> MAEDAState:
    """Synchronous entry point for callers with no event loop of their own
    (e.g. Streamlit, which reruns its script synchronously per interaction).

    Drives astream_pipeline() to completion under a single asyncio.run(),
    invoking on_node(node_name, state_so_far) as each node completes, and
    returns the final accumulated state.
    """
    state = build_initial_state(query, data_source_path)

    async def _drive() -> MAEDAState:
        final_state = state
        async for node_name, node_state in astream_pipeline(state):
            final_state = node_state
            if on_node is not None:
                on_node(node_name, node_state)
        return final_state

    return asyncio.run(_drive())
