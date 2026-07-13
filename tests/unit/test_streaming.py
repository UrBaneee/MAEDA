"""
Tests for src/graph/streaming.py — roadmap #14 (real per-node progress).

Kept independent of ui/app.py (Streamlit-specific, executes page-rendering
calls at import time) so this logic can be tested without a Streamlit
session: build_graph() is mocked with a lightweight fake compiled graph
whose astream() yields controlled chunks, rather than running the full
9-node pipeline (already covered by the end-to-end graph tests in
test_phase1.py).
"""
from unittest.mock import patch

import pytest


class _FakeCompiledGraph:
    """Stands in for graph.astream() with a fixed, controlled chunk sequence."""

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks

    async def astream(self, state, stream_mode="updates"):
        for chunk in self._chunks:
            yield chunk


# ─── build_initial_state ──────────────────────────────────────────────────────

def test_build_initial_state_no_source():
    from src.graph.streaming import build_initial_state
    state = build_initial_state("Show revenue", None)
    assert state["user_query"] == "Show revenue"
    assert state["data_sources"] == []


def test_build_initial_state_csv():
    from src.graph.streaming import build_initial_state
    state = build_initial_state("q", "data/sales.csv")
    assert state["data_sources"] == [{"path": "data/sales.csv", "type": "csv"}]


def test_build_initial_state_excel():
    from src.graph.streaming import build_initial_state
    state = build_initial_state("q", "data/sales.xlsx")
    assert state["data_sources"] == [{"path": "data/sales.xlsx", "type": "excel"}]


def test_build_initial_state_sql_gets_connection_string():
    from src.graph.streaming import build_initial_state
    state = build_initial_state("q", "data/orders.db")
    assert state["data_sources"] == [{"path": "sqlite:///data/orders.db", "type": "sql"}]


def test_build_initial_state_unknown_extension_defaults_to_csv():
    from src.graph.streaming import build_initial_state
    state = build_initial_state("q", "data/mystery")
    assert state["data_sources"] == [{"path": "data/mystery", "type": "csv"}]


# ─── run_pipeline_streaming ───────────────────────────────────────────────────

def test_run_pipeline_streaming_calls_on_node_for_each_chunk_in_order():
    from src.graph.streaming import run_pipeline_streaming
    chunks = [
        {"parse_intent": {"user_query": "q", "current_phase": "plan"}},
        {"execute_analysis": {"user_query": "q", "current_phase": "execute"}},
        {"run_eval": {"user_query": "q", "current_phase": "complete"}},
    ]
    fake_graph = _FakeCompiledGraph(chunks)
    seen = []
    with patch("src.graph.streaming.build_graph", return_value=fake_graph):
        result = run_pipeline_streaming("q", None, on_node=lambda name, state: seen.append(name))

    assert seen == ["parse_intent", "execute_analysis", "run_eval"]
    assert result["current_phase"] == "complete"


def test_run_pipeline_streaming_works_without_on_node_callback():
    """on_node is optional -- must not require a callback to function."""
    from src.graph.streaming import run_pipeline_streaming
    chunks = [{"run_eval": {"user_query": "q", "current_phase": "complete"}}]
    fake_graph = _FakeCompiledGraph(chunks)
    with patch("src.graph.streaming.build_graph", return_value=fake_graph):
        result = run_pipeline_streaming("q", None)
    assert result["current_phase"] == "complete"


def test_run_pipeline_streaming_returns_initial_state_if_graph_yields_nothing():
    """A graph that (hypothetically) never yields must still return a valid
    state rather than None or raising."""
    from src.graph.streaming import run_pipeline_streaming
    fake_graph = _FakeCompiledGraph([])
    with patch("src.graph.streaming.build_graph", return_value=fake_graph):
        result = run_pipeline_streaming("Show revenue", None)
    assert result["user_query"] == "Show revenue"


def test_run_pipeline_streaming_propagates_on_node_exception():
    """A bug in the UI callback should surface, not be silently swallowed."""
    from src.graph.streaming import run_pipeline_streaming
    chunks = [{"parse_intent": {"user_query": "q"}}]
    fake_graph = _FakeCompiledGraph(chunks)

    def _boom(name, state):
        raise RuntimeError("UI callback exploded")

    with patch("src.graph.streaming.build_graph", return_value=fake_graph):
        with pytest.raises(RuntimeError, match="UI callback exploded"):
            run_pipeline_streaming("q", None, on_node=_boom)


# ─── NODE_LABELS ───────────────────────────────────────────────────────────────

def test_node_labels_cover_every_registered_graph_node():
    """Every node name the graph can actually emit should have a human
    label -- an unrecognized name would fall back to a generic "Running
    {name}..." string in the UI, which is a degraded but non-broken result,
    so this is a completeness check, not a hard runtime dependency."""
    from src.graph.builder import build_graph
    from src.graph.streaming import NODE_LABELS

    g = build_graph()
    graph_nodes = set(g.get_graph().nodes) - {"__start__", "__end__"}
    assert graph_nodes.issubset(NODE_LABELS.keys())
