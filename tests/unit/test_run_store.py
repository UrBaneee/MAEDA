"""
Tests for src/persistence/run_store.py — roadmap #20 (persist
decision_trace/mcp_call_log so a run is auditable after the process exits).
"""
import asyncio

import pytest

from src.state.graph_state import initial_state


def _state_with_trace(query: str = "Show revenue by region") -> dict:
    state = initial_state(query)
    state["decision_trace"] = [
        {"agent_name": "intent_parser", "action": "parse_intent",
         "reasoning": "Query type: descriptive", "confidence": 0.9,
         "timestamp": "2026-01-01T00:00:00Z", "inputs": None, "outputs": None},
    ]
    state["mcp_call_log"] = [
        {"system": "data_cleaner", "tool": "profile_dataset", "mode": "fallback",
         "args": {}, "result_summary": "12240 rows", "duration_ms": 5.0, "error": None},
    ]
    state["current_phase"] = "complete"
    state["guardrail_passed"] = True
    return state


# ─── RunStore ──────────────────────────────────────────────────────────────────

def test_save_and_get_run_round_trip(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()

    run_id = store.save_run(state)
    assert run_id == state["run_id"]

    fetched = store.get_run(run_id)
    assert fetched is not None
    assert fetched["user_query"] == "Show revenue by region"
    assert fetched["current_phase"] == "complete"
    assert fetched["guardrail_passed"] is True
    assert fetched["decision_trace"] == state["decision_trace"]
    assert fetched["mcp_call_log"] == state["mcp_call_log"]


def test_get_run_missing_returns_none(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    assert store.get_run("nonexistent") is None


def test_save_run_persists_error_fields(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()
    state["current_phase"] = "error"
    state["error"] = "Guardrail checks failed after maximum retries"
    state["error_type"] = "safe_refusal"

    store.save_run(state)
    fetched = store.get_run(state["run_id"])
    assert fetched["error_type"] == "safe_refusal"
    assert fetched["error"] == "Guardrail checks failed after maximum retries"


def test_save_run_persists_eval_scores_when_present(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()
    state["eval_scores"] = {"_aggregate": 0.85, "answer_relevance": {"score": 0.9, "label": "pass"}}

    store.save_run(state)
    fetched = store.get_run(state["run_id"])
    assert fetched["eval_scores"]["_aggregate"] == 0.85


def test_save_run_upserts_on_same_run_id(tmp_path):
    """Calling save_run twice for the same run_id overwrites, not duplicates."""
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()

    store.save_run(state)
    state["current_phase"] = "complete"
    state["decision_trace"].append({"agent_name": "guardrail_agent", "action": "run_guardrails",
                                     "reasoning": "passed", "confidence": 1.0,
                                     "timestamp": "2026-01-01T00:00:05Z", "inputs": None, "outputs": None})
    store.save_run(state)

    fetched = store.get_run(state["run_id"])
    assert len(fetched["decision_trace"]) == 2
    assert len(store.list_runs()) == 1


def test_list_runs_orders_most_recent_first(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    store.save_run(_state_with_trace("first query"))
    store.save_run(_state_with_trace("second query"))

    runs = store.list_runs()
    assert len(runs) == 2
    assert {r["user_query"] for r in runs} == {"first query", "second query"}


def test_list_runs_respects_limit(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    for i in range(5):
        store.save_run(_state_with_trace(f"query {i}"))
    assert len(store.list_runs(limit=2)) == 2


def test_list_runs_summary_omits_large_json_fields(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    store.save_run(_state_with_trace())
    summary = store.list_runs()[0]
    assert "decision_trace" not in summary
    assert "mcp_call_log" not in summary


def test_run_store_uses_settings_default_path(tmp_path, monkeypatch):
    from src.config.settings import settings
    monkeypatch.setattr(settings, "runs_db_path", str(tmp_path / "settings_default.db"))
    from src.persistence.run_store import RunStore
    store = RunStore()
    store.save_run(_state_with_trace())
    assert (tmp_path / "settings_default.db").exists()


# ─── persist_run_node integration ─────────────────────────────────────────────

def test_persist_run_node_saves_and_returns_state(tmp_path, monkeypatch):
    import src.graph.nodes as nodes
    from src.config.settings import settings
    monkeypatch.setattr(settings, "runs_db_path", str(tmp_path / "runs.db"))
    nodes._run_store = None  # force re-init against the patched path

    state = _state_with_trace()
    result = nodes.persist_run_node(state)
    assert result is state

    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    assert store.get_run(state["run_id"]) is not None
    nodes._run_store = None


def test_persist_run_node_never_raises_on_storage_failure(monkeypatch):
    """A persistence failure must not break the pipeline the user is
    waiting on -- caught and logged, not propagated."""
    import src.graph.nodes as nodes

    class _BoomStore:
        def save_run(self, state):
            raise RuntimeError("disk full")

    monkeypatch.setattr(nodes, "_get_run_store", lambda: _BoomStore())
    state = _state_with_trace()
    result = nodes.persist_run_node(state)  # must not raise
    assert result is state
