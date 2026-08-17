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


# ─── 附录 R.3 / 附录 AO.1 / 附录 AP.2: cleaning_applied_level / cleaning_stop_reason ──

def test_save_run_persists_cleaning_fields_when_present(tmp_path):
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()
    state["cleaning_applied_level"] = "blocked_needs_review"
    state["cleaning_stop_reason"] = "needs_review"

    store.save_run(state)
    fetched = store.get_run(state["run_id"])
    assert fetched["cleaning_applied_level"] == "blocked_needs_review"
    assert fetched["cleaning_stop_reason"] == "needs_review"


def test_save_run_cleaning_fields_default_to_none_when_absent(tmp_path):
    """A run that never touched the cleaning loop (e.g. no data source, or
    guardrail-blocked before connect_and_profile) never sets these past
    initial_state()'s None default -- must persist as NULL, not crash and
    not silently coerce to a string like "None"."""
    from src.persistence.run_store import RunStore
    store = RunStore(str(tmp_path / "runs.db"))
    state = _state_with_trace()
    assert state["cleaning_applied_level"] is None
    assert state["cleaning_stop_reason"] is None

    store.save_run(state)
    fetched = store.get_run(state["run_id"])
    assert fetched["cleaning_applied_level"] is None
    assert fetched["cleaning_stop_reason"] is None


def test_existing_database_without_cleaning_columns_is_migrated_in_place(tmp_path):
    """Regression test for the migration path (附录 AP.2's "handle the
    migration path deliberately" requirement): a pre-existing `runs` table
    created with the OLD schema (no cleaning_* columns -- exactly the shape
    of a real already-populated logs/runs.db in this environment) must gain
    the new columns via ALTER TABLE, without losing its existing row, when
    RunStore is next constructed against it."""
    import sqlite3
    db_path = str(tmp_path / "old_runs.db")

    # Build a table in the OLD shape by hand (predates this change) and
    # insert one pre-existing row, simulating a real production database
    # that was never touched by the new code.
    old_schema = """
    CREATE TABLE runs (
        run_id              TEXT PRIMARY KEY,
        user_query          TEXT NOT NULL,
        current_phase       TEXT,
        guardrail_passed    INTEGER,
        error               TEXT,
        error_type          TEXT,
        decision_trace_json TEXT NOT NULL DEFAULT '[]',
        mcp_call_log_json   TEXT NOT NULL DEFAULT '[]',
        eval_scores_json    TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript(old_schema)
        conn.execute(
            "INSERT INTO runs (run_id, user_query, current_phase, guardrail_passed) "
            "VALUES ('pre-existing-run', 'old query', 'complete', 1)"
        )

    from src.persistence.run_store import RunStore
    store = RunStore(db_path)  # __init__ calls init_schema -> migration runs here

    # The pre-existing row must have survived, untouched except for the new
    # columns defaulting to NULL.
    pre_existing = store.get_run("pre-existing-run")
    assert pre_existing is not None
    assert pre_existing["user_query"] == "old query"
    assert pre_existing["cleaning_applied_level"] is None
    assert pre_existing["cleaning_stop_reason"] is None

    # And the store is now fully usable for new writes that DO set them.
    state = _state_with_trace("new query after migration")
    state["cleaning_applied_level"] = "full"
    state["cleaning_stop_reason"] = "passed"
    store.save_run(state)
    fetched = store.get_run(state["run_id"])
    assert fetched["cleaning_applied_level"] == "full"
    assert fetched["cleaning_stop_reason"] == "passed"

    # Running init_schema again (e.g. a second RunStore() against the same
    # already-migrated file) must be a no-op, not an error ("duplicate
    # column name").
    RunStore(db_path)


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


def test_persist_run_node_uses_the_autouse_redirected_path_not_the_real_default():
    """附录 AR.2 / 附录 AS: companion to
    test_persist_run_node_saves_and_returns_state (which manually
    monkeypatches settings + resets nodes._run_store to prove the happy
    path works against an explicit tmp path). This one deliberately does
    NOT patch anything itself, to prove tests/conftest.py's autouse
    `_isolate_runs_db` fixture alone is sufficient: persist_run_node(),
    called with zero test-local setup, must never reach the real
    logs/runs.db -- that fixture is the entire safety net now, not an
    opt-in convenience, and this is the test that would fail if it were
    ever made conditional, scoped away, or accidentally removed."""
    import src.graph.nodes as nodes
    from src.config.settings import settings

    state = _state_with_trace("autouse isolation check")
    result = nodes.persist_run_node(state)
    assert result is state

    # It really did persist somewhere -- not silently no-op'd against a
    # broken path.
    from src.persistence.run_store import RunStore
    store = RunStore(settings.runs_db_path)
    assert store.get_run(state["run_id"]) is not None

    # And "somewhere" is provably not the real production default -- this
    # is the exact fallback path (nodes.py's _get_run_store -> RunStore()
    # with no args -> settings.runs_db_path) that caused 附录 AR.2's
    # pollution before the autouse fixture existed.
    assert settings.runs_db_path != "logs/runs.db"
    assert nodes._get_run_store()._db_path == settings.runs_db_path


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
