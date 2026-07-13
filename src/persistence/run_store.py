"""
Persists decision_trace/mcp_call_log to SQLite so they survive past the
process that produced them.

Previously both were pure in-memory MAEDAState fields: real during a run,
gone the moment the graph finished and (for the CLI/eval harness) the
process exited, or (for the Streamlit UI) the moment the session ended or
the server restarted. "Every agent decision must be logged to
decision_trace" (CLAUDE.md) was true but hollow without this -- there was
no way to audit a run after the fact.

RunStore is called from graph/nodes.py's persist_run_node, wired as the
terminal node before END on every path (run_eval and handle_error both
route through it) -- every pipeline invocation gets persisted exactly
once, success or failure, without any node upstream needing to know this
exists.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.state.graph_state import MAEDAState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
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


def init_schema(db_path: str) -> None:
    """Create the runs table if it doesn't already exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)


class RunStore:
    """SQLite-backed store for completed pipeline runs."""

    def __init__(self, db_path: Optional[str] = None):
        from src.config.settings import settings
        self._db_path = db_path or settings.runs_db_path
        init_schema(self._db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(self, state: MAEDAState) -> str:
        """Persist one completed run. Returns the run_id it was saved under.

        Upserts on run_id so calling this more than once for the same run
        (shouldn't happen in the graph, but harmless if it does) overwrites
        rather than duplicates.
        """
        run_id = state.get("run_id") or ""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, user_query, current_phase, guardrail_passed,
                     error, error_type, decision_trace_json, mcp_call_log_json,
                     eval_scores_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    state.get("user_query", ""),
                    state.get("current_phase"),
                    int(bool(state.get("guardrail_passed"))),
                    state.get("error"),
                    state.get("error_type"),
                    json.dumps(state.get("decision_trace") or [], default=str),
                    json.dumps(state.get("mcp_call_log") or [], default=str),
                    json.dumps(state.get("eval_scores"), default=str)
                    if state.get("eval_scores") is not None else None,
                ),
            )
        return run_id

    def get_run(self, run_id: str) -> Optional[dict]:
        """Retrieve one persisted run by id, with JSON fields decoded."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return a summary of the most recent runs (most recent first).

        Summary only -- decision_trace/mcp_call_log are omitted since they
        can be large; call get_run(run_id) for the full record.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, user_query, current_phase, guardrail_passed,
                       error, error_type, created_at
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["decision_trace"] = json.loads(d.pop("decision_trace_json"))
        d["mcp_call_log"] = json.loads(d.pop("mcp_call_log_json"))
        eval_scores_json = d.pop("eval_scores_json")
        d["eval_scores"] = json.loads(eval_scores_json) if eval_scores_json else None
        d["guardrail_passed"] = bool(d["guardrail_passed"])
        return d
