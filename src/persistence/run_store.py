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
    cleaning_applied_level TEXT,
    cleaning_stop_reason   TEXT,
    terminal_state      TEXT,
    terminal_detail     TEXT,
    eval_error          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# 附录 R.3 / 附录 AO.1 / 附录 AP.2: these two columns didn't exist when this
# table was first created, and this store is written to a real, already
# populated `logs/runs.db` in this environment -- `CREATE TABLE IF NOT
# EXISTS` above is a no-op against an existing table, so it alone would
# never add them to a pre-existing database. `_migrate_schema` below adds
# any missing column via `ALTER TABLE ADD COLUMN` (additive only: existing
# rows get NULL for the new columns, nothing is dropped, rewritten, or
# recreated). This runs unconditionally on every `init_schema` call, so a
# brand-new database (already created with these columns by `_SCHEMA`
# above) just finds nothing missing and no-ops.
# E3 (附录 CU) adds three more, by the same additive ALTER TABLE route:
# `terminal_state`/`terminal_detail` (how the run ended, see
# src/state/terminal_state.py) and `eval_error` (eval itself failed).
# NOTE for anyone reading logs/runs.db: `terminal_state IS NULL` marks a row
# written BEFORE E3, and such a row's `error_type = 'pipeline_error'` covers
# the MCP and environment failures that now get their own terminal state --
# it must not be read as "agent reasoning failure".
_NEW_COLUMNS = ("cleaning_applied_level", "cleaning_stop_reason",
                "terminal_state", "terminal_detail", "eval_error")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col in _NEW_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")


def init_schema(db_path: str) -> None:
    """Create the runs table if it doesn't already exist, and migrate an
    existing table forward to the current column set (additive only)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _migrate_schema(conn)


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
                     eval_scores_json, cleaning_applied_level, cleaning_stop_reason,
                     terminal_state, terminal_detail, eval_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    # 附录 R.3: diagnostic fields, not derived at read time --
                    # see nodes.py's _cleaning_applied_level docstring for why
                    # this value must be captured here rather than recomputed
                    # from cleaning_stop_reason after the fact (the "full" vs
                    # "none" distinction also depends on cleaning_applied,
                    # which isn't persisted separately).
                    state.get("cleaning_applied_level"),
                    state.get("cleaning_stop_reason"),
                    # E3 (附录 CU): the terminal classification, persisted
                    # rather than recoverable only by string-matching
                    # decision_trace -- the same argument 附录 R.3/AO.1 made
                    # for cleaning_applied_level, applied to the field 阶段 4
                    # needs in order to count mcp_error/environment_error
                    # separately from agent reasoning failures at all.
                    state.get("terminal_state"),
                    state.get("terminal_detail"),
                    state.get("eval_error"),
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
                       error, error_type, terminal_state, terminal_detail,
                       eval_error, created_at
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
