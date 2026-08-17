"""
Session-wide test fixtures.

附录 E4 / 附录 AR.2 / 附录 AS (ECOSYSTEM_INTEGRATION_PLAN.md): RunStore's
module-level lazy singleton (src/graph/nodes.py's `_get_run_store`)
constructs `RunStore()` with no explicit path whenever nothing else has
initialized it yet, which falls back to `settings.runs_db_path` --
`logs/runs.db` by default, a real file a real user relies on. Any test
that runs the graph far enough to reach `persist_run_node` (wired as the
terminal node on every path, per run_store.py's own docstring) writes
real rows into that live database.

This was caught after the fact (附录 AR.2): four unrelated test-suite runs
had already added 12 real rows to the user's actual `logs/runs.db`
(968 -> 980), identifiable via `cleaning_applied_level IS NOT NULL` (that
column didn't exist before this same round of work, so no genuine
historical run could have a non-null value there) -- those rows are the
user's data now and are deliberately left alone here; this fixture only
prevents *further* pollution.

`_isolate_runs_db` is autouse so protection does not depend on any
individual test file remembering to opt in -- exactly the gap that let
the pollution happen in the first place. It redirects
`settings.runs_db_path` to a fresh location under this test's own
`tmp_path` before every test, and resets the graph module's cached
`RunStore` singleton so the *next* thing that touches it (this test, if
it reaches persist_run_node) re-initializes against the redirected path
rather than reusing a previous test's (or, on the very first test in a
session, the real) instance.

Tests that need their own explicit RunStore/path (e.g.
tests/unit/test_run_store.py's own suite, which passes explicit tmp_path
locations throughout) are unaffected: this fixture only supplies the
*fallback* default; any test-local `monkeypatch.setattr(settings,
"runs_db_path", ...)` or direct `RunStore(explicit_path)` construction
still works exactly as before, since those don't go through the fallback
at all (or run in the same test after this fixture applies, and simply
override it via the same underlying `monkeypatch` fixture instance).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_runs_db(tmp_path, monkeypatch):
    from src.config.settings import settings

    monkeypatch.setattr(settings, "runs_db_path", str(tmp_path / "test_runs.db"))

    import src.graph.nodes as nodes
    nodes._run_store = None
    yield
    nodes._run_store = None
