"""
E3 — 失败路径接入 eval (ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3 执行顺序表轮次 5,
附录 CU).

Covers the three things E3 asks for and one thing it deliberately does NOT
do:
  * terminal states classified once and persisted (not recoverable only by
    string-matching decision_trace);
  * a failed run scored on the metrics it can support, with the rest marked
    `not_applicable` -- and, crucially, WITHOUT invoking the LLM judge, which
    is what makes routing the failure path through eval free;
  * eval failing captured and classified instead of taking the run record
    down with it;
  * a failed run is NOT excluded row-wise from trial aggregation (see
    src/eval/trials.py::not_applicable_reason).
"""
import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.state import terminal_state as ts
from src.state.graph_state import initial_state


# ─── The vocabulary and the single gate ──────────────────────────────────────

def test_every_fallback_error_class_is_mapped_to_a_terminal_state():
    """
    ANTI-DRIFT: E3's terminal_detail is fallback.py's error_class verbatim,
    not a second taxonomy. Drives the real `_classify` rather than asserting
    against a copied list, so a fifth error class added there cannot silently
    fall through to "pipeline_error" here.
    """
    from src.mcp_client.client import MCPConnectionError, MCPContractError, MCPToolError
    from src.mcp_client.fallback import _classify

    exceptions = [
        MCPConnectionError("down"),
        MCPContractError("bad contract"),
        MCPToolError("unreadable", error_type="FileNotFoundError"),
        MCPToolError("something", error_type=None),
        ValueError("not an mcp error at all"),
    ]
    seen = {_classify(exc)[0] for exc in exceptions}
    assert seen == {"connection", "contract", "data_input", "internal_unknown"}, seen
    for error_class in seen:
        assert error_class in ts._DETAIL_TO_TERMINAL, error_class
        assert ts._DETAIL_TO_TERMINAL[error_class] in ts.TERMINAL_STATES


def test_success_state_resolves_to_success():
    state = initial_state("q")
    state["current_phase"] = "complete"
    assert ts.resolve_terminal_state(state) == (ts.SUCCESS, None)


def test_guardrail_rejection_is_the_safe_refusal_state():
    """The one place E3's six-state list collapses to five: guardrail
    rejection and safe refusal name the same event in this repo (see the
    module docstring of src/state/terminal_state.py)."""
    state = initial_state("q")
    state["current_phase"] = "error"
    state["error"] = "blocked"
    state["guardrail_checks"] = [{"overall_verdict": "fail"}]
    assert ts.resolve_terminal_state(state) == (ts.SAFE_REFUSAL, ts.DETAIL_GUARDRAIL_FAIL)


@pytest.mark.parametrize("detail,expected", [
    ("connection", ts.MCP_ERROR),
    ("contract", ts.MCP_ERROR),
    ("internal_unknown", ts.MCP_ERROR),
    (ts.DETAIL_SCOPE_FINGERPRINT_MISMATCH, ts.MCP_ERROR),
    ("data_input", ts.ENVIRONMENT_ERROR),
    (ts.DETAIL_NO_DATA_SOURCE, ts.ENVIRONMENT_ERROR),
    (None, ts.PIPELINE_ERROR),
    ("uncaught:KeyError", ts.PIPELINE_ERROR),
])
def test_detail_maps_to_terminal_state(detail, expected):
    state = initial_state("q")
    state["current_phase"] = "error"
    state["error"] = "boom"
    state["terminal_detail"] = detail
    assert ts.resolve_terminal_state(state)[0] == expected


def test_resolve_is_idempotent_and_never_reclassifies_a_stored_value():
    """One judgment site: handle_error_node stores, everyone else reads."""
    state = initial_state("q")
    state["current_phase"] = "error"
    state["error"] = "boom"
    state["terminal_state"] = ts.MCP_ERROR
    state["terminal_detail"] = "connection"
    assert ts.resolve_terminal_state(state) == (ts.MCP_ERROR, "connection")
    # Even if the raw signals would say something else now.
    state["guardrail_checks"] = [{"overall_verdict": "fail"}]
    assert ts.resolve_terminal_state(state) == (ts.MCP_ERROR, "connection")


def test_legacy_error_type_projection_matches_pre_e3_vocabulary():
    assert ts.legacy_error_type(ts.SUCCESS) is None
    assert ts.legacy_error_type(ts.SAFE_REFUSAL) == "safe_refusal"
    for state in (ts.PIPELINE_ERROR, ts.MCP_ERROR, ts.ENVIRONMENT_ERROR):
        assert ts.legacy_error_type(state) == "pipeline_error"


# ─── handle_error_node writes the classification ─────────────────────────────

def test_handle_error_classifies_mcp_failure_without_losing_error_type():
    from src.graph.nodes import handle_error_node

    state = initial_state("q")
    state["error"] = "Data cleaning failed (connection): down"
    state["terminal_detail"] = "connection"
    out = handle_error_node(state)

    assert out["terminal_state"] == ts.MCP_ERROR
    assert out["terminal_detail"] == "connection"
    # The pre-E3 field keeps its pre-E3 value, for score_system_metrics and
    # the runs.db column.
    assert out["error_type"] == "pipeline_error"
    assert out["current_phase"] == "error"


def test_handle_error_classifies_no_data_source_as_environment_error():
    from src.graph.nodes import handle_error_node

    state = initial_state("q")
    state["error"] = "No data source provided."
    state["terminal_detail"] = ts.DETAIL_NO_DATA_SOURCE
    out = handle_error_node(state)
    assert out["terminal_state"] == ts.ENVIRONMENT_ERROR
    assert out["error_type"] == "pipeline_error"


def test_handle_error_classifies_guardrail_fail_as_safe_refusal():
    from src.graph.nodes import handle_error_node

    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "fail", "retry_reason": "ungrounded"}]
    out = handle_error_node(state)
    assert out["terminal_state"] == ts.SAFE_REFUSAL
    assert out["error_type"] == "safe_refusal"
    assert out["error"] == "ungrounded"


def test_connect_schema_marks_missing_data_source_as_environment_error():
    """The classification has to originate where the failure does -- a
    detail set only in handle_error_node could never distinguish this from
    an agent crash."""
    from src.graph.nodes import connect_schema

    state = initial_state("q")
    state["data_sources"] = []
    out = asyncio.run(connect_schema(state))
    assert out["terminal_detail"] == ts.DETAIL_NO_DATA_SOURCE
    assert ts.resolve_terminal_state(out)[0] == ts.ENVIRONMENT_ERROR


# ─── The graph now routes failures through eval ──────────────────────────────

def test_handle_error_routes_into_run_eval_not_straight_to_persist():
    """E3's headline: 失败路径接入 eval. handle_error must be UPSTREAM of
    run_eval (it is what writes terminal_state, which run_eval reads)."""
    from src.graph.builder import build_graph

    graph = build_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("handle_error", "run_eval") in edges
    assert ("handle_error", "persist_run") not in edges
    assert ("run_eval", "persist_run") in edges


# ─── EvalRunner: failed runs are scored on less, not on zero ─────────────────

def _judge_that_must_not_be_called():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=AssertionError(
        "the LLM judge must not be invoked for a run that did not terminate "
        "in success -- that is what makes routing the failure path through "
        "eval free"))
    return llm


def _failed_state(**over):
    state = initial_state("Show revenue by region")
    state["current_phase"] = "error"
    state["error"] = "Data quality profiling failed (connection): down"
    state["terminal_state"] = ts.MCP_ERROR
    state["terminal_detail"] = "connection"
    state["error_type"] = "pipeline_error"
    state.update(over)
    return state


def _by_metric(result):
    return {s.metric: s for s in result.scores}


def test_failed_run_marks_answer_metrics_not_applicable_and_calls_no_judge():
    from src.eval.runner import EvalRunner
    from src.eval.trials import NOT_APPLICABLE

    runner = EvalRunner(llm=_judge_that_must_not_be_called())
    result = asyncio.run(runner.score(_failed_state()))
    scores = _by_metric(result)

    for metric in ("answer_relevance", "groundedness", "factual_accuracy"):
        assert scores[metric].label == NOT_APPLICABLE, metric
        assert scores[metric].valid is False, metric
        assert "mcp_error" in scores[metric].reasoning, metric


def test_failed_run_still_scores_the_metrics_it_can_support():
    """"只算适用指标", not "算零个指标": error_rate and safe_refusal are the
    metrics a failed run exists to report."""
    from src.eval.runner import EvalRunner

    runner = EvalRunner(llm=_judge_that_must_not_be_called())
    result = asyncio.run(runner.score(_failed_state()))
    scores = _by_metric(result)

    assert scores["error_rate"].valid is True
    assert scores["error_rate"].score == 0.0     # a real measurement: this run crashed
    assert scores["safe_refusal"].valid is True
    assert scores["safe_refusal"].score == 0.0   # it was not a refusal, it was an mcp_error
    assert scores["token_cost"].valid is True


def test_failed_run_scores_intent_accuracy_when_an_intent_was_parsed():
    """A run that parsed its intent and then died at profiling gives a REAL
    intent_accuracy measurement -- discarding it would lose signal E3 wants
    kept."""
    from src.eval.runner import EvalRunner
    from src.eval.trials import NOT_APPLICABLE

    runner = EvalRunner(llm=_judge_that_must_not_be_called())
    with_intent = _failed_state(parsed_intent={
        "query_type": "descriptive", "confidence": 0.9, "target_metrics": ["revenue"],
    })
    scored = _by_metric(asyncio.run(runner.score(with_intent)))["intent_accuracy"]
    assert scored.valid is True
    assert scored.score > 0.0

    without = _by_metric(asyncio.run(runner.score(_failed_state())))["intent_accuracy"]
    assert without.label == NOT_APPLICABLE
    assert without.valid is False


def test_failed_run_has_no_comparable_aggregate():
    """None, not 0.0 -- the same distinction trials.pass_at_k already makes.
    A near-zero aggregate here IS the "打零分" E3 removes."""
    from src.eval.runner import EvalRunner

    runner = EvalRunner(llm=_judge_that_must_not_be_called())
    result = asyncio.run(runner.score(_failed_state()))
    assert result.aggregate_score is None
    assert result.terminal_state == ts.MCP_ERROR
    assert result.terminal_detail == "connection"
    assert result.to_dict()["terminal_state"] == ts.MCP_ERROR


def test_guardrail_refused_run_is_not_judged_on_the_report_it_withheld():
    """A safe refusal still HAS a report in state -- the text the pipeline
    decided not to deliver. Judging it would look like a quality measurement
    and cost real money to produce."""
    from src.eval.runner import EvalRunner
    from src.eval.trials import NOT_APPLICABLE

    state = _failed_state(
        report="# Report\n\nUngrounded claims here.",
        terminal_state=ts.SAFE_REFUSAL,
        terminal_detail=ts.DETAIL_GUARDRAIL_FAIL,
        error_type="safe_refusal",
        analysis_results=[{"method": "groupby", "result_summary": "North=500K", "failed": False}],
    )
    runner = EvalRunner(llm=_judge_that_must_not_be_called())
    scores = _by_metric(asyncio.run(runner.score(state)))
    assert scores["answer_relevance"].label == NOT_APPLICABLE
    assert scores["safe_refusal"].score == 1.0
    # Steps really did execute -- that measurement survives.
    assert scores["step_success_rate"].valid is True


def test_successful_run_is_scored_exactly_as_before(monkeypatch):
    """E3 must not move any 阶段 4 baseline number: on the success path
    _not_applicable_metrics returns empty and nothing is skipped, including
    for a successful run with no charts."""
    from src.eval.metrics import MetricScore
    from src.eval.runner import EvalRunner, _not_applicable_metrics
    from src.eval.trials import NOT_APPLICABLE

    state = initial_state("Show revenue by region")
    state["current_phase"] = "complete"
    state["report"] = "# Report\n\nNorth: 500K."
    state["charts"] = []
    state["analysis_results"] = []
    assert _not_applicable_metrics(state, ts.SUCCESS) == {}

    async def _fake_relevance(*a, **k):
        return MetricScore("answer_relevance", 0.9, "pass")

    async def _fake_groundedness(*a, **k):
        return MetricScore("groundedness", 0.8, "pass")

    monkeypatch.setattr("src.eval.runner.score_answer_relevance", _fake_relevance)
    monkeypatch.setattr("src.eval.runner.score_groundedness", _fake_groundedness)

    result = asyncio.run(EvalRunner().score(state))
    assert result.terminal_state == ts.SUCCESS
    assert isinstance(result.aggregate_score, float)
    assert all(s.label != NOT_APPLICABLE for s in result.scores)


def test_not_applicable_entries_cannot_move_the_aggregate():
    """The reuse argument: valid=False already means "no measurement", and
    _aggregate_score already skips those."""
    from src.eval.metrics import MetricScore, not_applicable_metric
    from src.eval.runner import _aggregate_score

    base = [MetricScore("answer_relevance", 0.8, "pass"), MetricScore("groundedness", 0.6, "pass")]
    assert _aggregate_score(base) == _aggregate_score(
        base + [not_applicable_metric("factual_accuracy", "terminated")]
    )


def test_a_raising_metric_is_captured_not_propagated(monkeypatch):
    """eval 自身失败也要捕获分类: one broken metric must not cost the caller
    every other metric."""
    from src.eval.metrics import MetricScore
    from src.eval.runner import EvalRunner

    def _boom(*a, **k):
        raise RuntimeError("metric exploded")

    monkeypatch.setattr("src.eval.runner.score_factual_accuracy", _boom)

    async def _fake(*a, **k):
        return MetricScore("answer_relevance", 0.9, "pass")

    state = initial_state("q")
    state["current_phase"] = "complete"
    state["report"] = "# Report"
    monkeypatch.setattr("src.eval.runner.score_answer_relevance", _fake)
    monkeypatch.setattr("src.eval.runner.score_groundedness", _fake)

    result = asyncio.run(EvalRunner().score(state))
    factual = _by_metric(result)["factual_accuracy"]
    assert factual.valid is False
    assert factual.label == "error"
    assert "RuntimeError" in factual.reasoning
    assert any(s.metric == "error_rate" for s in result.scores)


# ─── eval failing must not destroy the run record ────────────────────────────

def test_run_eval_node_captures_its_own_failure_and_still_returns(monkeypatch):
    from src.graph import nodes

    class _Exploding:
        async def score(self, *a, **k):
            raise RuntimeError("judge client misconfigured")

    monkeypatch.setattr(nodes, "_get_eval_runner", lambda: _Exploding())
    state = initial_state("q")
    out = asyncio.run(nodes.run_eval_node(state))
    assert "RuntimeError" in out["eval_error"]
    assert out.get("eval_scores") is None


def test_run_eval_node_does_not_overwrite_the_error_phase(monkeypatch):
    """It is now on the failure path too; stamping current_phase="complete"
    there would erase the phase every persisted failure row records."""
    from src.eval.metrics import MetricScore
    from src.eval.runner import EvalResult
    from src.graph import nodes

    class _Runner:
        async def score(self, state, **k):
            return EvalResult(run_id="r", query="q",
                              scores=[MetricScore("error_rate", 0.0, "fail")],
                              aggregate_score=None)

    monkeypatch.setattr(nodes, "_get_eval_runner", lambda: _Runner())
    state = initial_state("q")
    state["error"] = "boom"
    state["current_phase"] = "error"
    out = asyncio.run(nodes.run_eval_node(state))
    assert out["current_phase"] == "error"
    assert out["eval_scores"]["_aggregate"] is None


def test_persist_run_defaults_terminal_state_to_success(monkeypatch, tmp_path):
    from src.graph import nodes
    from src.persistence.run_store import RunStore

    store = RunStore(db_path=str(tmp_path / "runs.db"))
    monkeypatch.setattr(nodes, "_get_run_store", lambda: store)
    state = initial_state("q")
    state["current_phase"] = "complete"
    out = nodes.persist_run_node(state)
    assert out["terminal_state"] == ts.SUCCESS
    assert store.get_run(state["run_id"])["terminal_state"] == ts.SUCCESS


# ─── Persistence ─────────────────────────────────────────────────────────────

def test_run_store_persists_terminal_classification(tmp_path):
    from src.persistence.run_store import RunStore

    store = RunStore(db_path=str(tmp_path / "runs.db"))
    state = initial_state("q")
    state["terminal_state"] = ts.ENVIRONMENT_ERROR
    state["terminal_detail"] = "data_input"
    state["eval_error"] = "RuntimeError: judge down"
    store.save_run(state)

    row = store.get_run(state["run_id"])
    assert row["terminal_state"] == ts.ENVIRONMENT_ERROR
    assert row["terminal_detail"] == "data_input"
    assert row["eval_error"] == "RuntimeError: judge down"
    assert store.list_runs()[0]["terminal_state"] == ts.ENVIRONMENT_ERROR


def test_run_store_migrates_a_pre_e3_database(tmp_path):
    """logs/runs.db in this environment already exists and is populated --
    CREATE TABLE IF NOT EXISTS is a no-op against it (same argument 附录
    R.3/AO.1 made for cleaning_applied_level)."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, user_query TEXT NOT NULL, current_phase TEXT,
                guardrail_passed INTEGER, error TEXT, error_type TEXT,
                decision_trace_json TEXT NOT NULL DEFAULT '[]',
                mcp_call_log_json TEXT NOT NULL DEFAULT '[]', eval_scores_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.execute("INSERT INTO runs (run_id, user_query, error_type) VALUES ('old1','q','pipeline_error')")

    from src.persistence.run_store import RunStore
    store = RunStore(db_path=str(db))
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert {"terminal_state", "terminal_detail", "eval_error"} <= cols
    # A pre-E3 row keeps its data and is identifiable BY the NULL -- its
    # `pipeline_error` covers what E3 now splits out, and must not be read
    # as "agent reasoning failure".
    assert store.get_run("old1")["terminal_state"] is None
    assert store.get_run("old1")["error_type"] == "pipeline_error"


# ─── Aggregation: reported, never used as a row-level exclusion ──────────────

def test_a_failed_row_is_not_dropped_from_trial_aggregation():
    """Deliberate non-decision: dropping it would take those runs out of
    error_rate's own denominator, so the measured failure rate would improve
    as the system failed more."""
    from src.eval.trials import is_applicable, not_applicable_reason

    row = {"terminal_state": ts.MCP_ERROR, "cleaning_applied_level": "none",
           "rag_arm_invalid_reason": None}
    assert not_applicable_reason(row) is None
    assert is_applicable(row) is True


def test_summarize_case_tallies_terminal_states():
    from src.eval.trials import summarize_case

    def _row(terminal):
        return {"test_case_id": "A", "eval_result": {
            "terminal_state": terminal, "cleaning_applied_level": "none",
            "scores": [{"metric": "error_rate", "score": 1.0, "label": "pass", "valid": True}],
        }}

    out = summarize_case("A", [_row(ts.SUCCESS), _row(ts.SUCCESS), _row(ts.MCP_ERROR)], [1])
    assert out["terminal_states"] == {ts.SUCCESS: 2, ts.MCP_ERROR: 1}
    assert out["n_applicable"] == 3      # failures still count


def test_detect_regressions_skips_a_missing_aggregate():
    from src.eval.metrics import MetricScore
    from src.eval.runner import EvalResult, detect_regressions

    baseline = EvalResult("b", "q", [MetricScore("groundedness", 0.9, "pass")], aggregate_score=0.9)
    current = EvalResult("c", "q", [MetricScore("groundedness", 0.9, "pass")], aggregate_score=None)
    alerts = detect_regressions(baseline, current)
    assert [a.metric for a in alerts] == []


def test_overall_aggregate_excludes_and_counts_non_success_rows():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_eval_script", Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"eval_result": {"aggregate_score": 0.8}},
        {"eval_result": {"aggregate_score": 0.6}},
        {"eval_result": {"aggregate_score": None}},
    ]
    mean, excluded = mod._overall_aggregate(rows)
    assert mean == pytest.approx(0.7)
    assert excluded == 1
