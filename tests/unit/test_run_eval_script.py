"""
Tests for scripts/run_eval.py's D0 multi-trial additions (阶段 3 / 附录 AW
block 2): `run_trials()`, the `--trials`/`--concurrency` CLI, and the two
report shapes (single-trial unchanged vs. multi-trial `per_trial`).

`scripts/` is not an installed package (pyproject.toml only packages
`src`), so this file inserts `scripts/` onto `sys.path` and imports
`run_eval` by bare name -- the same approach scripts/measure_noise.py
already uses to import from run_eval.py itself.

No real graph/LLM calls anywhere in this file: `_FakeGraph`/
`_FakeEvalRunner` below stand in for `build_graph()`/`EvalRunner()`.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _make_case(cid: str, split: str = "dev"):
    from src.eval.runner import GoldenTestCase
    return GoldenTestCase(
        id=cid, query=f"Query for {cid}", query_type="descriptive",
        expected_metrics=[], expected_dimensions=[], ground_truth={}, split=split,
    )


class _FakeGraph:
    """Stands in for build_graph()'s compiled graph. ainvoke() just returns
    the state it was given (already carrying a real unique run_id from
    initial_state()), optionally tracking concurrent in-flight calls."""

    def __init__(self, delay: float = 0.0, track_concurrency: bool = False):
        self._delay = delay
        self._track = track_concurrency
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def ainvoke(self, state):
        if self._track:
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._track:
            async with self._lock:
                self.current -= 1
        state.setdefault("current_phase", "complete")
        return state


class _FakeEvalRunner:
    """Stands in for EvalRunner(). score() builds a real EvalResult (so
    .to_dict()/.aggregate_score behave exactly like production) without
    invoking any judge/LLM logic."""

    def __init__(self, aggregate_by_case: dict | None = None,
                 cleaning_by_case: dict | None = None):
        self._aggregate_by_case = aggregate_by_case or {}
        self._cleaning_by_case = cleaning_by_case or {}

    async def score(self, state, test_case=None, start_time=None, run_id=None):
        from src.eval.metrics import MetricScore
        from src.eval.runner import EvalResult
        cid = test_case.id if test_case else "unknown"
        agg = self._aggregate_by_case.get(cid, 0.8)
        applied_level, stop_reason = self._cleaning_by_case.get(cid, (None, None))
        return EvalResult(
            run_id=run_id or cid,
            query=state.get("user_query", ""),
            scores=[MetricScore("answer_relevance", agg, "pass" if agg >= 0.6 else "fail")],
            aggregate_score=agg,
            test_case_id=cid,
            cleaning_applied_level=applied_level,
            cleaning_stop_reason=stop_reason,
        )


# ─── run_trials() ──────────────────────────────────────────────────────────────

def test_run_trials_keeps_each_trials_rows_separate_not_merged():
    import run_eval
    cases = [_make_case("A"), _make_case("B")]
    per_trial = asyncio.run(run_trials_helper(run_eval, cases, trials=3, concurrency=2))
    assert len(per_trial) == 3
    for rows in per_trial:
        assert {r["test_case_id"] for r in rows} == {"A", "B"}
        assert len(rows) == 2


def test_run_trials_respects_concurrency_limit():
    """The whole point of reusing measure_noise.py's asyncio.Semaphore
    pattern -- verify it actually caps in-flight (trial, case) runs, not
    just that the code runs without error."""
    import run_eval
    cases = [_make_case(cid) for cid in ("A", "B", "C")]
    graph = _FakeGraph(delay=0.02, track_concurrency=True)
    per_trial = asyncio.run(
        run_eval.run_trials(cases, graph, _FakeEvalRunner(), trials=3, concurrency=2)
    )
    assert graph.peak <= 2, "concurrency=2 must never allow more than 2 in-flight calls"
    assert graph.peak >= 2, "9 tasks with concurrency=2 and a real await should reach the cap"
    assert sum(len(rows) for rows in per_trial) == 9  # 3 trials x 3 cases


def test_run_trials_concurrency_one_is_fully_serial():
    import run_eval
    cases = [_make_case(cid) for cid in ("A", "B")]
    graph = _FakeGraph(delay=0.01, track_concurrency=True)
    asyncio.run(run_eval.run_trials(cases, graph, _FakeEvalRunner(), trials=2, concurrency=1))
    assert graph.peak == 1


def test_run_trials_carries_cleaning_fields_through_to_report_rows():
    """D0's whole reason for needing 附录 R.3's block-0 persistence: each
    trial's row must retain cleaning_applied_level/cleaning_stop_reason
    so block 3 can compute pass@k denominators without a RunStore lookup."""
    import run_eval
    cases = [_make_case("A")]
    eval_runner = _FakeEvalRunner(cleaning_by_case={"A": ("blocked_needs_review", "needs_review")})
    per_trial = asyncio.run(
        run_eval.run_trials(cases, _FakeGraph(), eval_runner, trials=1, concurrency=1)
    )
    row = per_trial[0][0]
    assert row["eval_result"]["cleaning_applied_level"] == "blocked_needs_review"
    assert row["eval_result"]["cleaning_stop_reason"] == "needs_review"


async def run_trials_helper(run_eval_module, cases, trials, concurrency):
    return await run_eval_module.run_trials(
        cases, _FakeGraph(), _FakeEvalRunner(), trials=trials, concurrency=concurrency,
    )


# ─── run_one_case() surfaces the real graph run_id in meta ────────────────────

def test_run_one_case_meta_carries_the_real_graph_run_id():
    """eval_runner.score() is called with run_id=tc.id (the golden case
    id -- identical across every trial of the same case), which is NOT
    RunStore's actual per-invocation primary key. meta['run_id'] is the
    only place in the report that lets a specific trial's row be
    cross-referenced back to its RunStore-persisted decision_trace."""
    import run_eval
    from src.state.graph_state import initial_state

    tc = _make_case("A")
    graph = _FakeGraph()
    eval_result, meta = asyncio.run(run_eval.run_one_case(tc, graph, _FakeEvalRunner()))
    assert meta["run_id"] is not None
    assert meta["run_id"] != tc.id  # the real uuid4, not the case id
    assert eval_result.run_id == tc.id  # unchanged pre-existing behaviour


# ─── main(): CLI defaults + report shape ───────────────────────────────────────

def _run_main_with_args(monkeypatch, tmp_path, argv_tail, suite, eval_runner=None, graph=None):
    import run_eval

    monkeypatch.setattr(run_eval, "load_golden_suite", lambda: suite)
    monkeypatch.setattr(run_eval, "build_graph", lambda: graph or _FakeGraph())
    monkeypatch.setattr(run_eval, "EvalRunner", lambda: eval_runner or _FakeEvalRunner())
    monkeypatch.setattr(run_eval, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv_tail])
    try:
        asyncio.run(run_eval.main())
    except SystemExit as exc:
        return exc.code
    return 0


def test_trials_and_concurrency_default_to_one():
    import run_eval
    import argparse
    # Inspect the parser's own defaults rather than duplicating them here,
    # so this test fails if the defaults ever drift instead of silently
    # asserting a copy of them.
    import inspect
    src = inspect.getsource(run_eval.main)
    assert '"--trials", type=int, default=1' in src
    assert '"--concurrency", type=int, default=1' in src


def test_single_trial_report_shape_is_unchanged(monkeypatch, tmp_path):
    """--trials 1 (the default) must produce byte-for-byte the same
    top-level report keys as every report this script produced before D0
    existed -- no "trials"/"per_trial" keys leaking into the default
    path's output."""
    suite = [_make_case("A", split="dev"), _make_case("B", split="dev")]
    code = _run_main_with_args(monkeypatch, tmp_path, ["--split", "dev"], suite)
    assert code == 0

    reports = list(tmp_path.glob("eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    # 定案 #15 (阶段 3 收尾执行计划轮次 1, 附录 CD): "arm" is the one
    # intentional addition to this otherwise-frozen shape -- run-level
    # three-state-switch config, not a new per-case dimension, so it earns
    # updating this lock rather than a --suite-flag-style rejection.
    #
    # E3 (执行顺序表轮次 5, 附录 CU) adds the second:
    # "n_cases_without_aggregate". It is here for the same reason "arm" was
    # -- it qualifies an existing top-level number rather than adding a new
    # dimension. `overall_aggregate` used to be a mean over every case; E3
    # makes it a mean over the cases that terminated in success, and without
    # this key the change is invisible in the saved JSON (a shrinking
    # denominator with nothing recording that it shrank is the failure shape
    # 附录 CK.3 exists to catch). Additive: every existing consumer reads by
    # key and is unaffected.
    assert set(report.keys()) == {
        "timestamp", "n_cases", "overall_aggregate", "n_cases_without_aggregate",
        "split", "cases", "arm",
    }
    assert report["n_cases_without_aggregate"] == 0
    assert report["arm"] == {"cleaner_mode": "auto", "rag_mode": "auto", "mcp_strict_mode": "degraded"}
    assert report["n_cases"] == 2
    assert len(report["cases"]) == 2


def test_multi_trial_report_shape_has_per_trial(monkeypatch, tmp_path):
    suite = [_make_case("A", split="dev")]
    code = _run_main_with_args(
        monkeypatch, tmp_path, ["--split", "dev", "--trials", "3", "--concurrency", "2"], suite,
    )
    assert code == 0

    reports = list(tmp_path.glob("eval_trials3_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["trials"] == 3
    assert report["concurrency"] == 2
    assert len(report["per_trial"]) == 3
    for trial in report["per_trial"]:
        assert trial["n_cases"] == 1
        assert "overall_aggregate" in trial
    # No top-level "cases"/"overall_aggregate" keys -- those are the
    # single-trial shape and must not silently also appear here.
    assert "cases" not in report
    assert "overall_aggregate" not in report
    assert report["arm"] == {"cleaner_mode": "auto", "rag_mode": "auto", "mcp_strict_mode": "degraded"}


def test_zero_trials_is_rejected(monkeypatch, tmp_path):
    suite = [_make_case("A")]
    code = _run_main_with_args(monkeypatch, tmp_path, ["--trials", "0"], suite)
    assert code == 2


def test_zero_concurrency_is_rejected(monkeypatch, tmp_path):
    suite = [_make_case("A")]
    code = _run_main_with_args(monkeypatch, tmp_path, ["--concurrency", "0"], suite)
    assert code == 2


def test_pipeline_error_still_exits_nonzero_in_multi_trial_mode():
    """The pre-existing crash-detection behaviour (a real pipeline_error,
    not a safe_refusal, must exit non-zero so CI can't go green on a
    crashed suite) must still hold once trials > 1 -- checked across every
    trial, not just the first."""
    import run_eval
    from src.eval.runner import EvalResult
    from src.eval.metrics import MetricScore

    class _CrashingEvalRunner:
        async def score(self, state, test_case=None, start_time=None, run_id=None):
            return EvalResult(
                run_id=run_id or test_case.id, query="q",
                scores=[MetricScore("error_rate", 0.0, "fail")],
                aggregate_score=0.0, test_case_id=test_case.id,
            )

    class _CrashingGraph:
        async def ainvoke(self, state):
            state["error"] = "boom"
            state["error_type"] = "pipeline_error"
            return state

    suite = [_make_case("A")]
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        import unittest.mock as mock
        with mock.patch.object(run_eval, "load_golden_suite", lambda: suite), \
             mock.patch.object(run_eval, "build_graph", lambda: _CrashingGraph()), \
             mock.patch.object(run_eval, "EvalRunner", lambda: _CrashingEvalRunner()), \
             mock.patch.object(run_eval, "REPORT_DIR", Path(tmpdir)), \
             mock.patch.object(sys, "argv", ["run_eval.py", "--trials", "2"]):
            with pytest.raises(SystemExit) as exc_info:
                asyncio.run(run_eval.main())
            assert exc_info.value.code == 1


# ─── 定案 #15 / 阶段 3 收尾执行计划轮次 1 / 附录 CB.3.4: arm reporting ─────────

def test_run_one_case_meta_carries_the_arm(monkeypatch):
    """meta["cleaner_mode"]/meta["rag_mode"] mirror state["cleaner_mode"]/
    state["rag_mode"] (src/state/graph_state.py's initial_state() snapshot
    of settings.maeda_cleaner_mode/maeda_rag_mode) -- the per-row half of
    CB.3.4's double-write, independent of report["arm"]."""
    import run_eval
    from src.config.settings import settings as _settings
    from src.state.graph_state import initial_state

    monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_on")
    monkeypatch.setattr(_settings, "maeda_rag_mode", "force_off")

    tc = _make_case("A")
    graph = _FakeGraph()
    eval_result, meta = asyncio.run(run_eval.run_one_case(tc, graph, _FakeEvalRunner()))
    assert meta["cleaner_mode"] == "force_on"
    assert meta["rag_mode"] == "force_off"
    # sanity: initial_state() really is what produced these, not a stray default
    assert initial_state("q")["cleaner_mode"] == "force_on"


def test_run_one_case_meta_carries_the_rag_arm_invalidation(monkeypatch):
    """附录 CK.3: a force_on trial whose retrieval degraded is dropped from
    aggregation by src/eval/trials.py reading `eval_result`; this mirrors
    the reason into `meta` too, so a reader scanning the report can see
    WHY a row vanished from the denominator instead of only that it did."""
    import run_eval

    class _InvalidatingGraph(_FakeGraph):
        async def ainvoke(self, state):
            state = await super().ainvoke(state)
            state["rag_arm_invalid_reason"] = "retrieval_mode='bm25_only'"
            return state

    tc = _make_case("A")
    _, meta = asyncio.run(run_eval.run_one_case(tc, _InvalidatingGraph(), _FakeEvalRunner()))
    assert meta["rag_arm_invalid_reason"] == "retrieval_mode='bm25_only'"


def test_current_arm_reflects_settings(monkeypatch):
    import run_eval
    from src.config.settings import settings as _settings

    monkeypatch.setattr(_settings, "maeda_cleaner_mode", "force_off")
    monkeypatch.setattr(_settings, "maeda_rag_mode", "force_on")
    monkeypatch.setattr(_settings, "mcp_strict_mode", "strict")
    assert run_eval._current_arm() == {
        "cleaner_mode": "force_off", "rag_mode": "force_on", "mcp_strict_mode": "strict",
    }
