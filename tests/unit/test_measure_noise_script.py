"""
Tests for scripts/measure_noise.py's `full`-mode split-aware case selection
— 附录 CC.1/CB.2.5, ECOSYSTEM_INTEGRATION_PLAN.md 阶段 3 收尾执行计划轮次 1
(the measure_noise.py side-item of the three-state switch round).

Before `--split` existed, `measure_noise.py full` (no `--cases`) silently
ran the WHOLE golden suite — including every test-split case — 8 times,
spending 硬约束 1's one-time test-split reveal in what looked like routine
noise-floor maintenance, with nothing warning that it had happened. These
tests lock the fix (`--split` defaults to "dev") the same way as BJ's
`_note` audit: a real, always-run check, not just a comment saying what the
default is supposed to be.

`scripts/` is not an installed package (pyproject.toml only packages
`src`), so this file inserts `scripts/` onto `sys.path` and imports
`measure_noise` by bare name — same approach as
tests/unit/test_run_eval_script.py uses for run_eval.py.

No real pipeline/LLM calls anywhere in this file.

Also covers 冻结前必办 (16) / 附录 DG.3, DH.5: `full --split test`/`--split
all` must REFUSE, not warn, unless a declaration-style opt-in is set in the
process environment. `--split dev` (the default, and the split 清单 (5)'s
σ_hetero measurement runs on) must be completely unaffected.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _make_case(cid: str, split: str):
    from src.eval.runner import GoldenTestCase
    return GoldenTestCase(
        id=cid, query=f"Query for {cid}", query_type="descriptive",
        expected_metrics=[], expected_dimensions=[], ground_truth={}, split=split,
    )


def _suite():
    return [
        _make_case("D01", "dev"), _make_case("D02", "dev"),
        _make_case("T01", "test"), _make_case("T02", "test"),
    ]


# ─── --split default (the actual fix) ──────────────────────────────────────

def test_split_argument_defaults_to_dev():
    """Locks the CLI default the way test_run_eval_script.py's
    test_trials_and_concurrency_default_to_one does: inspect the parser's
    own source rather than duplicating the value, so this fails loudly if
    the default is ever changed back to unfiltered/None without touching
    this test too."""
    import measure_noise
    src = inspect.getsource(measure_noise.main)
    assert '"--split", choices=["dev", "test", "all"], default="dev"' in src


def test_no_cases_no_split_selects_only_dev():
    """The exact scenario CC.1 found dangerous: a bare invocation with
    neither --cases nor --split must NOT touch the test split."""
    import measure_noise
    selected = measure_noise.select_full_mode_cases(_suite(), None, "dev")
    assert {tc.id for tc in selected} == {"D01", "D02"}
    assert all(tc.split == "dev" for tc in selected)


def test_split_test_is_opt_in_only():
    import measure_noise
    selected = measure_noise.select_full_mode_cases(_suite(), None, "test")
    assert {tc.id for tc in selected} == {"T01", "T02"}


def test_split_all_returns_everything_but_is_never_the_default():
    import measure_noise
    selected = measure_noise.select_full_mode_cases(_suite(), None, "all")
    assert {tc.id for tc in selected} == {"D01", "D02", "T01", "T02"}


def test_explicit_cases_overrides_split_entirely():
    """Pre-existing behavior, unchanged: an explicit --cases allowlist is
    already a deliberate, informed choice and is not filtered by --split
    at all -- including when it names test-split ids."""
    import measure_noise
    selected = measure_noise.select_full_mode_cases(_suite(), ["T01"], "dev")
    assert {tc.id for tc in selected} == {"T01"}


# ─── warning banner (附录 CC.1: "明确的、读得懂的警告") ────────────────────────

def test_warns_when_test_split_selected(capsys):
    import measure_noise
    measure_noise.warn_if_test_split_involved([_make_case("T01", "test")])
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "TEST-SPLIT" in out
    assert "T01" in out


def test_no_warning_when_only_dev_selected(capsys):
    import measure_noise
    measure_noise.warn_if_test_split_involved([_make_case("D01", "dev")])
    out = capsys.readouterr().out
    assert out == ""


def test_warns_even_for_explicit_cases_that_happen_to_include_test(capsys):
    """A --cases list is a deliberate choice, but a copy-pasted/typo'd test
    id is still cheap to flag -- the warning fires off the SELECTED set,
    not off whether --split was the thing that chose it."""
    import measure_noise
    selected = measure_noise.select_full_mode_cases(_suite(), ["D01", "T01"], "dev")
    measure_noise.warn_if_test_split_involved(selected)
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "T01" in captured


# ─── 冻结前必办 (16) / 附录 DG.3, DH.5: `full` refuses test/all without opt-in ──

def test_opt_in_reads_only_the_named_env_var_and_the_shared_truthy_set():
    """Pure function, no argv/subprocess involved -- same style as
    tests/paid_calls.py's paid_tests_opted_in(), which this gate is
    deliberately shaped after (declaration-style, process-environment-only,
    same accepted truthy spellings)."""
    import measure_noise
    assert measure_noise.test_split_noise_measurement_opted_in({}) is False
    assert measure_noise.test_split_noise_measurement_opted_in(
        {"MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT": "0"}) is False
    assert measure_noise.test_split_noise_measurement_opted_in(
        {"MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT": "false"}) is False
    for v in ("1", "true", "YES", "On"):
        assert measure_noise.test_split_noise_measurement_opted_in(
            {"MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT": v}) is True
    # A same-shaped but differently-named var (e.g. the (15) one) must not
    # satisfy this gate -- each declaration-style opt-in is per-mechanism.
    assert measure_noise.test_split_noise_measurement_opted_in(
        {"MAEDA_ALLOW_PAID_TESTS": "1"}) is False


class _MustNotBeBuilt:
    """`build_graph` stand-in that fails the test if the (16) gate let a
    non-opted-in `full --split test`/`--split all` run reach the point of
    constructing the pipeline -- same technique, same class name, as
    tests/unit/test_run_eval_script.py uses for (15). `build_graph()` is
    the first thing `run_full_mode()` calls, and every billable call is
    downstream of it, so "never called" is what "before any billable
    action" means in a test."""

    def __call__(self):
        raise AssertionError(
            "build_graph() was called on a non-opted-in `full --split "
            "test`/`all` run -- the 附录 DG.3/DH.5 (16) gate did not fail fast"
        )


def _must_not_load_suite():
    raise AssertionError(
        "load_golden_suite() was reached -- the (16) gate did not fail fast"
    )


class _NoopGraph:
    async def ainvoke(self, state):
        state.setdefault("current_phase", "complete")
        return state


class _NoopEvalRunner:
    async def score(self, state, test_case=None, start_time=None, run_id=None):
        from src.eval.metrics import MetricScore
        from src.eval.runner import EvalResult
        cid = test_case.id if test_case else "unknown"
        return EvalResult(
            run_id=run_id or cid, query=state.get("user_query", ""),
            scores=[MetricScore("answer_relevance", 0.8, "pass")],
            aggregate_score=0.8, test_case_id=cid,
        )


def _run_main(monkeypatch, tmp_path, argv_tail, env=None,
              suite_loader=None, graph_factory=None):
    """Drives measure_noise.main() through its real argv/env parsing path
    (not by calling internal functions directly), the same way
    test_run_eval_script.py's _run_main_capturing does for run_eval.py --
    a gate this cheap to bypass by testing the wrong entry point is not
    proven at all."""
    import measure_noise
    monkeypatch.setattr(measure_noise, "load_golden_suite",
                         suite_loader or (lambda: []))
    monkeypatch.setattr(measure_noise, "build_graph",
                         graph_factory or (lambda: _NoopGraph()))
    monkeypatch.setattr(measure_noise, "EvalRunner", lambda: _NoopEvalRunner())
    monkeypatch.setattr(measure_noise, "REPORT_DIR", tmp_path)
    monkeypatch.delenv("MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["measure_noise.py", *argv_tail])
    try:
        asyncio.run(measure_noise.main())
    except SystemExit as exc:
        return exc.code
    return 0


def test_full_split_test_is_refused_before_anything_is_built_or_loaded(tmp_path, monkeypatch, capsys):
    """The claim under test is not "the message is worded right" -- it is
    that the command stops before the process can spend money on the
    holdout. Both the suite loader and the graph builder are booby-trapped:
    reaching either fails the test."""
    code = _run_main(
        monkeypatch, tmp_path, ["full", "--split", "test"],
        suite_loader=_must_not_load_suite, graph_factory=_MustNotBeBuilt(),
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT" in err  # names the variable
    assert "DG.3" in err and "DH.5" in err                     # names the appendices
    assert "冻结前必办 (16)" in err                              # names the checklist item
    assert "docs/phase4_preregistration.md" in err             # names the pre-registration
    assert "Nothing was run and nothing was billed" in err
    assert list(tmp_path.iterdir()) == [], "a refused run must write no report"


def test_full_split_all_is_also_refused(tmp_path, monkeypatch):
    """DG.3/DH.5 name both `test` and `all` -- `all` reaches the same
    holdout cases by a different flag and must be refused identically."""
    code = _run_main(
        monkeypatch, tmp_path, ["full", "--split", "all"],
        suite_loader=_must_not_load_suite, graph_factory=_MustNotBeBuilt(),
    )
    assert code == 2


def test_full_split_dev_is_not_gated_even_without_opt_in(tmp_path, monkeypatch):
    """(16) must not touch `--split dev`, the default -- 清单 (5)'s
    σ_hetero measurement runs `measure_noise.py full` on dev, and it must
    keep working exactly as before regardless of whether the (16)
    opt-in env var is set. Proven by actually letting the run reach the
    (fake) pipeline and write a report, with no opt-in present."""
    code = _run_main(monkeypatch, tmp_path, ["full", "--split", "dev"])
    assert code == 0
    reports = list(tmp_path.glob("full_noise_*.json"))
    assert len(reports) == 1


def test_full_split_test_proceeds_when_explicitly_opted_in(tmp_path, monkeypatch):
    """The escape hatch is declaration-style, not a permanent switch: with
    the env var set for this invocation only, `--split test` must reach
    and complete the (fake) pipeline rather than being refused."""
    code = _run_main(
        monkeypatch, tmp_path, ["full", "--split", "test"],
        env={"MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT": "1"},
        suite_loader=lambda: [_make_case("T01", "test")],
    )
    assert code == 0
    reports = list(tmp_path.glob("full_noise_*.json"))
    assert len(reports) == 1


def test_judge_mode_is_unaffected_by_the_split_gate(tmp_path, monkeypatch):
    """`judge` mode has no --split argument at all; the gate's condition
    (`args.mode == "full" and ...`) must short-circuit before ever touching
    `args.split`, or this would crash with AttributeError instead of
    running judge mode normally."""
    import measure_noise

    async def _fake_judge_mode(cases, repeats, concurrency):
        return {"per_case": {}, "pooled": {"raw": {}, "median": {}}, "cases": cases, "repeats": repeats}

    monkeypatch.setattr(measure_noise, "run_judge_mode", _fake_judge_mode)
    monkeypatch.setattr(measure_noise, "REPORT_DIR", tmp_path)
    monkeypatch.delenv("MAEDA_ALLOW_TEST_SPLIT_NOISE_MEASUREMENT", raising=False)
    monkeypatch.setattr(sys, "argv", ["measure_noise.py", "judge", "--cases", "D01", "--repeats", "1"])
    try:
        asyncio.run(measure_noise.main())
    except SystemExit as exc:
        raise AssertionError(f"judge mode unexpectedly exited: {exc.code}")
    reports = list(tmp_path.glob("judge_noise_*.json"))
    assert len(reports) == 1
