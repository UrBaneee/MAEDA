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
"""
from __future__ import annotations

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
