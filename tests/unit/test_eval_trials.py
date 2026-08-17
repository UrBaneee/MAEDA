"""
Tests for src/eval/trials.py — D0's pass@k/pass^k/variance layer (阶段 3 /
附录 AW block 3). Pure math + data wrangling, no graph/LLM/judge calls
anywhere in this file.
"""
from __future__ import annotations

import math

from src.eval.trials import (
    is_applicable,
    is_binary_metric,
    pass_at_k,
    pass_hat_k,
    summarize_case,
    summarize_report,
)


# ─── pass_at_k ──────────────────────────────────────────────────────────────────

def test_pass_at_k_all_successes_is_one():
    assert pass_at_k(n=5, c=5, k=1) == 1.0
    assert pass_at_k(n=5, c=5, k=5) == 1.0


def test_pass_at_k_all_failures_is_zero():
    assert pass_at_k(n=5, c=0, k=1) == 0.0
    assert pass_at_k(n=5, c=0, k=3) == 0.0


def test_pass_at_k_k1_equals_c_over_n():
    """pass@1 has a simple closed form: probability a single random draw
    from n trials (c successes) is a success = c/n. Every other k value's
    formula collapses to this at k=1."""
    for n, c in [(5, 3), (10, 1), (8, 7), (2, 1)]:
        assert math.isclose(pass_at_k(n, c, 1), c / n)


def test_pass_at_k_fewer_failures_than_k_is_guaranteed_success():
    """n-c < k: not enough failures to fill a k-subset with zero
    successes, so any k-subset is guaranteed to contain >= 1 success."""
    assert pass_at_k(n=5, c=4, k=2) == 1.0  # only 1 failure, can't fill a 2-subset with it
    assert pass_at_k(n=2, c=1, k=2) == 1.0


def test_pass_at_k_known_value():
    """n=4, c=2, k=2: C(2,2)/C(4,2) = 1/6 chance BOTH chosen are failures
    -> pass@2 = 1 - 1/6 = 5/6."""
    assert math.isclose(pass_at_k(n=4, c=2, k=2), 5 / 6)


def test_pass_at_k_not_enough_trials_is_none_not_zero():
    """n < k must be None (unanswerable), never 0.0 -- 0.0 would be
    indistinguishable from a real "always fails" measurement."""
    assert pass_at_k(n=2, c=0, k=3) is None
    assert pass_at_k(n=0, c=0, k=1) is None


# ─── pass_hat_k ─────────────────────────────────────────────────────────────────

def test_pass_hat_k_all_successes_is_one():
    assert pass_hat_k(n=5, c=5, k=5) == 1.0
    assert pass_hat_k(n=5, c=5, k=1) == 1.0


def test_pass_hat_k_k1_equals_c_over_n():
    for n, c in [(5, 3), (10, 1), (8, 7)]:
        assert math.isclose(pass_hat_k(n, c, 1), c / n)


def test_pass_hat_k_fewer_successes_than_k_is_zero_not_none():
    """c < k legitimately means zero probability all k succeed -- a real
    answer, unlike n < k which is unanswerable."""
    assert pass_hat_k(n=5, c=2, k=3) == 0.0


def test_pass_hat_k_known_value():
    """n=4, c=2, k=2: C(2,2)/C(4,2) = 1/6."""
    assert math.isclose(pass_hat_k(n=4, c=2, k=2), 1 / 6)


def test_pass_hat_k_not_enough_trials_is_none():
    assert pass_hat_k(n=2, c=2, k=3) is None


def test_pass_at_k_and_pass_hat_k_agree_when_k_equals_n():
    """When k == n, "at least one of all n trials succeeds" (pass@n) and
    "all n trials succeed" (pass^n) are DIFFERENT questions unless c == n
    or c == 0 -- sanity check they're not accidentally the same formula."""
    assert pass_at_k(n=5, c=3, k=5) == 1.0     # >=1 success among all 5: yes, c=3>0
    assert pass_hat_k(n=5, c=3, k=5) == 0.0    # ALL 5 succeed: no, only 3 did


# ─── is_binary_metric / is_applicable ──────────────────────────────────────────

def test_safe_refusal_is_not_binary():
    assert is_binary_metric("safe_refusal") is False


def test_other_metrics_are_binary():
    for m in ["answer_relevance", "groundedness", "factual_accuracy",
              "tool_selection", "error_rate", "token_cost"]:
        assert is_binary_metric(m) is True


def test_is_applicable_full_and_none_are_applicable():
    assert is_applicable({"cleaning_applied_level": "full"}) is True
    assert is_applicable({"cleaning_applied_level": "none"}) is True
    assert is_applicable({"cleaning_applied_level": None}) is True
    assert is_applicable({}) is True  # missing key -> None -> applicable


def test_is_applicable_blocked_needs_review_is_not_applicable():
    assert is_applicable({"cleaning_applied_level": "blocked_needs_review"}) is False


# ─── summarize_case ─────────────────────────────────────────────────────────────

def _row(cid: str, score: float, label: str, cleaning_applied_level=None, valid=True):
    return {
        "test_case_id": cid,
        "eval_result": {
            "cleaning_applied_level": cleaning_applied_level,
            "scores": [
                {"metric": "answer_relevance", "score": score, "label": label, "valid": valid},
                {"metric": "safe_refusal", "score": 0.0, "label": "info", "valid": True},
            ],
        },
        "meta": {},
    }


def test_summarize_case_binary_counts_and_pass_at_1():
    rows = [
        _row("A", 0.9, "pass"), _row("A", 0.9, "pass"),
        _row("A", 0.2, "fail"), _row("A", 0.9, "pass"),
    ]
    result = summarize_case("A", rows, k_values=[1])
    b = result["binary"]["answer_relevance"]
    assert b["n_scored"] == 4
    assert b["c"] == 3
    assert math.isclose(b["pass_at_k"][1], 0.75)
    assert math.isclose(b["pass_hat_k"][1], 0.75)  # k=1 identity holds here too


def test_summarize_case_excludes_safe_refusal_from_binary():
    rows = [_row("A", 0.9, "pass")]
    result = summarize_case("A", rows, k_values=[1])
    assert "safe_refusal" not in result["binary"]
    assert "safe_refusal" in result["continuous"]  # still reported, just not pass@k'd


def test_summarize_case_excludes_blocked_needs_review_trials():
    rows = [
        _row("A", 0.9, "pass"),
        _row("A", 0.9, "pass"),
        _row("A", 0.1, "fail", cleaning_applied_level="blocked_needs_review"),
    ]
    result = summarize_case("A", rows, k_values=[1])
    assert result["n_trials"] == 3
    assert result["n_applicable"] == 2
    assert result["n_not_applicable"] == 1
    b = result["binary"]["answer_relevance"]
    # The excluded trial was a "fail" -- if it had been counted, c/n would
    # be 2/3, not 2/2. Confirms exclusion actually changes the number, not
    # just the bookkeeping fields.
    assert b["n_scored"] == 2
    assert b["c"] == 2
    assert b["pass_at_k"][1] == 1.0


def test_summarize_case_none_cleaning_level_is_not_excluded():
    """Regression test for is_applicable's deliberately narrower-than-
    "!= full" reading: a "none" trial (data never needed cleaning) must
    count fully, not be silently dropped."""
    rows = [
        _row("A", 0.9, "pass", cleaning_applied_level="none"),
        _row("A", 0.9, "pass", cleaning_applied_level="full"),
    ]
    result = summarize_case("A", rows, k_values=[1])
    assert result["n_applicable"] == 2
    assert result["n_not_applicable"] == 0


def test_summarize_case_invalid_score_excluded_from_n_and_c():
    rows = [
        _row("A", 0.9, "pass"),
        _row("A", 0.5, "error", valid=False),
    ]
    result = summarize_case("A", rows, k_values=[1])
    b = result["binary"]["answer_relevance"]
    assert b["n_scored"] == 1  # the invalid one doesn't count toward n
    assert b["n_excluded_invalid_score"] == 1
    assert b["c"] == 1


def test_summarize_case_continuous_summary_matches_noise_summarize():
    from src.eval.noise import summarize as noise_summarize
    rows = [_row("A", 0.9, "pass"), _row("A", 0.7, "pass"), _row("A", 0.8, "pass")]
    result = summarize_case("A", rows, k_values=[1])
    expected = noise_summarize("A/answer_relevance", [0.9, 0.7, 0.8])
    assert result["continuous"]["answer_relevance"]["summary"]["mean"] == round(expected.mean, 4)
    assert result["continuous"]["answer_relevance"]["summary"]["std"] == round(expected.std, 4)


def test_summarize_case_single_trial_has_no_continuous_summary():
    """src/eval/noise.py's summarize() itself returns None for n<2 --
    confirm that propagates through, not silently coerced to some
    placeholder."""
    rows = [_row("A", 0.9, "pass")]
    result = summarize_case("A", rows, k_values=[1])
    assert result["continuous"]["answer_relevance"]["summary"] is None


# ─── summarize_report ───────────────────────────────────────────────────────────

def test_summarize_report_groups_by_case_across_trials():
    per_trial = [
        [_row("A", 0.9, "pass"), _row("B", 0.9, "pass")],
        [_row("A", 0.9, "pass"), _row("B", 0.1, "fail")],
    ]
    summary = summarize_report(per_trial, k_values=[1])
    assert summary["n_cases"] == 2
    assert set(summary["per_case"]) == {"A", "B"}
    assert summary["per_case"]["A"]["n_trials"] == 2
    assert summary["per_case"]["B"]["binary"]["answer_relevance"]["c"] == 1


def test_summarize_report_suite_binary_is_mean_of_case_pass_at_k():
    per_trial = [
        [_row("A", 0.9, "pass"), _row("B", 0.9, "pass")],
        [_row("A", 0.9, "pass"), _row("B", 0.1, "fail")],
    ]
    summary = summarize_report(per_trial, k_values=[1])
    # A: pass@1 = 2/2 = 1.0 ; B: pass@1 = 1/2 = 0.5 ; mean = 0.75
    suite = summary["suite_binary"]["answer_relevance"][1]
    assert math.isclose(suite["mean_pass_at_k"], 0.75)
    assert suite["n_cases_defined"] == 2
    assert suite["n_cases_total"] == 2


def test_summarize_report_default_k_is_one():
    per_trial = [[_row("A", 0.9, "pass")]]
    summary = summarize_report(per_trial)
    assert summary["k_values"] == [1]


def test_summarize_report_empty_input():
    summary = summarize_report([])
    assert summary["n_cases"] == 0
    assert summary["per_case"] == {}
    assert summary["suite_binary"] == {}
    assert summary["suite_continuous"] == {}
