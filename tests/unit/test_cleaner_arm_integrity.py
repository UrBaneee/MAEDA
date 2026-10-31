"""附录 FE.4(1) / FF: cleaner on-arm integrity gate."""
from src.eval.trials import is_applicable, not_applicable_reason
from src.graph.nodes import _cleaner_arm_integrity_problems as problems, _invalidate_cleaner_arm


def test_clean_llm_run_is_valid():
    assert problems({"mode": "mcp"}, {"planner_mode_used": "llm"}, "llm") == []


def test_fallback_is_invalid_even_when_rule_was_requested():
    assert "fallback" in problems({"mode": "fallback"}, {}, "rule")[0]


def test_rule_requested_makes_no_llm_claim():
    assert problems({"mode": "mcp"}, {"planner_mode_used": "rule"}, "rule") == []


def test_degraded_planner_keeps_reason_verbatim():
    out = problems({"mode": "mcp"},
                   {"planner_mode_used": "rule", "planner_fallback_reason": "429"}, "llm")
    assert "429" in out[0]


def test_missing_echo_is_invalid():
    assert "echo" in problems({"mode": "mcp"}, {}, "llm")[0]


def test_later_round_cannot_clear_reason():
    s = {"decision_trace": []}
    s = _invalidate_cleaner_arm(s, ["round 1 degraded"])
    s = _invalidate_cleaner_arm(s, ["round 2 degraded"])
    assert "round 1" in s["cleaning_arm_invalid_reason"]
    assert "round 2" in s["cleaning_arm_invalid_reason"]


def test_trial_is_excluded_with_reason():
    row = {"cleaning_arm_invalid_reason": "planner degraded"}
    assert not is_applicable(row)
    assert not_applicable_reason(row) == "cleaning_arm_invalid: planner degraded"
