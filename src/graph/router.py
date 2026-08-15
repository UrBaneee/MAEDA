"""
Conditional edge logic for the MAEDA LangGraph state graph.
Each function receives the current state and returns a routing key string.
"""
from src.state.graph_state import MAEDAState

# Maximum re-profile iterations before giving up on cleaning
_MAX_CLEAN_ITERATIONS = 3


def route_after_intent(state: MAEDAState) -> str:
    """
    After parse_intent:
      - "clarify"  → agent needs more info from the user (max 1 time)
      - "proceed"  → intent is clear enough to move forward
    """
    if state.get("clarification_needed") and state.get("clarification_count", 0) < 1:
        return "clarify"
    return "proceed"


def route_after_profiling(state: MAEDAState) -> str:
    """
    After connect_and_profile_data (TB0.5 v1, ECOSYSTEM_INTEGRATION_PLAN.md
    附录 B.3/B.4/B.5):
      - "error"  → no data source provided, or an unrecoverable MCP failure
                   (param/contract/data-input error — 附录 B.3's "工具错误"
                   row; the node already turned this into state["error"],
                   never a fake "empty dataset" that looks like "ready")
      - "clean"  → still has_critical_issues and the cleaning loop hasn't
                   hit a terminal stop condition yet (附录 B.4)
      - "ready"  → data is ready for analysis, OR the loop already reached
                   a terminal stop condition. The node is what tells the
                   two apart: it writes `cleaning_stop_reason` (+
                   `cleaning_caveat` when it isn't a clean "passed") the
                   moment a round is terminal, so a capped/regressed/
                   no-improvement stop still routes to "ready" but is never
                   *silent* about it (M7: "不得静默按 ready 处理") — the
                   caveat is what analysis/report generation surfaces.

    `iterations` here counts *completed clean_dataset calls* (附录 B.5),
    not how many times this node has been entered — fixes the pre-M7
    off-by-one where the counter incremented on every entry, capping the
    loop at 2 actual clean attempts despite `_MAX_CLEAN_ITERATIONS = 3`.
    """
    if state.get("error") or state.get("current_phase") == "error":
        return "error"

    # The node already decided this round is terminal — proceed to
    # analysis regardless of has_critical_issues. Kept as a plain read
    # (routers in this module never mutate state); the node is solely
    # responsible for setting cleaning_stop_reason before returning.
    if state.get("cleaning_stop_reason"):
        return "ready"

    report = state.get("data_quality_report") or {}
    has_critical = report.get("has_critical_issues", False)
    iterations = state.get("iteration_count", 0)

    if has_critical and iterations < _MAX_CLEAN_ITERATIONS:
        return "clean"
    return "ready"


def route_after_guardrails(state: MAEDAState) -> str:
    """
    After run_guardrails:
      - "passed"  → all checks passed; proceed to eval
      - "retry"   → fixable issues found; loop back to execute_analysis
      - "fail"    → unfixable issues; route to handle_error
    """
    checks = state.get("guardrail_checks", [])
    if not checks:
        # No checks run yet — treat as passed (shouldn't happen in practice)
        return "passed"

    # Use the structured verdict from the last guardrail run if present
    last_check = checks[-1] if checks else {}
    verdict = last_check.get("overall_verdict", "approved")

    if verdict == "approved":
        return "passed"
    if verdict == "retry" and state.get("guardrail_retry_count", 0) < 2:
        return "retry"
    return "fail"
