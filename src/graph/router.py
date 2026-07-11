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
    After connect_and_profile_data:
      - "error"  → no data source provided or unrecoverable connection error
      - "clean"  → Data Cleaner MCP found critical quality issues; re-clean + re-profile
      - "ready"  → data is ready for analysis
    Caps at _MAX_CLEAN_ITERATIONS to prevent infinite loops.
    """
    if state.get("error") or state.get("current_phase") == "error":
        return "error"

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
