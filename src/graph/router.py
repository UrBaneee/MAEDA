"""
Conditional edge logic for the MAEDA LangGraph state graph.
Each function receives the current state and returns a routing key string.
"""
from src.config.settings import settings
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


def route_after_schema(state: MAEDAState) -> str:
    """
    After connect_schema (E2, ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ):
      - "profile" → straight to profile_and_clean, no refine this round
      - "refine"  → through refine_intent first

    Never refines (falls straight to "profile") when:
      - connect_schema's "no data source" exit already set current_phase
        to "error" — profile_and_clean's own top-of-function guard is what
        actually stops the round; this check just avoids wasting an LLM
        call on a state that's headed to handle_error regardless.
      - schema extraction itself failed this round (`schema_columns` is
        None) — there is no real schema to inject, so refine_intent would
        just re-run `_parse()` against the same "Schema unavailable: ..."
        placeholder text connect_schema already put in schema_summary, no
        better informed than the first parse.
      - `intent_refine_done` is already True — defensive gate against
        re-triggering refine within one round. Structurally unreachable
        today (the clean self-loop targets profile_and_clean directly,
        bypassing this edge entirely — src/graph/builder.py), kept so
        that stays true even if the topology changes later, rather than
        relying on the self-loop's target alone.

    Otherwise defers to settings.intent_refine_trigger:
      - "always" refines unconditionally.
      - "if_unresolved" (the default) only refines when a deterministic
        pre-check finds at least one intent mention that doesn't match
        any real column — reusing `_resolve_intent_columns`, the exact
        function profile_and_clean runs for real afterward, purely to
        decide this routing key. Its result here is discarded, never
        written to state, so it can never drift from what
        profile_and_clean computes for the actual intent payload — that
        happens again, independently, once profile_and_clean runs
        (against whatever `parsed_intent` refine leaves behind).
      - any other/misconfigured value fails safe to the "if_unresolved"
        pre-check rather than silently always-refining.
    """
    if state.get("current_phase") == "error":
        return "profile"
    if state.get("schema_columns") is None:
        return "profile"
    if state.get("intent_refine_done"):
        return "profile"

    if settings.intent_refine_trigger == "always":
        return "refine"

    from src.graph.nodes import _resolve_intent_columns  # local: avoid a module-level nodes<->router cycle
    _, unresolved, _ = _resolve_intent_columns(
        state.get("parsed_intent") or {}, state.get("schema_columns") or [],
    )
    return "refine" if unresolved else "profile"


def route_after_profiling(state: MAEDAState) -> str:
    """
    After profile_and_clean (TB0.5 v1, ECOSYSTEM_INTEGRATION_PLAN.md
    附录 B.3/B.4/B.5; the node was "connect_and_profile_data" before the
    E2 BO.1 split, 附录 BQ):
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
