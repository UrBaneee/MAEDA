"""
Conditional edge logic for the MAEDA LangGraph state graph.
Each function receives the current state and returns a routing key string.
"""
from src.config.settings import settings
from src.state.graph_state import MAEDAState

# Maximum re-profile iterations before giving up on cleaning
_MAX_CLEAN_ITERATIONS = 3


def cleaner_should_attempt_clean(has_critical_issues: bool) -> bool:
    """
    定案 #15 / 附录 CC.2/CC.3: the ONE place that resolves whether a round
    should attempt clean_dataset, reused both by profile_and_clean
    (src/graph/nodes.py -- decides whether THIS round's actual call
    happens, and is also where a "force_off" terminal cleaning_stop_reason
    + `mode="skipped"` log entry get recorded, since that's node-level
    side-effecting work this pure function must not do) and by
    route_after_profiling below (decides whether to loop back for another
    round when the loop hasn't already hit a B.4 terminal stop condition).
    Kept as a single function specifically so force_on's "bypass
    has_critical_issues itself, not just whatever was blocking it" (CC.2)
    can't drift out of sync between the two call sites -- two
    independently-edited copies of the same gate is exactly the kind of
    thing that silently regresses to "force_on == auto" on data where
    has_critical_issues never fires (CC.2's real-world finding).

      "auto"       -- unchanged: `has_critical_issues` as computed by
                       profile_dataset/re-profile, same as before 定案 #15.
      "force_on"   -- always True, regardless of `has_critical_issues`.
      "force_off"  -- always False, regardless of `has_critical_issues`.
    """
    mode = settings.maeda_cleaner_mode
    if mode == "force_off":
        return False
    if mode == "force_on":
        return True
    return has_critical_issues


# 阶段 3 收尾执行计划轮次 3 / 附录 CC.7 裁定 4. Reason strings for the RAG
# retrieval routing decision, kept as constants so the decision trace, the
# `mode="skipped"` mcp_call_log record and the tests all quote the same
# text instead of three drifting copies.
RAG_SKIP_FORCE_OFF = "MAEDA_RAG_MODE=force_off"
RAG_RETRIEVE_FORCE_ON = "MAEDA_RAG_MODE=force_on"
RAG_RETRIEVE_AUTO_DEGENERATE = (
    "MAEDA_RAG_MODE=auto; 附录 CC.7 裁定 4: auto's own judgement "
    "(skip retrieval for purely-computational queries) is NOT implemented — "
    "auto deliberately degenerates to always-retrieve"
)


def rag_retrieval_decision(state: MAEDAState) -> tuple[bool, str]:
    """
    定案 #15 / 附录 CC.7 裁定 4 / 阶段 3 收尾执行计划轮次 3: the ONE place
    that decides whether a run retrieves domain knowledge, and why.
    Returns (should_retrieve, reason).

    Same single-shared-gate construction as cleaner_should_attempt_clean
    above, for the same reason (CC.2): this criterion is consumed in three
    places — the conditional edge out of `generate_viz`
    (route_after_viz → src/graph/builder.py), the skip node that records
    the `mode="skipped"` call log entry, and retrieve_knowledge_node's own
    defensive backstop for direct invocation. Three independently-edited
    copies of one criterion is exactly how "force_off" quietly becomes
    "force_off everywhere except the one path nobody re-read".

    The `reason` half is not decoration: 阶段 3's 轮次 3 requires the
    **routing decision itself** to land in the decision trace, not just
    its consequence. A trace that shows "no retrieval happened" cannot
    distinguish force_off from an auto-judged skip from a crash; a trace
    that shows *which rule fired* can.

      "force_off" — never retrieve. The call is not made at all, not made
                    and discarded (附录 CC.3 裁定 1).
      "force_on"  — always retrieve.
      "auto"      — **also always retrieves today.** 附录 CC.7 裁定 4
                    explicitly froze this degeneration: the conditional
                    edge is real (that is what 轮次 3 delivers, and what
                    makes the skip path testable at all), but the
                    query-type judgement that would let `auto` return
                    False is a separate piece of work and is NOT done.
                    This is a known, adjudicated, temporary degeneration
                    — not an oversight, and not "auto routing exists".
                    When it is implemented, this branch is the only place
                    that changes; force_on/force_off keep their meaning.
    """
    mode = settings.maeda_rag_mode
    if mode == "force_off":
        return False, RAG_SKIP_FORCE_OFF
    if mode == "force_on":
        return True, RAG_RETRIEVE_FORCE_ON
    return True, RAG_RETRIEVE_AUTO_DEGENERATE


def route_after_viz(state: MAEDAState) -> str:
    """
    After generate_viz (阶段 3 收尾执行计划轮次 3):
      - "retrieve" → retrieve_domain_knowledge, the real MCP call
      - "skip"     → skip_retrieval, which records the deliberate
                     `mode="skipped"` call-log entry + empty rag context
                     and falls through to generate_insights

    Replaces the unconditional `add_edge("generate_viz",
    "retrieve_domain_knowledge")` this graph had until now — the edge 附录
    CC.7 and docs/handoff_maeda_to_subsystems.md both flagged as the
    reason 阶段 4's "don't force retrieval on purely-computational
    queries" claim was untestable: there was no route that could ever not
    retrieve.

    Routes to a NODE rather than straight to generate_insights precisely
    so the skip stays observable — routers in this module never mutate
    state, and a skip that left no mcp_call_log entry would be
    indistinguishable from an ordinary auto run where retrieval simply
    wasn't triggered (附录 CB.1.3).
    """
    should_retrieve, _ = rag_retrieval_decision(state)
    return "retrieve" if should_retrieve else "skip"


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
    from src.tools import glossary
    _, unresolved, _ = _resolve_intent_columns(
        state.get("parsed_intent") or {}, state.get("schema_columns") or [],
        # 阶段 3 轮次 4 (附录 CQ): the same alias index profile_and_clean will
        # use, off the same connect_schema-produced state. Omitting it here
        # would make this pre-check stricter than the real reconciliation and
        # send runs to refine_intent over mentions the glossary already
        # resolves — the drift this pre-check was explicitly built to avoid.
        alias_index=glossary.alias_index(state.get("glossary_entries") or []),
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
      - "clean"  → the cleaning loop hasn't hit a terminal stop condition
                   yet (附录 B.4) AND `cleaner_should_attempt_clean` (定案
                   #15, above) says so -- "auto" reads that straight off
                   has_critical_issues same as always; "force_on"/
                   "force_off" override it unconditionally
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

    # 定案 #15: force_off never loops back here in practice (profile_and_clean
    # sets a terminal cleaning_stop_reason on round 1, caught by the early
    # return above) -- routed through the same shared gate anyway as a
    # defensive backstop against that invariant ever drifting. force_on DOES
    # need this: a later round can find has_critical_issues has turned False
    # (post-clean re-profile) while validate_quality still hasn't passed and
    # no B.4 stop condition has fired -- without this, force_on would
    # silently fall back to auto-like early termination mid-loop.
    if cleaner_should_attempt_clean(has_critical) and iterations < _MAX_CLEAN_ITERATIONS:
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
