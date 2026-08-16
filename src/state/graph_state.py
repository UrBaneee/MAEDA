"""
MAEDAState — single source of truth for all inter-agent data.
All agents read from and write to this TypedDict; no unstructured message passing.
"""
import uuid
from typing import TypedDict, Optional, Literal


class MAEDAState(TypedDict):
    # === Identity ===
    run_id: str    # unique per pipeline invocation; see src/persistence/run_store.py

    # === User Input ===
    user_query: str
    conversation_history: list[dict]

    # === Intent Parsing ===
    parsed_intent: dict        # {type, entities, constraints, ambiguity_score}
    clarification_needed: bool
    clarification_question: Optional[str]

    # === Data Connection ===
    data_sources: list[dict]   # [{type, path/uri, schema, preview}]
    active_source: Optional[dict]
    schema_summary: str

    # === Data Quality (DELEGATED to Data Cleaner via MCP) ===
    data_quality_report: Optional[dict]   # From Data Cleaner MCP
    cleaning_applied: bool
    cleaning_summary: Optional[dict]  # CleaningResult.changes_summary — real shape is a dict (M3)
    # M7 / TB0.5 附录 B: set once the cleaning loop reaches a terminal state.
    # None while still looping. One of "passed"/"no_diff"/"no_improvement"/
    # "regressed"/"max_rounds"/"needs_review" once stopped (附录 B.4).
    cleaning_stop_reason: Optional[str]
    # Human-readable note attached whenever the loop stops without a clean
    # "passed" — surfaced by Insight Agent/report generation so a run that
    # proceeds on still-imperfect data says so instead of presenting it as
    # if cleaning fully succeeded.
    cleaning_caveat: Optional[str]
    # Round-over-round bookkeeping for 附录 B.4 stop conditions #3/#4 (no
    # improvement / regression). {"missing": bool, "duplicate": bool,
    # "schema": bool} — the same three dimensions has_critical_issues is
    # built from (附录 B.0/B.1), recomputed from validate_quality.details
    # since the cleaner doesn't expose them as a public verdict directly.
    cleaning_last_signature: Optional[dict]
    cleaning_last_score: Optional[float]
    # 附录 U.7: the last successful clean_dataset call's execution_plan
    # (plan_id/planner_mode_*/steps[], each step carrying U.5/U.6's
    # risk_tier/escalated_by/lossy/reversible/rollback_ref/impact/
    # confidence lineage). Kept in state, not just decision_trace, so the
    # Insight Agent can read it at report-generation time to build the
    # structured caveats U.7's disclosure chain requires -- decision_trace
    # entries are for human/log readability, not meant to be re-parsed.
    cleaning_execution_plan: Optional[dict]
    # 附录 U.2/U.7: the intent payload built for (and reused across) this
    # round's MCP calls -- kept so the Insight Agent can read
    # column_scope.columns for U.7's caveat "columns" field without
    # re-deriving it from parsed_intent a second time.
    cleaning_intent: Optional[dict]
    # 附录 R.3: derived trial-recording field -- "full" (cleaning ran and
    # completed a round) / "blocked_needs_review" (a step's risk tier, or
    # the server, flagged the round for manual review) / "none" (cleaning
    # never triggered). Set alongside cleaning_stop_reason at each terminal
    # branch of the cleaning loop so it's already captured whenever D0's
    # multi-trial support lands and needs it (R.3: "D0 落地时必须已经在记").
    cleaning_applied_level: Optional[str]
    # 附录 R.3: derived trial-recording field -- "full" (cleaning ran and
    # completed a round) / "blocked_needs_review" (a step's risk tier, or
    # the server, flagged the round for manual review) / "none" (cleaning
    # never triggered). Set alongside cleaning_stop_reason at each terminal
    # branch of the cleaning loop so it's already captured whenever D0's
    # multi-trial support lands and needs it (R.3: "D0 落地时必须已经在记").

    # === Analysis ===
    analysis_plan: list[dict]    # [{step, method, rationale}]
    analysis_results: list[dict] # [{step, result, confidence}]
    intermediate_data: Optional[dict]

    # === Visualization ===
    charts: list[dict]  # [{type, config, image_path}]

    # === Insight Generation (RAG via MCP) ===
    rag_context: list[dict]    # From RAG-MCP-Server
    rag_sources: list[dict]    # Source attribution from RAG
    insights: list[dict]       # [{finding, evidence, confidence, recommendation}]
    report: Optional[str]      # Final markdown report

    # === Guardrails ===
    guardrail_checks: list[dict]
    guardrail_passed: bool

    # === Eval ===
    eval_scores: Optional[dict]  # {accuracy, groundedness, relevance}

    # === Meta ===
    decision_trace: list[dict]   # Unified trace across all 3 systems
    mcp_call_log: list[dict]     # All MCP calls to sub-systems
    token_usage: dict            # {agent_name: {input, output, cost}}
    current_phase: Literal["plan", "execute", "synthesize", "guardrail", "complete", "error"]
    error: Optional[str]
    error_type: Optional[Literal["safe_refusal", "pipeline_error"]]  # what kind of failure `error` represents
    # Number of *completed* clean_dataset calls (附录 B.5) — incremented only
    # after clean_dataset returns successfully, not on every node entry.
    # Previously incremented unconditionally at the top of
    # connect_and_profile_node regardless of whether cleaning ran at all,
    # which was both an off-by-one against `_MAX_CLEAN_ITERATIONS` and
    # conflated "node re-entered" with "cleaning actually attempted".
    iteration_count: int
    guardrail_retry_count: int   # For guardrail retry loops (separate counter)
    clarification_count: int     # For clarification loops (cap at 1)


def initial_state(
    user_query: str,
    data_sources: Optional[list[dict]] = None,
    conversation_history: Optional[list[dict]] = None,
) -> MAEDAState:
    """Return a fully-initialized MAEDAState with safe defaults.

    conversation_history carries prior turns (see IntentParserAgent) so a
    follow-up query ("now break that down by quarter") can be resolved
    against what was actually asked/found last turn — each new pipeline
    run still starts from a fresh state, only this list persists across
    turns (threaded in by the caller, e.g. ui/app.py's session state).
    """
    return MAEDAState(
        run_id=uuid.uuid4().hex,
        user_query=user_query,
        conversation_history=conversation_history or [],
        parsed_intent={},
        clarification_needed=False,
        clarification_question=None,
        data_sources=data_sources or [],
        active_source=None,
        schema_summary="",
        data_quality_report=None,
        cleaning_applied=False,
        cleaning_summary=None,
        cleaning_stop_reason=None,
        cleaning_caveat=None,
        cleaning_last_signature=None,
        cleaning_last_score=None,
        cleaning_execution_plan=None,
        cleaning_intent=None,
        cleaning_applied_level=None,
        analysis_plan=[],
        analysis_results=[],
        intermediate_data=None,
        charts=[],
        rag_context=[],
        rag_sources=[],
        insights=[],
        report=None,
        guardrail_checks=[],
        guardrail_passed=False,
        eval_scores=None,
        decision_trace=[],
        mcp_call_log=[],
        token_usage={},
        current_phase="plan",
        error=None,
        error_type=None,
        iteration_count=0,
        guardrail_retry_count=0,
        clarification_count=0,
    )
