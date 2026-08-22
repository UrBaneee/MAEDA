"""
MAEDAState — single source of truth for all inter-agent data.
All agents read from and write to this TypedDict; no unstructured message passing.
"""
import uuid
from typing import TypedDict, Optional, Literal

from src.config.settings import settings


class MAEDAState(TypedDict):
    # === Identity ===
    run_id: str    # unique per pipeline invocation; see src/persistence/run_store.py
    # 定案 #15 / 阶段 3 收尾执行计划轮次 1: the MAEDA_CLEANER_MODE/
    # MAEDA_RAG_MODE in effect for this run, snapshotted at initial_state()
    # construction time rather than read fresh from `settings` wherever
    # needed later. 附录 CB.3.4: unlike intent_refined (a per-run *result*,
    # only known once refine_intent has actually run), the three-state
    # switch is a run-level *configuration* fixed before the graph even
    # starts -- every case and every trial of one script invocation shares
    # the same value. Snapshotting it here means it's visible even on
    # error paths that never reach profile_and_clean/retrieve_knowledge_node
    # (e.g. "no data source"), and src/eval/runner.py's EvalResult can read
    # it straight from state like every other diagnostic field, instead of
    # re-importing settings.

    cleaner_mode: str   # settings.maeda_cleaner_mode snapshot ("auto"/"force_on"/"force_off")
    rag_mode: str       # settings.maeda_rag_mode snapshot ("auto"/"force_on"/"force_off")

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
    # E2 BO.1 split (ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ, submission 1):
    # node-to-node handoff between `connect_schema` and `profile_and_clean`
    # (src/graph/nodes.py) — not meant to be read anywhere else. Recomputed
    # fresh by connect_schema on every entry; effective_dataset_path is
    # 定案 #16's already-resolved path, schema_columns is the raw
    # ColumnInfo objects _resolve_intent_columns needs (the dict form
    # already stored in data_sources[0]['schema']['columns'] doesn't
    # support the attribute access that function does). None means schema
    # extraction failed this round (mirrors the pre-split `schema` local's
    # None case).
    effective_dataset_path: Optional[str]
    schema_columns: Optional[list]
    # 阶段 3 轮次 4 (ECOSYSTEM_INTEGRATION_PLAN.md 附录 CQ): the business
    # glossary (口径词表), already reconciled against this round's real schema
    # by the single gate `resolve_glossary` (src/tools/glossary.py) that
    # connect_schema runs the moment a schema exists. Written once per round
    # and read by every consumer — the two injection points ([2] the cleaner
    # intent payload, [3] the plan_analysis prompt) and the U.2 glossary_alias
    # match tier — so none of them can re-judge coverage and drift apart.
    #
    # glossary_coverage is the three-state field: "full" (every schema column
    # has a curated definition) / "partial" / "absent" (nothing curated covers
    # this data source — unknown data, no definitions). None only before
    # connect_schema has run. It is deliberately carried all the way onto the
    # wire and into the prompt: "absent" must be *stated*, because omitting the
    # glossary and having confirmed there is no definition look identical to
    # everything downstream (附录 CH.2/CI.2/CK.3/CO are all that same shape).
    glossary_coverage: Optional[str]
    glossary_entries: list[dict]           # schema-filtered, curation-layer form (carries `aliases`)
    glossary_uncovered_columns: list[str]  # real schema columns with no curated definition
    # E2 (ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ, submission 2): schema-aware
    # intent refinement -- a second `_parse()` call once connect_schema has
    # populated schema_summary, so the schema-injection branch that path
    # was always capable of (src/agents/intent_parser.py's `_parse`) can
    # actually fire. intent_refine_done is the conditional edge's gate
    # (route_after_schema, src/graph/router.py) -- set True the moment
    # refine_intent finishes (whether or not it actually called the LLM),
    # so it can never re-trigger within one round even if a future graph
    # change reintroduces a path back to connect_schema/refine_intent that
    # doesn't exist today. intent_refined is the D0-facing derived flag
    # (src/eval/runner.py's EvalResult, same three-step pattern as
    # cleaning_applied_level) -- None until refine_intent runs, then True
    # if the second parse succeeded or False if it raised and the
    # pre-refine intent was kept.
    intent_refine_done: bool
    intent_refined: Optional[bool]

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
    # 附录 CK.3: non-None means this run retrieved under
    # MAEDA_RAG_MODE=force_on but the retrieval was NOT a valid on-arm
    # one — a fallback/hard failure, or a weaker retrieval tier than
    # settings.maeda_rag_expected_retrieval_mode asserts. The string is
    # the concrete reason(s). src/eval/runner.py copies it onto
    # EvalResult and src/eval/trials.py::is_applicable then keeps the
    # trial out of pass@k and the continuous summaries. None on every
    # ordinary run, including every auto/force_off run — those make no
    # claim about RAG's contribution, so nothing about them can be
    # invalidated on this axis.
    rag_arm_invalid_reason: Optional[str]
    insights: list[dict]       # [{finding, evidence, confidence, recommendation}]
    report: Optional[str]      # Final markdown report

    # === Guardrails ===
    guardrail_checks: list[dict]
    guardrail_passed: bool

    # === Eval ===
    eval_scores: Optional[dict]  # {accuracy, groundedness, relevance}
    # E3 (阶段 3 执行顺序表轮次 5, 附录 CU): eval failing must not take the run
    # record down with it. run_eval_node catches anything EvalRunner.score
    # raises and records the classified failure here instead of propagating
    # out of graph.ainvoke -- which would skip persist_run_node entirely and
    # lose the whole decision_trace/mcp_call_log for that run.
    eval_error: Optional[str]

    # === Meta ===
    decision_trace: list[dict]   # Unified trace across all 3 systems
    mcp_call_log: list[dict]     # All MCP calls to sub-systems
    token_usage: dict            # {agent_name: {input, output, cost}}
    current_phase: Literal["plan", "execute", "synthesize", "guardrail", "complete", "error"]
    error: Optional[str]
    # E3 (附录 CU): `error_type` is now a PROJECTION of `terminal_state`
    # below, not an independent judgment -- see
    # src/state/terminal_state.py::legacy_error_type. Kept at its original
    # two values because src/eval/metrics.py::score_system_metrics, the
    # `error_type` column of every row already in logs/runs.db, and
    # scripts/run_eval.py's crash reporting are all frozen against them.
    error_type: Optional[Literal["safe_refusal", "pipeline_error"]]  # what kind of failure `error` represents
    # E3: how the run ended, as one of src/state/terminal_state.py's five
    # values ("success" included -- an explicit success state, not the
    # absence of an error). Written once by handle_error_node (failures) /
    # persist_run_node (successes); every other consumer reads it through
    # `resolve_terminal_state`. None only on a state that has not reached a
    # terminal node yet.
    terminal_state: Optional[str]
    # The sub-classification behind `terminal_state`. For anything that came
    # out of a sub-system call this is verbatim
    # SubSystemHardFailure.error_class (src/mcp_client/fallback.py::_classify)
    # -- E3 deliberately does not mint a second error taxonomy alongside
    # that one.
    terminal_detail: Optional[str]
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
        cleaner_mode=settings.maeda_cleaner_mode,
        rag_mode=settings.maeda_rag_mode,
        user_query=user_query,
        conversation_history=conversation_history or [],
        parsed_intent={},
        clarification_needed=False,
        clarification_question=None,
        data_sources=data_sources or [],
        active_source=None,
        schema_summary="",
        effective_dataset_path=None,
        schema_columns=None,
        glossary_coverage=None,
        glossary_entries=[],
        glossary_uncovered_columns=[],
        intent_refine_done=False,
        intent_refined=None,
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
        rag_arm_invalid_reason=None,
        insights=[],
        report=None,
        guardrail_checks=[],
        guardrail_passed=False,
        eval_scores=None,
        eval_error=None,
        decision_trace=[],
        mcp_call_log=[],
        token_usage={},
        current_phase="plan",
        error=None,
        error_type=None,
        terminal_state=None,
        terminal_detail=None,
        iteration_count=0,
        guardrail_retry_count=0,
        clarification_count=0,
    )
