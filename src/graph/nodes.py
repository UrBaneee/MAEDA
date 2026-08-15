"""
Node function registry for the MAEDA LangGraph graph.

Nodes for completed phases use real agent implementations.
Nodes for future phases remain as labeled stubs.

All I/O-bound nodes are `async def` and run under a single event loop —
the graph must be invoked via `graph.ainvoke(state)` (see src/graph/builder.py
and every call site: scripts/run_eval.py, scripts/demo_scenarios.py,
ui/app.py, src/mcp_server/server.py). Previously each node individually
wrapped its work in `asyncio.run()`, spinning up and tearing down a fresh
event loop per node — harmless in isolation, but async clients created in
one node (e.g. the MCP transport's httpx.AsyncClient) don't survive being
used from a *different* loop in a later node, producing the "Event loop is
closed" errors visible throughout this project's logs. handle_error_node
does no I/O and is left as a plain sync function — LangGraph runs sync and
async nodes together transparently under ainvoke().
"""
import hashlib
from datetime import datetime, timezone
from typing import Optional

from src.config.settings import settings
from src.graph.router import _MAX_CLEAN_ITERATIONS
from src.mcp_client.fallback import SubSystemHardFailure
from src.state.graph_state import MAEDAState
from src.utils.logger import get_logger

logger = get_logger("maeda.nodes")

# ─── Agent / client singletons (lazy-init to avoid import-time construction) ──
_intent_parser = None
_subsystem_client = None
_data_connector = None
_analysis_agent = None
_viz_agent = None
_insight_agent = None
_guardrail_agent = None
_eval_runner = None
_run_store = None

def _get_intent_parser():
    global _intent_parser
    if _intent_parser is None:
        from src.agents.intent_parser import IntentParserAgent
        _intent_parser = IntentParserAgent()
    return _intent_parser

def _get_subsystem_client():
    global _subsystem_client
    if _subsystem_client is None:
        from src.mcp_client.fallback import build_subsystem_client
        _subsystem_client = build_subsystem_client()
    return _subsystem_client

def _get_data_connector():
    global _data_connector
    if _data_connector is None:
        from src.tools.data_connector import DataConnector
        _data_connector = DataConnector()
    return _data_connector

def _get_analysis_agent():
    global _analysis_agent
    if _analysis_agent is None:
        from src.agents.analysis_agent import AnalysisAgent
        _analysis_agent = AnalysisAgent()
    return _analysis_agent

def _get_viz_agent():
    global _viz_agent
    if _viz_agent is None:
        from src.agents.viz_agent import VizAgent
        _viz_agent = VizAgent()
    return _viz_agent

def _get_insight_agent():
    global _insight_agent
    if _insight_agent is None:
        from src.agents.insight_agent import InsightAgent
        _insight_agent = InsightAgent()
    return _insight_agent

def _get_guardrail_agent():
    global _guardrail_agent
    if _guardrail_agent is None:
        from src.agents.guardrail_agent import GuardrailAgent
        _guardrail_agent = GuardrailAgent()
    return _guardrail_agent

def _get_eval_runner():
    global _eval_runner
    if _eval_runner is None:
        from src.eval.runner import EvalRunner
        _eval_runner = EvalRunner()
    return _eval_runner

def _get_run_store():
    global _run_store
    if _run_store is None:
        from src.persistence.run_store import RunStore
        _run_store = RunStore()
    return _run_store


# ─── Cleaning-loop helpers (M7, TB0.5 v1 — 附录 B) ─────────────────────────────

# 附录 B.1's thresholds, frozen under quality_contract_version "1" (附录 B.7:
# "禁止在'1'版本号下改变上述任何判据"). Mirrors cleaner's own
# _HCI_MISSING_THRESHOLD/_HCI_DUP_THRESHOLD (mcp_app.py) so the signature
# computed here matches has_critical_issues's own logic exactly — if TB0.5
# ever bumps to v2 these must move in lockstep with the server-side values.
_HCI_MISSING_THRESHOLD = 0.10
_HCI_DUP_THRESHOLD = 0.05


def _file_is_readable(path: str) -> bool:
    """Existence + readability check for 附录 B's "输出文件存在、可读、被接管
    才置 cleaning_applied=true" requirement — a plain Path.is_file() can be
    true for a file you don't actually have permission to read."""
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _sha256_file(path: str) -> Optional[str]:
    """
    Self-computed content hash for 附录 B.4 stop condition #2 ("no actual
    change"). cleaner doesn't return input_sha256/output_sha256 yet (C3 not
    landed — 附录 E.1); 定案 #11's same-machine shared filesystem assumption
    means MAEDA can hash the files itself in the meantime. None means "could
    not read the file", which callers must treat as inconclusive, not as
    proof of "no diff".
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _cleaning_signature(validation) -> dict:
    """
    Recomputes the same three-dimension verdict has_critical_issues is
    built from (附录 B.0/B.1) out of validate_quality.details, so
    round-over-round comparison (附录 B.4 #3) can tell "identical failure
    two rounds running" apart from "still bad, but for a different/fewer
    reasons, still making progress". Comparing the single OR'd
    has_critical_issues boolean alone can't make that distinction.
    """
    details = validation.details or {}
    return {
        "missing": details.get("mean_null_ratio", 0.0) > _HCI_MISSING_THRESHOLD,
        "duplicate": details.get("duplicate_row_ratio", 0.0) > _HCI_DUP_THRESHOLD,
        "schema": details.get("schema_score", 1.0) < 1.0,
    }


def _trace(state: MAEDAState, agent_name: str, action: str, reasoning: str) -> MAEDAState:
    """Append a minimal decision trace record (no LLM, no cost)."""
    record = {
        "agent_name": agent_name,
        "action": action,
        "reasoning": reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": None,
        "outputs": None,
        "confidence": 1.0,
    }
    state["decision_trace"] = [*state.get("decision_trace", []), record]
    return state


# ─── Nodes ────────────────────────────────────────────────────────────────────

async def parse_intent_node(state: MAEDAState) -> MAEDAState:
    """Phase 2: real LLM-based intent parsing."""
    logger.info("Node: parse_intent | query=%s", state.get("user_query", "")[:80])
    state["current_phase"] = "plan"
    return await _get_intent_parser().process(state)


async def ask_clarification_node(state: MAEDAState) -> MAEDAState:
    """Phase 2: surface clarification question to user."""
    logger.info("Node: ask_clarification")
    state["clarification_count"] = state.get("clarification_count", 0) + 1
    return await _get_intent_parser().generate_clarification_question(state)


async def connect_and_profile_node(state: MAEDAState) -> MAEDAState:
    """
    Phase 4: connect to data source, extract schema + NL summary, delegate
    QC to MCP, and run at most one clean+validate+re-profile round if the
    data has critical issues (TB0.5 v1, ECOSYSTEM_INTEGRATION_PLAN.md 附录
    B). route_after_profiling decides whether another LangGraph tick
    through this node is needed — this function's job each tick is to do
    exactly one round and leave state honest about what happened.

    Fixes a confirmed real bug (M7): the cleaned file used to never get
    consumed by later rounds or by analysis — `data_sources[0].path` was
    never updated, so every loop iteration re-profiled and re-cleaned the
    *original* dirty file, and `data_quality_report` stayed frozen at the
    pre-clean state forever. This version writes `cleaned_path` back into
    `data_sources[0]`/`active_source`, re-profiles the cleaned file so the
    router sees fresh `has_critical_issues`, and calls `validate_quality`
    as the actual exit check (previously wired up but never invoked).
    """
    logger.info("Node: connect_and_profile_data")
    state["current_phase"] = "plan"

    sources = state.get("data_sources", [])
    if not sources:
        state["error"] = "No data source provided. Please upload a file or specify a data path."
        state["current_phase"] = "error"
        return state

    source = sources[0]
    source_path = source.get("path", "")
    connector = _get_data_connector()
    mcp_client = _get_subsystem_client()

    # M4 / 定案 #16: per-call run_id derived from the pipeline's own run_id,
    # now that cleaner's C3 actually honors it (附录 E P4 is resolved).
    # Each call gets a DISTINCT id, not one shared id for the whole pipeline
    # run: run_two_stage's stage1/stage2 directories are fixed names under
    # run_root, so reusing the same run_id across two separate clean_dataset
    # calls would silently overwrite round 1's artifacts with round 2's.
    # The shared prefix still lets every artifact from this pipeline run be
    # found/grouped by directory-name prefix (附录 E.2 P1).
    pipeline_run_id = state.get("run_id", "")
    round_index = state.get("iteration_count", 0)

    def _round_run_id(label: str, index: int) -> str:
        return f"{pipeline_run_id}_{label}{index}" if pipeline_run_id else ""

    # Step 1: Connect and extract schema + NL summary
    try:
        schema, nl_summary = await connector.connect_with_summary(source)
        state["active_source"] = schema.to_source_dict()
        state["schema_summary"] = nl_summary
        # Merge schema back into the source descriptor in state
        state["data_sources"] = [
            {**source, "schema": schema.to_dict(), "preview": schema.preview},
            *sources[1:],
        ]
        effective_path = source_path
    except Exception as exc:
        logger.warning("DataConnector failed for %s: %s", source_path, exc)
        # Schema extraction failed — still run MCP profiling on original path
        state["schema_summary"] = f"Schema unavailable: {exc}"
        effective_path = source_path

    # Step 2: Delegate quality profiling to Data Cleaner MCP
    try:
        report, prof_log = await mcp_client.profile_dataset(
            effective_path, run_id=_round_run_id("profile", round_index),
        )
    except SubSystemHardFailure as exc:
        # 错误处理矩阵 (M1): param/contract/data-input errors must not be
        # papered over in either mode -- surface as a real pipeline error
        # rather than proceeding as if profiling succeeded.
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Data quality profiling failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"profile_dataset hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), prof_log]
    state["data_quality_report"] = report.to_dict()

    if report.needs_review:  # 附录 B.3 — no cleaner tool emits this today; forward-compat
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged this dataset for manual review before any "
            "automatic cleaning; proceeding to analysis on the original data."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "profile_dataset returned needs_review; skipping auto-clean")

    if not report.has_critical_issues:
        # Nothing (more) to clean this round. If we already completed at
        # least one clean round earlier in this run, this is genuine
        # convergence, not "never needed cleaning" — record it either way
        # so a stale stop_reason from an earlier round can't linger.
        if state.get("iteration_count", 0) > 0:
            state["cleaning_stop_reason"] = "passed"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"Connected to {source_path}; critical_issues=False")

    # has_critical_issues=True from here on — run one clean+validate+re-profile round.
    if state.get("iteration_count", 0) >= _MAX_CLEAN_ITERATIONS:
        # route_after_profiling should have stopped looping before this
        # node ran again; this is a defensive backstop, not the primary
        # enforcement point, so it doesn't silently fall through to "ready"
        # without a caveat if it's ever reached some other way.
        state["cleaning_stop_reason"] = "max_rounds"
        state["cleaning_caveat"] = (
            f"Reached the maximum of {_MAX_CLEAN_ITERATIONS} automatic cleaning "
            "rounds without the data passing quality validation; proceeding "
            "to analysis on the best available (partially cleaned) version."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "max_rounds already reached; not attempting another clean")

    pre_clean_hash = _sha256_file(effective_path)

    try:
        # 定案 #3: call clean_dataset directly, without a get_cleaning_plan
        # round-trip. Confirmed live (not hypothetical) that the round-trip
        # is actually broken, not just an avoidable double-plan: cleaner's
        # get_cleaning_plan response is a MAEDA-facing reshaping of its
        # internal plan (steps filtered/renamed for display), not the full
        # internal "Plan Contract" document clean_dataset's `plan` argument
        # expects back — feeding it back gets rejected server-side with
        # "Plan missing top-level field: 'intent'". Passing no plan lets
        # clean_dataset generate and use its own internally-valid one.
        # M4: planner_mode read from Settings (定案 #6) and max_rounds
        # pinned to 1 (定案 #5) — MAEDA's graph loop is the outer round
        # controller, cleaner's own internal multi-round feedback loop
        # would be a second, uncoordinated one if left at its default.
        result, clean_log = await mcp_client.clean_dataset(
            effective_path,
            planner_mode=settings.data_cleaner_planner_mode,
            max_rounds=1,
            run_id=_round_run_id("clean", round_index + 1),
        )
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), clean_log]
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Data cleaning failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"cleaning hard failure: {exc}")

    # 附录 B.5: counts *completed* clean_dataset calls, incremented right
    # after the call returns successfully — not on every node entry (that
    # was the pre-M7 off-by-one that capped the loop at 2 actual clean
    # attempts despite _MAX_CLEAN_ITERATIONS = 3), and not gated on whether
    # the output turns out to be usable below ("调用成功返回后自增").
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # M4: record the execution plan cleaner actually used (live since its
    # C3) as its own decision-trace entry — separate from the round-summary
    # trace below because this is worth recording even on the branches that
    # return early afterward (needs_review, unusable output, no_diff, ...).
    plan = result.execution_plan
    _trace(
        state, "data_connector", "clean_dataset_execution_plan",
        f"planner_mode_requested={plan.get('planner_mode_requested')!r}, "
        f"planner_mode_used={plan.get('planner_mode_used')!r}, "
        f"planner_fallback_reason={plan.get('planner_fallback_reason')!r}, "
        f"steps={len(plan.get('steps', []))}",
    )

    if result.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged this cleaning run for manual review; "
            "stopping automatic cleaning and proceeding with the "
            "pre-cleaning data."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "clean_dataset returned needs_review")

    cleaned_path = result.cleaned_path
    if not cleaned_path or not _file_is_readable(cleaned_path):
        # cleaner reported success but the output can't actually be used —
        # 附录 B's "仅当输出文件存在、可读、被接管时才置 cleaning_applied=true"
        # means this must not be treated as if cleaning happened.
        state["cleaning_stop_reason"] = "no_diff"
        state["cleaning_caveat"] = (
            f"Data Cleaner reported success but its output ({cleaned_path!r}) "
            "doesn't exist or isn't readable; proceeding to analysis on the "
            "pre-cleaning data."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      f"clean_dataset output not usable: {cleaned_path!r}")

    post_clean_hash = _sha256_file(cleaned_path)
    no_diff = (
        pre_clean_hash is not None and post_clean_hash is not None
        and pre_clean_hash == post_clean_hash
    )

    # Adopt the cleaned file — this is the fix for the confirmed bug: later
    # rounds (and downstream analysis) must read the cleaned data, not the
    # original.
    state["cleaning_applied"] = True
    state["cleaning_summary"] = result.changes_summary
    state["data_sources"] = [
        {**state["data_sources"][0], "path": cleaned_path},
        *state["data_sources"][1:],
    ]
    try:
        cleaned_source = {**source, "path": cleaned_path}
        schema2, nl2 = await connector.connect_with_summary(cleaned_source)
        state["active_source"] = schema2.to_source_dict()
        state["schema_summary"] = nl2
    except Exception:
        pass  # Keep prior schema if re-extraction fails

    if no_diff:
        state["cleaning_stop_reason"] = "no_diff"
        state["cleaning_caveat"] = (
            "Data Cleaner ran but produced byte-identical output; stopping "
            "the cleaning loop rather than repeating a no-op round."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "clean_dataset produced no actual change (hash match)")

    # Re-profile the cleaned file so the router sees a fresh verdict —
    # this replaces the stale pre-clean report that used to linger for the
    # rest of the run.
    try:
        report2, prof2_log = await mcp_client.profile_dataset(
            cleaned_path, run_id=_round_run_id("reprofile", state["iteration_count"]),
        )
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"Re-profiling cleaned data failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"re-profile hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), prof2_log]
    state["data_quality_report"] = report2.to_dict()

    if report2.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged the cleaned data for manual review; "
            "stopping automatic cleaning."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "re-profile returned needs_review")

    # validate_quality as the actual exit check (附录 B.4 #1) — previously
    # wired up in mcp_client but never called from the graph at all.
    try:
        validation, val_log = await mcp_client.validate_quality(
            cleaned_path, run_id=_round_run_id("validate", state["iteration_count"]),
        )
    except SubSystemHardFailure as exc:
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["error"] = f"validate_quality failed ({exc.error_class}): {exc}"
        state["current_phase"] = "error"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"validate_quality hard failure: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), val_log]

    if validation.needs_review:
        state["cleaning_stop_reason"] = "needs_review"
        state["cleaning_caveat"] = (
            "Data Cleaner flagged the cleaned data for manual review during "
            "validation; stopping automatic cleaning."
        )
        return _trace(state, "data_connector", "connect_and_profile",
                      "validate_quality returned needs_review")

    if validation.passed:
        state["cleaning_stop_reason"] = "passed"
        return _trace(state, "data_connector", "connect_and_profile",
                      f"validate_quality passed after {state['iteration_count']} round(s)")

    # Not passed — check for stall (附录 B.4 #3) or regression (#4) against
    # the previous round before letting the loop continue.
    signature = _cleaning_signature(validation)
    prev_signature = state.get("cleaning_last_signature")
    prev_score = state.get("cleaning_last_score")

    if prev_signature is not None and signature == prev_signature:
        state["cleaning_stop_reason"] = "no_improvement"
        state["cleaning_caveat"] = (
            "Cleaning round produced the same quality verdict as the "
            "previous round (missing/duplicate/schema dimensions unchanged); "
            "stopping to avoid an unproductive loop. Proceeding to analysis "
            "with the best available cleaned data."
        )
    elif prev_score is not None and validation.score < prev_score:
        state["cleaning_stop_reason"] = "regressed"
        state["cleaning_caveat"] = (
            f"Cleaning round made quality worse (score {prev_score:.2f} -> "
            f"{validation.score:.2f}); stopping and flagging for manual "
            "review. Proceeding to analysis with the best available "
            "cleaned data."
        )
    elif state["iteration_count"] >= _MAX_CLEAN_ITERATIONS:
        state["cleaning_stop_reason"] = "max_rounds"
        state["cleaning_caveat"] = (
            f"Reached the maximum of {_MAX_CLEAN_ITERATIONS} automatic cleaning "
            "rounds without the data passing quality validation; proceeding "
            "to analysis on the best available (partially cleaned) version."
        )
    # else: still making progress and rounds remain -- leave
    # cleaning_stop_reason unset so route_after_profiling loops back.

    state["cleaning_last_signature"] = signature
    state["cleaning_last_score"] = validation.score

    return _trace(
        state, "data_connector", "connect_and_profile",
        f"round {state['iteration_count']} complete; validate_quality.passed=False, "
        f"score={validation.score}, stop_reason={state.get('cleaning_stop_reason')}",
    )


async def plan_analysis_node(state: MAEDAState) -> MAEDAState:
    """Phase 5: LLM generates AnalysisPlan from parsed intent + schema."""
    logger.info("Node: plan_analysis")
    state["current_phase"] = "plan"
    return await _get_analysis_agent().plan(state)


async def execute_analysis_node(state: MAEDAState) -> MAEDAState:
    """Phase 5: execute plan steps with dependency tracking and error recovery."""
    logger.info("Node: execute_analysis")
    state["current_phase"] = "execute"
    return await _get_analysis_agent().execute(state)


async def generate_viz_node(state: MAEDAState) -> MAEDAState:
    """Phase 6: recommend charts, generate static/interactive, caption, dashboard."""
    logger.info("Node: generate_viz")
    state["current_phase"] = "synthesize"
    return await _get_viz_agent().process(state)


async def retrieve_knowledge_node(state: MAEDAState) -> MAEDAState:
    """Phase 7: build focused retrieval query, delegate to RAG-MCP-Server."""
    logger.info("Node: retrieve_domain_knowledge")

    client = _get_subsystem_client()
    insight_agent = _get_insight_agent()
    # 7.1 Build focused retrieval query from analysis results + intent
    query = insight_agent.build_retrieval_query(state)

    try:
        chunks, log = await client.retrieve_knowledge(
            query, top_k=5, collection=settings.rag_collection
        )
    except SubSystemHardFailure as exc:
        # RAG is enrichment-only (CLAUDE.md: "MAEDA must be able to run
        # standalone"), unlike the cleaner's profiling call which is on the
        # critical path -- degrade to no domain context instead of aborting
        # analysis that has already been computed, even in strict mode.
        # The failure is still fully logged, just not treated as terminal.
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), exc.log]
        state["rag_context"] = []
        state["rag_sources"] = []
        return _trace(state, "insight_agent", "retrieve_knowledge",
                      f"RAG hard failure ({exc.error_class}), continuing without domain context: {exc}")
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), log]
    state["rag_context"] = [c.to_dict() for c in chunks]
    state["rag_sources"] = [
        {"source_file": c.source_file, "page": c.page, "chunk_id": c.chunk_id}
        for c in chunks if c.source_file
    ]
    return _trace(state, "insight_agent", "retrieve_knowledge",
                  f"Query: {query[:80]!r} → {len(state['rag_context'])} chunks")


async def generate_insights_node(state: MAEDAState) -> MAEDAState:
    """Phase 7: combine analysis results + RAG context → insights + report."""
    logger.info("Node: generate_insights")
    state["current_phase"] = "synthesize"
    return await _get_insight_agent().generate(state)


async def run_guardrails_node(state: MAEDAState) -> MAEDAState:
    """Phase 8: run all guardrail checks on outputs before user delivery."""
    logger.info("Node: run_guardrails")
    state["current_phase"] = "guardrail"
    # Increment guardrail_retry_count so route_after_guardrails can cap retry loops
    state["guardrail_retry_count"] = state.get("guardrail_retry_count", 0) + 1
    return await _get_guardrail_agent().process(state)


async def run_eval_node(state: MAEDAState) -> MAEDAState:
    """Phase 9: score the completed pipeline run against all eval metrics."""
    logger.info("Node: run_eval")
    state["current_phase"] = "complete"

    runner = _get_eval_runner()
    result = await runner.score(state, run_id=state.get("run_id"))
    state["eval_scores"] = {
        s.metric: {"score": s.score, "label": s.label, "reasoning": s.reasoning, "valid": s.valid}
        for s in result.scores
    }
    state["eval_scores"]["_aggregate"] = result.aggregate_score
    return state


def handle_error_node(state: MAEDAState) -> MAEDAState:
    checks = state.get("guardrail_checks") or []
    # Reaching handle_error via route_after_guardrails' "fail" branch means the
    # guardrail correctly blocked an unsafe/ungrounded output after exhausting
    # retries — a safe refusal, not a system crash. Any other path here (e.g.
    # no data source, connection failure) is a genuine pipeline error. This
    # distinction is what eval's error_rate/safe_refusal metrics key off of.
    is_safe_refusal = bool(checks) and checks[-1].get("overall_verdict") == "fail"
    state["error_type"] = "safe_refusal" if is_safe_refusal else "pipeline_error"

    if not state.get("error"):
        if is_safe_refusal:
            reason = checks[-1].get("retry_reason") or "Guardrail checks failed after maximum retries"
            state["error"] = reason
        else:
            state["error"] = "Pipeline terminated due to unrecoverable error"

    logger.error("Node: handle_error | error_type=%s | error=%s", state["error_type"], state.get("error"))
    state["current_phase"] = "error"
    state = _trace(state, "orchestrator", "handle_error",
                    f"{state['error_type']}: {state.get('error')}")
    return state


def persist_run_node(state: MAEDAState) -> MAEDAState:
    """
    Terminal node on every path (both run_eval and handle_error route here
    before END) — persists decision_trace/mcp_call_log to SQLite so they
    survive past this process, instead of vanishing the moment the graph
    finishes. See src/persistence/run_store.py.

    Persistence failures must never break the pipeline the user is
    actually waiting on: caught and logged, not raised.
    """
    try:
        _get_run_store().save_run(state)
    except Exception as exc:
        logger.warning("Failed to persist run %s: %s", state.get("run_id"), exc)
    return state
