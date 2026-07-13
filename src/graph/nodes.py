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
from datetime import datetime, timezone

from src.config.settings import settings
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
    """Phase 4: connect to data source, extract schema + NL summary, delegate QC to MCP."""
    logger.info("Node: connect_and_profile_data")
    state["current_phase"] = "plan"
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    sources = state.get("data_sources", [])
    if not sources:
        state["error"] = "No data source provided. Please upload a file or specify a data path."
        state["current_phase"] = "error"
        return state

    source = sources[0]
    source_path = source.get("path", "")
    connector = _get_data_connector()
    mcp_client = _get_subsystem_client()

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
    report, prof_log = await mcp_client.profile_dataset(effective_path)
    state["mcp_call_log"] = [*state.get("mcp_call_log", []), prof_log]

    # Step 3: If critical issues, delegate cleaning then re-extract schema
    if report.has_critical_issues:
        plan, plan_log = await mcp_client.get_cleaning_plan(effective_path)
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), plan_log]
        result, clean_log = await mcp_client.clean_dataset(effective_path, plan)
        state["mcp_call_log"] = [*state.get("mcp_call_log", []), clean_log]
        state["cleaning_applied"] = True
        state["cleaning_summary"] = result.changes_summary
        # Re-extract schema from cleaned data
        if result.cleaned_path and result.cleaned_path != effective_path:
            try:
                cleaned_source = {**source, "path": result.cleaned_path}
                schema2, nl2 = await connector.connect_with_summary(cleaned_source)
                state["active_source"] = schema2.to_source_dict()
                state["schema_summary"] = nl2
            except Exception:
                pass  # Keep original schema if re-extraction fails

    state["data_quality_report"] = report.to_dict()
    return _trace(state, "data_connector", "connect_and_profile",
                  f"Connected to {source_path}; "
                  f"critical_issues={state['data_quality_report']['has_critical_issues']}")


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

    chunks, log = await client.retrieve_knowledge(
        query, top_k=5, collection=settings.rag_collection
    )
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
        s.metric: {"score": s.score, "label": s.label, "reasoning": s.reasoning}
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
