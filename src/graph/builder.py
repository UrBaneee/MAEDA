"""
LangGraph state graph construction for MAEDA.

Call build_graph() to get the compiled graph.
Import the singleton `graph` for use in the application.
"""
from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    ask_clarification_node,
    connect_schema,
    execute_analysis_node,
    generate_insights_node,
    generate_viz_node,
    handle_error_node,
    parse_intent_node,
    persist_run_node,
    plan_analysis_node,
    profile_and_clean,
    refine_intent_node,
    retrieve_knowledge_node,
    run_eval_node,
    run_guardrails_node,
    skip_retrieval_node,
)
from src.graph.router import (
    route_after_guardrails,
    route_after_intent,
    route_after_profiling,
    route_after_schema,
    route_after_viz,
)
from src.state.graph_state import MAEDAState


def build_graph() -> StateGraph:
    """Construct and compile the full MAEDA LangGraph state graph."""
    g = StateGraph(MAEDAState)

    # ── Register nodes ──────────────────────────────────────────────────────
    # MAEDA's own agents
    g.add_node("parse_intent", parse_intent_node)
    g.add_node("ask_clarification", ask_clarification_node)
    g.add_node("plan_analysis", plan_analysis_node)
    g.add_node("execute_analysis", execute_analysis_node)
    g.add_node("generate_viz", generate_viz_node)
    g.add_node("generate_insights", generate_insights_node)
    g.add_node("run_guardrails", run_guardrails_node)
    g.add_node("run_eval", run_eval_node)
    g.add_node("handle_error", handle_error_node)
    g.add_node("persist_run", persist_run_node)

    # Delegated sub-system nodes (call Data Cleaner + RAG via MCP)
    # E2 BO.1 split (ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ): what used to
    # be one "connect_and_profile_data" node is now three —
    # connect_schema (schema extraction + E1 table selection), an
    # optional refine_intent (schema-aware second-pass intent parse, only
    # on route_after_schema's "refine" edge), and profile_and_clean
    # (intent reconciliation + the MCP profile/clean/validate round).
    g.add_node("connect_schema", connect_schema)
    g.add_node("refine_intent", refine_intent_node)
    g.add_node("profile_and_clean", profile_and_clean)
    g.add_node("retrieve_domain_knowledge", retrieve_knowledge_node)
    # 阶段 3 收尾执行计划轮次 3: the "don't retrieve this run" branch. A node,
    # not a bare edge to generate_insights, because the skip has to leave a
    # `mode="skipped"` mcp_call_log record and a decision-trace entry behind
    # (附录 CB.1.3) and routers never mutate state.
    g.add_node("skip_retrieval", skip_retrieval_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    g.set_entry_point("parse_intent")

    # ── Edges ───────────────────────────────────────────────────────────────

    # Intent → clarify or proceed to data profiling
    g.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {"proceed": "connect_schema", "clarify": "ask_clarification"},
    )
    # Clarification loops back to re-parse
    g.add_edge("ask_clarification", "parse_intent")

    # connect_schema may or may not refine before profiling (E2, 附录 BQ) --
    # route_after_schema's "profile" key skips refine_intent entirely
    # (profile_and_clean's own top-of-function guard on
    # current_phase == "error" is still what makes connect_schema's "no
    # data source" exit a no-op, same as before E2 existed).
    g.add_conditional_edges(
        "connect_schema",
        route_after_schema,
        {"refine": "refine_intent", "profile": "profile_and_clean"},
    )
    g.add_edge("refine_intent", "profile_and_clean")

    # Data profiling (may loop for cleaning then re-profile). The self-loop
    # targets profile_and_clean directly, NOT connect_schema -- 附录 BQ:
    # this is what makes intent_refine_done's gate on route_after_schema
    # actually matter (without it, every clean round would re-trigger
    # refine_intent, one extra LLM call per round for no benefit, since
    # the intent doesn't change between rounds of the SAME run). Bypassing
    # connect_schema on later rounds also means profile_and_clean itself
    # must keep effective_dataset_path/schema_columns current across
    # rounds now (see its "Adopt the cleaned file" section) -- connect_schema
    # is no longer there to re-derive them every round the way it did when
    # the self-loop pointed back to it (E2 submission 1).
    g.add_conditional_edges(
        "profile_and_clean",
        route_after_profiling,
        {"clean": "profile_and_clean", "ready": "plan_analysis", "error": "handle_error"},
    )

    # Linear analysis pipeline
    g.add_edge("plan_analysis", "execute_analysis")
    g.add_edge("execute_analysis", "generate_viz")

    # RAG enrichment then insight generation. 阶段 3 收尾执行计划轮次 3 /
    # 附录 CC.7 裁定 4: this was an unconditional `add_edge` until now,
    # which is why 阶段 4's "purely-computational queries shouldn't be
    # forced through retrieval" could not be tested at all -- no route
    # existed that didn't retrieve. It is now a real conditional route
    # with two reachable destinations.
    #
    # Read the route honestly: `auto` still always returns "retrieve"
    # (route_after_viz → rag_retrieval_decision). 裁定 4 froze that
    # degeneration on purpose; the edge is what 轮次 3 owed, the
    # query-type judgement behind `auto` is separate work that has NOT
    # been done. Only MAEDA_RAG_MODE=force_off takes the "skip" key today.
    g.add_conditional_edges(
        "generate_viz",
        route_after_viz,
        {"retrieve": "retrieve_domain_knowledge", "skip": "skip_retrieval"},
    )
    g.add_edge("retrieve_domain_knowledge", "generate_insights")
    g.add_edge("skip_retrieval", "generate_insights")

    # Guardrails with feedback loop
    g.add_edge("generate_insights", "run_guardrails")
    g.add_conditional_edges(
        "run_guardrails",
        route_after_guardrails,
        {
            "passed": "run_eval",
            "retry": "execute_analysis",   # Guardrail feedback loop
            "fail": "handle_error",
        },
    )

    # Terminal nodes — both routed through persist_run so every pipeline
    # invocation is audited (success or failure) without either node
    # needing to know persistence exists.
    g.add_edge("run_eval", "persist_run")
    g.add_edge("handle_error", "persist_run")
    g.add_edge("persist_run", END)

    return g.compile()


# Compiled singleton — import this in agents and the UI
graph = build_graph()
