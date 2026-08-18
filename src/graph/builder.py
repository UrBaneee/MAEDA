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
    retrieve_knowledge_node,
    run_eval_node,
    run_guardrails_node,
)
from src.graph.router import (
    route_after_guardrails,
    route_after_intent,
    route_after_profiling,
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
    # E2 BO.1 split (ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ, submission 1):
    # what used to be one "connect_and_profile_data" node is now two —
    # connect_schema (schema extraction + E1 table selection) followed by
    # profile_and_clean (intent reconciliation + the MCP
    # profile/clean/validate round). Submission 1 wires the self-loop back
    # to connect_schema, reproducing the pre-split node's exact per-round
    # behavior (schema is re-extracted every clean round, same as before)
    # — see nodes.py's connect_and_profile_node docstring and 附录 BQ for
    # why. Submission 2 (E2's refine_intent node) is the one that moves the
    # self-loop target to profile_and_clean, once there's an
    # intent_refine_done gate to actually protect.
    g.add_node("connect_schema", connect_schema)
    g.add_node("profile_and_clean", profile_and_clean)
    g.add_node("retrieve_domain_knowledge", retrieve_knowledge_node)

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

    # connect_schema always falls through to profile_and_clean (unconditional
    # -- profile_and_clean's own top-of-function guard on
    # current_phase == "error" is what makes connect_schema's "no data
    # source" exit a no-op here, not a graph-level branch).
    g.add_edge("connect_schema", "profile_and_clean")

    # Data profiling (may loop for cleaning then re-profile)
    g.add_conditional_edges(
        "profile_and_clean",
        route_after_profiling,
        {"clean": "connect_schema", "ready": "plan_analysis", "error": "handle_error"},
    )

    # Linear analysis pipeline
    g.add_edge("plan_analysis", "execute_analysis")
    g.add_edge("execute_analysis", "generate_viz")

    # RAG enrichment then insight generation
    g.add_edge("generate_viz", "retrieve_domain_knowledge")
    g.add_edge("retrieve_domain_knowledge", "generate_insights")

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
