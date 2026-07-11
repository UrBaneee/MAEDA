"""
LangGraph state graph construction for MAEDA.

Call build_graph() to get the compiled graph.
Import the singleton `graph` for use in the application.
"""
from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    ask_clarification_node,
    connect_and_profile_node,
    execute_analysis_node,
    generate_insights_node,
    generate_viz_node,
    handle_error_node,
    parse_intent_node,
    plan_analysis_node,
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

    # Delegated sub-system nodes (call Data Cleaner + RAG via MCP)
    g.add_node("connect_and_profile_data", connect_and_profile_node)
    g.add_node("retrieve_domain_knowledge", retrieve_knowledge_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    g.set_entry_point("parse_intent")

    # ── Edges ───────────────────────────────────────────────────────────────

    # Intent → clarify or proceed to data profiling
    g.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {"proceed": "connect_and_profile_data", "clarify": "ask_clarification"},
    )
    # Clarification loops back to re-parse
    g.add_edge("ask_clarification", "parse_intent")

    # Data profiling (may loop for cleaning then re-profile)
    g.add_conditional_edges(
        "connect_and_profile_data",
        route_after_profiling,
        {"clean": "connect_and_profile_data", "ready": "plan_analysis", "error": "handle_error"},
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

    # Terminal nodes
    g.add_edge("run_eval", END)
    g.add_edge("handle_error", END)

    return g.compile()


# Compiled singleton — import this in agents and the UI
graph = build_graph()
