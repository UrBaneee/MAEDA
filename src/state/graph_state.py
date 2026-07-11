"""
MAEDAState — single source of truth for all inter-agent data.
All agents read from and write to this TypedDict; no unstructured message passing.
"""
from typing import TypedDict, Optional, Literal


class MAEDAState(TypedDict):
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
    cleaning_summary: Optional[str]

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
    iteration_count: int         # For data-cleaning retry loops
    guardrail_retry_count: int   # For guardrail retry loops (separate counter)
    clarification_count: int     # For clarification loops (cap at 1)


def initial_state(user_query: str, data_sources: Optional[list[dict]] = None) -> MAEDAState:
    """Return a fully-initialized MAEDAState with safe defaults."""
    return MAEDAState(
        user_query=user_query,
        conversation_history=[],
        parsed_intent={},
        clarification_needed=False,
        clarification_question=None,
        data_sources=data_sources or [],
        active_source=None,
        schema_summary="",
        data_quality_report=None,
        cleaning_applied=False,
        cleaning_summary=None,
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
        iteration_count=0,
        guardrail_retry_count=0,
        clarification_count=0,
    )
