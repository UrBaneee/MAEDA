"""
Phase 1 tests — Project Scaffold & LangGraph Foundation.
Run with: pytest tests/unit/test_phase1.py -v
"""
import pytest

# ─── 1.2 State definition ─────────────────────────────────────────────────────

def test_initial_state_all_fields():
    from src.state.graph_state import initial_state
    state = initial_state("What is the average revenue by region?")
    required_fields = [
        "user_query", "conversation_history",
        "parsed_intent", "clarification_needed", "clarification_question",
        "data_sources", "active_source", "schema_summary",
        "data_quality_report", "cleaning_applied", "cleaning_summary",
        "analysis_plan", "analysis_results", "intermediate_data",
        "charts",
        "rag_context", "rag_sources", "insights", "report",
        "guardrail_checks", "guardrail_passed",
        "eval_scores",
        "decision_trace", "mcp_call_log", "token_usage",
        "current_phase", "error", "iteration_count", "guardrail_retry_count",
        "clarification_count",
    ]
    for field in required_fields:
        assert field in state, f"Missing field: {field}"


def test_initial_state_defaults():
    from src.state.graph_state import initial_state
    state = initial_state("test query")
    assert state["user_query"] == "test query"
    assert state["current_phase"] == "plan"
    assert state["iteration_count"] == 0
    assert state["cleaning_applied"] is False
    assert state["guardrail_passed"] is False
    assert state["decision_trace"] == []
    assert state["token_usage"] == {}


# ─── 1.3 / 1.4 Graph + Router ─────────────────────────────────────────────────

def test_graph_compiles():
    """Graph must compile without errors."""
    from src.graph.builder import build_graph
    g = build_graph()
    assert g is not None


def test_graph_has_expected_nodes():
    from src.graph.builder import build_graph
    g = build_graph()
    node_names = set(g.nodes.keys())
    expected = {
        "parse_intent", "ask_clarification", "connect_and_profile_data",
        "plan_analysis", "execute_analysis", "generate_viz",
        "retrieve_domain_knowledge", "generate_insights",
        "run_guardrails", "run_eval", "handle_error",
    }
    assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


def test_graph_runs_end_to_end():
    """Graph must run through without exceptions using placeholder nodes."""
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.graph.builder import build_graph
    from src.state.graph_state import initial_state

    _mock_response = MagicMock()
    _mock_response.content = json.dumps({
        "query_type": "descriptive", "target_metrics": ["sales"],
        "dimensions": ["region"], "filters": [], "time_range": None,
        "aggregation": "sum", "sort_by": None, "limit": None,
        "confidence": 0.95, "ambiguities": [],
    })
    _mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_response)

    # Plan LLM response (empty plan keeps execution trivial)
    plan_response = MagicMock()
    plan_response.content = "[]"
    plan_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_plan_llm = MagicMock()
    mock_plan_llm.ainvoke = AsyncMock(return_value=plan_response)

    from src.mcp_client.models import DataQualityReport
    mock_mcp = MagicMock()
    mock_report = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)
    mock_mcp.profile_dataset = AsyncMock(
        return_value=(mock_report, {"system": "data_cleaner", "tool": "profile_dataset",
                                    "mode": "mcp", "args": {}, "duration_ms": 1.0})
    )
    mock_mcp.retrieve_knowledge = AsyncMock(
        return_value=([], {"system": "rag_server", "tool": "retrieve_with_metadata",
                           "mode": "fallback", "args": {}, "duration_ms": 1.0, "error": None})
    )

    with patch("src.agents.intent_parser._build_llm", return_value=mock_llm), \
         patch("src.agents.analysis_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.viz_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.insight_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.guardrail_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.eval.metrics._build_eval_llm", return_value=mock_plan_llm), \
         patch("src.tools.data_connector._build_llm", return_value=mock_plan_llm):
        import src.graph.nodes as _nodes
        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._data_connector = None
        _nodes._subsystem_client = mock_mcp
        g = build_graph()
        state = initial_state("Show me sales by region",
                              data_sources=[{"path": "data/demo/sales_data.csv", "type": "csv"}])
        result = g.invoke(state)
        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._subsystem_client = None

    assert result["current_phase"] == "complete"
    assert result["guardrail_passed"] is True
    assert len(result["decision_trace"]) > 0


def test_router_clarify_path():
    from src.graph.router import route_after_intent
    from src.state.graph_state import initial_state
    state = initial_state("What?")
    state["clarification_needed"] = True
    assert route_after_intent(state) == "clarify"


def test_router_proceed_path():
    from src.graph.router import route_after_intent
    from src.state.graph_state import initial_state
    state = initial_state("Show revenue by region for Q1 2024")
    state["clarification_needed"] = False
    assert route_after_intent(state) == "proceed"


def test_router_profiling_ready():
    from src.graph.router import route_after_profiling
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["data_quality_report"] = {"has_critical_issues": False}
    assert route_after_profiling(state) == "ready"


def test_router_profiling_clean():
    from src.graph.router import route_after_profiling
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["data_quality_report"] = {"has_critical_issues": True}
    state["iteration_count"] = 0
    assert route_after_profiling(state) == "clean"


def test_router_profiling_caps_iterations():
    from src.graph.router import route_after_profiling
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["data_quality_report"] = {"has_critical_issues": True}
    state["iteration_count"] = 10  # Way over limit
    assert route_after_profiling(state) == "ready"


def test_router_guardrails_passed():
    from src.graph.router import route_after_guardrails
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "approved", "passed": True}]
    assert route_after_guardrails(state) == "passed"


def test_router_guardrails_retry():
    from src.graph.router import route_after_guardrails
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "retry"}]
    state["guardrail_retry_count"] = 0
    assert route_after_guardrails(state) == "retry"


def test_router_guardrails_fail():
    from src.graph.router import route_after_guardrails
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "retry"}]
    state["guardrail_retry_count"] = 5  # Exhausted retries
    assert route_after_guardrails(state) == "fail"


# ─── handle_error_node: safe_refusal vs pipeline_error classification ────────

def test_handle_error_node_classifies_guardrail_fail_as_safe_refusal():
    from src.graph.nodes import handle_error_node
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["guardrail_checks"] = [{
        "overall_verdict": "fail",
        "retry_reason": "Hallucinated revenue figure",
    }]
    result = handle_error_node(state)
    assert result["error_type"] == "safe_refusal"
    assert result["error"] == "Hallucinated revenue figure"


def test_handle_error_node_classifies_missing_datasource_as_pipeline_error():
    from src.graph.nodes import handle_error_node
    from src.state.graph_state import initial_state
    state = initial_state("q")
    state["error"] = "No data source provided. Please upload a file or specify a data path."
    # No guardrail_checks — this path never reached guardrails
    result = handle_error_node(state)
    assert result["error_type"] == "pipeline_error"


def test_handle_error_node_defaults_to_pipeline_error_with_no_signal():
    from src.graph.nodes import handle_error_node
    from src.state.graph_state import initial_state
    state = initial_state("q")
    result = handle_error_node(state)
    assert result["error_type"] == "pipeline_error"
    assert result["error"] == "Pipeline terminated due to unrecoverable error"


# ─── 1.5 Logger ──────────────────────────────────────────────────────────────

def test_logger_returns_logger():
    from src.utils.logger import get_logger
    logger = get_logger("test.logger")
    assert logger is not None
    assert logger.name == "test.logger"


def test_decision_tracer_record():
    from src.utils.logger import DecisionTracer
    tracer = DecisionTracer("test_agent")
    record = tracer.log(
        action="test_action",
        reasoning="because reasons",
        inputs={"x": 1},
        outputs={"y": 2},
        confidence=0.9,
    )
    assert record["agent_name"] == "test_agent"
    assert record["action"] == "test_action"
    assert record["confidence"] == 0.9
    assert "timestamp" in record
    assert "trace_id" in record


# ─── 1.6 Cost Tracker ─────────────────────────────────────────────────────────

def test_cost_tracker_records_usage():
    from src.utils.cost_tracker import CostTracker
    tracker = CostTracker()
    rec = tracker.record(
        agent_name="intent_parser",
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        call_label="parse_intent",
    )
    assert rec.total_tokens == 150
    assert tracker.total_tokens == 150
    assert tracker.total_cost > 0


def test_cost_tracker_budget_exceeded():
    from src.utils.cost_tracker import BudgetExceededError, CostTracker
    tracker = CostTracker(max_cost_usd=0.000001)  # Tiny limit
    with pytest.raises(BudgetExceededError):
        tracker.record("agent", "gpt-4o", 10000, 10000)


def test_cost_tracker_per_agent():
    from src.utils.cost_tracker import CostTracker
    tracker = CostTracker()
    tracker.record("agent_a", "gpt-4o-mini", 100, 50)
    tracker.record("agent_b", "gpt-4o-mini", 200, 100)
    summary = tracker.to_state_dict()
    assert "agent_a" in summary
    assert "agent_b" in summary
    assert summary["agent_a"]["total_tokens"] == 150
    assert summary["agent_b"]["total_tokens"] == 300


# ─── 1.7 Base Agent ──────────────────────────────────────────────────────────

def test_base_agent_log_decision():
    from src.agents.base_agent import BaseAgent
    from src.state.graph_state import initial_state

    class DummyAgent(BaseAgent):
        async def process(self, state):
            return state

    agent = DummyAgent("dummy")
    state = initial_state("q")
    state = agent.log_decision(state, "test_action", "test_reasoning", confidence=0.8)
    assert len(state["decision_trace"]) == 1
    assert state["decision_trace"][0]["agent_name"] == "dummy"
    assert state["decision_trace"][0]["confidence"] == 0.8


def test_base_agent_track_cost():
    from src.agents.base_agent import BaseAgent
    from src.state.graph_state import initial_state

    class DummyAgent(BaseAgent):
        async def process(self, state):
            return state

    agent = DummyAgent("dummy")
    state = initial_state("q")
    state = agent.track_cost(state, "gpt-4o-mini", 100, 50)
    assert "dummy" in state["token_usage"]
    assert state["token_usage"]["dummy"]["total_tokens"] == 150


# ─── 1.8 Config System ───────────────────────────────────────────────────────

def test_settings_loads():
    from src.config.settings import MAEDASettings
    s = MAEDASettings()
    assert s.llm_model is not None
    assert s.llm_temperature >= 0.0


def test_settings_prompts_importable():
    from src.config import agent_prompts
    assert hasattr(agent_prompts, "INTENT_PARSER_SYSTEM")
    assert hasattr(agent_prompts, "ANALYSIS_PLANNER_SYSTEM")
    assert hasattr(agent_prompts, "GUARDRAIL_SYSTEM")
    assert len(agent_prompts.INTENT_PARSER_SYSTEM) > 50


def test_decision_trace_accumulated():
    """End-to-end: graph run should accumulate multiple trace records."""
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.graph.builder import build_graph
    from src.state.graph_state import initial_state

    _mock_response = MagicMock()
    _mock_response.content = json.dumps({
        "query_type": "descriptive", "target_metrics": ["orders"],
        "dimensions": ["month"], "filters": [], "time_range": None,
        "aggregation": "count", "sort_by": None, "limit": None,
        "confidence": 0.95, "ambiguities": [],
    })
    _mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_response)

    plan_response = MagicMock()
    plan_response.content = "[]"
    plan_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_plan_llm = MagicMock()
    mock_plan_llm.ainvoke = AsyncMock(return_value=plan_response)

    from src.mcp_client.models import DataQualityReport
    mock_mcp2 = MagicMock()
    mock_report2 = DataQualityReport(row_count=5, columns=[], quality_issues=[], has_critical_issues=False)
    mock_mcp2.profile_dataset = AsyncMock(
        return_value=(mock_report2, {"system": "data_cleaner", "tool": "profile_dataset",
                                     "mode": "mcp", "args": {}, "duration_ms": 1.0})
    )
    mock_mcp2.retrieve_knowledge = AsyncMock(
        return_value=([], {"system": "rag_server", "tool": "retrieve_with_metadata",
                           "mode": "fallback", "args": {}, "duration_ms": 1.0, "error": None})
    )

    with patch("src.agents.intent_parser._build_llm", return_value=mock_llm), \
         patch("src.agents.analysis_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.viz_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.insight_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.guardrail_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.eval.metrics._build_eval_llm", return_value=mock_plan_llm), \
         patch("src.tools.data_connector._build_llm", return_value=mock_plan_llm):
        import src.graph.nodes as _nodes
        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._data_connector = None
        _nodes._subsystem_client = mock_mcp2
        g = build_graph()
        result = g.invoke(initial_state("How many orders per month?",
                                        data_sources=[{"path": "data/demo/sales_data.csv", "type": "csv"}]))
        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._subsystem_client = None

    # At minimum one record per node that ran
    assert len(result["decision_trace"]) >= 7
