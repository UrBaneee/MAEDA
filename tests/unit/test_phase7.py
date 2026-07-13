"""
Phase 7 tests — Insight Agent (RAG via MCP).
Run with: pytest tests/unit/test_phase7.py -v
"""
import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.state.graph_state import initial_state

# ─── Insight dataclass ────────────────────────────────────────────────────────

def test_insight_from_dict_basic():
    from src.agents.insight_agent import Insight
    d = {
        "finding": "North region leads",
        "evidence": "Step 1 groupby shows North=500",
        "confidence": 0.85,
        "recommendation": "Expand North operations",
    }
    ins = Insight.from_dict(d)
    assert ins.finding == "North region leads"
    assert ins.confidence == 0.85
    assert ins.impact == "high"
    assert ins.recommendation == "Expand North operations"
    assert isinstance(ins.evidence, list)


def test_insight_from_dict_evidence_list():
    from src.agents.insight_agent import Insight
    d = {
        "finding": "Anomaly detected",
        "evidence": ["Step 2: outlier at row 45"],
        "confidence": 0.6,
        "recommendation": "Investigate row 45",
    }
    ins = Insight.from_dict(d)
    assert ins.evidence == ["Step 2: outlier at row 45"]


def test_insight_impact_thresholds():
    from src.agents.insight_agent import Insight
    assert Insight.from_dict({"finding": "x", "confidence": 0.9}).impact == "high"
    assert Insight.from_dict({"finding": "x", "confidence": 0.65}).impact == "medium"
    assert Insight.from_dict({"finding": "x", "confidence": 0.3}).impact == "low"


def test_insight_to_dict_round_trip():
    from src.agents.insight_agent import Insight
    ins = Insight(
        finding="Sales up 20%",
        evidence=["step 1"],
        confidence=0.9,
        domain_context="Market trend Q4",
        impact="high",
        recommendation="Double down on Q4",
        sources=["knowledge_base.pdf"],
    )
    d = ins.to_dict()
    assert d["finding"] == "Sales up 20%"
    assert d["impact"] == "high"
    assert d["sources"] == ["knowledge_base.pdf"]


# ─── 7.1 Retrieval query builder ──────────────────────────────────────────────

def test_build_retrieval_query_uses_intent_and_findings():
    from src.agents.insight_agent import InsightAgent
    agent = InsightAgent(llm=MagicMock())
    state = initial_state("Show sales by region")
    state["parsed_intent"] = {
        "query_type": "descriptive",
        "target_metrics": ["sales"],
        "dimensions": ["region"],
    }
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "North leads with 500", "failed": False},
    ]
    query = agent.build_retrieval_query(state)
    assert "sales" in query.lower()
    assert len(query) > 5


def test_build_retrieval_query_falls_back_to_user_query():
    from src.agents.insight_agent import InsightAgent
    agent = InsightAgent(llm=MagicMock())
    state = initial_state("Analyse revenue trends")
    state["parsed_intent"] = {}
    state["analysis_results"] = []
    query = agent.build_retrieval_query(state)
    assert "revenue" in query.lower()


def test_build_retrieval_query_includes_top_findings():
    from src.agents.insight_agent import InsightAgent
    agent = InsightAgent(llm=MagicMock())
    state = initial_state("q")
    state["parsed_intent"] = {"query_type": "comparative", "target_metrics": ["orders"], "dimensions": []}
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "Jan had most orders", "failed": False},
        {"step": 2, "method": "comparison", "result_summary": "Q4 outperformed Q1", "failed": False},
    ]
    query = agent.build_retrieval_query(state)
    assert "Jan had most orders" in query or "Q4 outperformed" in query


# ─── 7.3 / 7.4 Insight generation ────────────────────────────────────────────

def test_generate_insights_with_llm():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        {
            "finding": "North region generates 60% of revenue",
            "evidence": "Step 1 groupby",
            "confidence": 0.9,
            "recommendation": "Focus marketing on North region",
        }
    ])
    mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 30}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("Show revenue by region")
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "North=600, South=400",
         "confidence": 0.9, "failed": False},
    ]
    state["rag_context"] = [{"content": "Enterprise market in North region is growing"}]
    state["rag_sources"] = [{"source_file": "market_report.pdf", "page": 3, "chunk_id": "c1"}]

    result = asyncio.run(agent.generate(state))
    assert len(result["insights"]) == 1
    assert result["insights"][0]["finding"] == "North region generates 60% of revenue"


def test_generate_insights_llm_fallback():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("Analyse data")
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "Key metric: 42",
         "confidence": 0.8, "failed": False},
    ]
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    assert len(result["insights"]) >= 1
    assert result["insights"][0]["finding"]


def test_generate_insights_no_analysis():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("q")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    # Fallback must produce at least one insight
    assert len(result["insights"]) >= 1


def test_generate_insights_source_attribution():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        {"finding": "Sales up", "evidence": "step 1", "confidence": 0.85, "recommendation": "Keep going"}
    ])
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("q")
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "Sales up 10%", "confidence": 0.85, "failed": False}
    ]
    state["rag_context"] = [{"content": "Industry context"}]
    state["rag_sources"] = [{"source_file": "kb.pdf", "page": 1, "chunk_id": "c1"}]

    result = asyncio.run(agent.generate(state))
    # Source attribution — sources should be set
    insight = result["insights"][0]
    assert "sources" in insight


def test_generate_logs_decision():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("q")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    assert any(t["action"] == "generate_insights" for t in result["decision_trace"])


def test_generate_tracks_token_usage():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        {"finding": "Finding A", "evidence": "e", "confidence": 0.8, "recommendation": "rec"}
    ])
    mock_response.usage_metadata = {"input_tokens": 50, "output_tokens": 25}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("q")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    assert "insight_agent" in result["token_usage"]


# ─── Multi-turn conversation history (roadmap #17) ────────────────────────────

def test_generate_appends_assistant_turn_to_conversation_history():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        {"finding": "North region generates 60% of revenue", "evidence": "e",
         "confidence": 0.9, "recommendation": "rec"}
    ])
    mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 30}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("Show revenue by region")
    state["parsed_intent"] = {
        "query_type": "descriptive", "target_metrics": ["revenue"],
        "dimensions": ["region"], "filters": [],
    }
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "North=600, South=400",
         "confidence": 0.9, "failed": False},
    ]
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    history = result["conversation_history"]
    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert "target_metrics=['revenue']" in history[0]["content"]
    assert "dimensions=['region']" in history[0]["content"]
    assert "North region generates 60% of revenue" in history[0]["content"]


def test_generate_preserves_prior_conversation_history():
    """A follow-up turn's assistant summary must be appended, not replace,
    whatever history already existed from earlier turns."""
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("Now break that down by quarter")
    state["conversation_history"] = [
        {"role": "user", "content": "Show revenue by region"},
        {"role": "assistant", "content": "query_type=descriptive; ..."},
    ]
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    assert len(result["conversation_history"]) == 3
    assert result["conversation_history"][0]["content"] == "Show revenue by region"


def test_format_assistant_turn_summary_includes_filters_and_time_range():
    from src.agents.insight_agent import Insight, _format_assistant_turn_summary
    state = initial_state("q")
    state["parsed_intent"] = {
        "query_type": "diagnostic", "target_metrics": ["revenue"],
        "dimensions": ["quarter"],
        "filters": [{"column": "region", "op": "=", "value": "North"}],
        "time_range": {"start": "2023-01-01", "end": "2023-12-31"},
    }
    insights = [Insight(finding="Revenue dropped in Q3", evidence=[], confidence=0.8,
                        domain_context="", impact="high", recommendation="", sources=[])]
    summary = _format_assistant_turn_summary(state, insights)
    assert "filters=[{'column': 'region'" in summary
    assert "time_range={'start': '2023-01-01'" in summary
    assert "Revenue dropped in Q3" in summary


def test_format_assistant_turn_summary_handles_no_insights():
    from src.agents.insight_agent import _format_assistant_turn_summary
    state = initial_state("q")
    state["parsed_intent"] = {"query_type": "descriptive", "target_metrics": [], "dimensions": []}
    summary = _format_assistant_turn_summary(state, [])
    assert "key_findings" not in summary
    assert "query_type=descriptive" in summary


# ─── 7.5 Report generator ─────────────────────────────────────────────────────

def test_report_is_markdown():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    # First call (insights) fails → fallback insights
    # Second call (report) also needs to produce a report
    insight_response = MagicMock()
    insight_response.content = json.dumps([
        {"finding": "Sales up", "evidence": "step 1", "confidence": 0.8, "recommendation": "Keep it up"}
    ])
    insight_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    report_response = MagicMock()
    report_response.content = "# Executive Summary\nSales are strong."
    report_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(side_effect=[insight_response, report_response])

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("Analyse revenue")
    state["analysis_results"] = [
        {"step": 1, "method": "groupby", "result_summary": "Revenue up", "confidence": 0.8, "failed": False}
    ]
    state["rag_context"] = []
    state["rag_sources"] = []

    result = asyncio.run(agent.generate(state))
    assert isinstance(result["report"], str)
    assert len(result["report"]) > 10


def test_report_fallback_contains_insights():
    from src.agents.insight_agent import InsightAgent, _rule_based_report, Insight
    ins = Insight(
        finding="Revenue grew 15%",
        evidence=["step 1"],
        confidence=0.85,
        domain_context="",
        impact="high",
        recommendation="Continue Q4 strategy",
        sources=[],
    )
    state = initial_state("Revenue analysis")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []
    state["charts"] = []
    state["data_quality_report"] = {}

    report = _rule_based_report(state, [ins])
    assert "Revenue grew 15%" in report
    assert "# MAEDA Analysis Report" in report
    assert "Continue Q4 strategy" in report


def test_report_fallback_handles_empty_insights():
    from src.agents.insight_agent import InsightAgent, _rule_based_report
    state = initial_state("q")
    state["analysis_results"] = []
    state["rag_context"] = []
    state["rag_sources"] = []
    state["charts"] = []
    state["data_quality_report"] = {}
    report = _rule_based_report(state, [])
    assert "# MAEDA Analysis Report" in report


def test_quality_note_reads_quality_issues_key():
    """DataQualityReport.to_dict() emits "quality_issues" — the note used to
    read a nonexistent "issues" key, so profiler findings never reached the
    report at all."""
    from src.agents.insight_agent import _format_quality_note
    report = {
        "row_count": 100,
        "quality_issues": [
            {"column": "age", "issue": "high_null_rate", "severity": "warning",
             "detail": "62.0% nulls"},
            {"column": None, "issue": "duplicate_rows", "severity": "warning",
             "detail": "3 fully duplicated rows (3.0%)"},
        ],
        "has_critical_issues": False,
    }
    note = _format_quality_note(report)
    assert "high_null_rate" in note
    assert "age" in note
    assert "duplicate_rows" in note


def test_quality_note_clean_report():
    from src.agents.insight_agent import _format_quality_note
    note = _format_quality_note({"row_count": 100, "quality_issues": [],
                                 "has_critical_issues": False})
    assert note == "Data quality checks passed."


# ─── 7.6 Source attribution ───────────────────────────────────────────────────

def test_source_attribution_from_rag_sources():
    from src.agents.insight_agent import InsightAgent
    mock_llm = MagicMock()
    # LLM returns insight without sources field
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        {"finding": "Costs rising", "evidence": "step 2", "confidence": 0.75, "recommendation": "Audit costs"}
    ])
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    report_resp = MagicMock()
    report_resp.content = "# Report\nCosts rising."
    report_resp.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(side_effect=[mock_response, report_resp])

    agent = InsightAgent(llm=mock_llm)
    state = initial_state("q")
    state["analysis_results"] = [
        {"step": 2, "method": "comparison", "result_summary": "Costs up", "confidence": 0.75, "failed": False}
    ]
    state["rag_context"] = [{"content": "Industry costs"}]
    state["rag_sources"] = [
        {"source_file": "cost_report.pdf", "page": 1, "chunk_id": "c1"},
    ]

    result = asyncio.run(agent.generate(state))
    insight = result["insights"][0]
    # Sources should reference the RAG doc
    assert "cost_report.pdf" in insight.get("sources", [])


# ─── Node integration ─────────────────────────────────────────────────────────

def test_retrieve_knowledge_node_uses_built_query():
    import src.graph.nodes as _nodes
    from src.graph.nodes import retrieve_knowledge_node

    mock_llm = MagicMock()

    with patch("src.agents.insight_agent._build_llm", return_value=mock_llm):
        _nodes._insight_agent = None
        _nodes._subsystem_client = None
        state = initial_state("Show sales by region")
        state["parsed_intent"] = {
            "query_type": "descriptive",
            "target_metrics": ["sales"],
            "dimensions": ["region"],
        }
        state["analysis_results"] = [
            {"step": 1, "method": "groupby", "result_summary": "North leads",
             "confidence": 0.9, "failed": False},
        ]
        result = asyncio.run(retrieve_knowledge_node(state))
        _nodes._insight_agent = None
        _nodes._subsystem_client = None

    # Node must set rag_context (fallback returns empty list, not error)
    assert "rag_context" in result
    assert isinstance(result["rag_context"], list)


def test_generate_insights_node_produces_report():
    import src.graph.nodes as _nodes
    from src.graph.nodes import generate_insights_node

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("no LLM"))

    with patch("src.agents.insight_agent._build_llm", return_value=mock_llm):
        _nodes._insight_agent = None
        state = initial_state("Analyse data")
        state["analysis_results"] = [
            {"step": 1, "method": "groupby", "result_summary": "Key result",
             "confidence": 0.8, "failed": False}
        ]
        state["rag_context"] = []
        state["rag_sources"] = []
        result = asyncio.run(generate_insights_node(state))
        _nodes._insight_agent = None

    assert "insights" in result
    assert "report" in result
    assert isinstance(result["report"], str)
    assert len(result["report"]) > 0
    assert result["current_phase"] == "synthesize"


# ─── End-to-end graph still compiles and runs ─────────────────────────────────

def test_graph_end_to_end_with_phase7():
    """Full graph run with all Phase 7 nodes wired."""
    import src.graph.nodes as _nodes
    from src.graph.builder import build_graph

    intent_resp = MagicMock()
    intent_resp.content = json.dumps({
        "query_type": "descriptive", "target_metrics": ["revenue"],
        "dimensions": ["region"], "filters": [], "time_range": None,
        "aggregation": "sum", "sort_by": None, "limit": None,
        "confidence": 0.95, "ambiguities": [],
    })
    intent_resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}

    plan_resp = MagicMock()
    plan_resp.content = "[]"
    plan_resp.usage_metadata = {"input_tokens": 5, "output_tokens": 5}

    insight_resp = MagicMock()
    insight_resp.content = json.dumps([
        {"finding": "Revenue strong", "evidence": "no steps", "confidence": 0.8, "recommendation": "Keep going"}
    ])
    insight_resp.usage_metadata = {"input_tokens": 15, "output_tokens": 20}

    report_resp = MagicMock()
    report_resp.content = "# Report\nRevenue is strong."
    report_resp.usage_metadata = {"input_tokens": 10, "output_tokens": 15}

    mock_intent_llm = MagicMock()
    mock_intent_llm.ainvoke = AsyncMock(return_value=intent_resp)

    mock_plan_llm = MagicMock()
    mock_plan_llm.ainvoke = AsyncMock(return_value=plan_resp)

    mock_insight_llm = MagicMock()
    mock_insight_llm.ainvoke = AsyncMock(side_effect=[insight_resp, report_resp])

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

    with patch("src.agents.intent_parser._build_llm", return_value=mock_intent_llm), \
         patch("src.agents.analysis_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.viz_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.agents.insight_agent._build_llm", return_value=mock_insight_llm), \
         patch("src.agents.guardrail_agent._build_llm", return_value=mock_plan_llm), \
         patch("src.eval.metrics._build_eval_llm", return_value=mock_plan_llm), \
         patch("src.tools.data_connector._build_llm", return_value=mock_plan_llm):
        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._data_connector = None
        _nodes._subsystem_client = mock_mcp

        g = build_graph()
        result = asyncio.run(g.ainvoke(initial_state("Show revenue by region",
                                        data_sources=[{"path": "data/demo/sales_data.csv", "type": "csv"}])))

        _nodes._intent_parser = None
        _nodes._analysis_agent = None
        _nodes._viz_agent = None
        _nodes._insight_agent = None
        _nodes._guardrail_agent = None
        _nodes._eval_runner = None
        _nodes._subsystem_client = None

    assert result["current_phase"] == "complete"
    assert "insights" in result
    assert "report" in result
    assert isinstance(result["report"], str)
