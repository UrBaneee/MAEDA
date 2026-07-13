"""
Phase 2 tests — Intent Parser Agent.
Run with: pytest tests/unit/test_phase2.py -v

All LLM calls are mocked; tests validate parsing logic, not model quality.
The 5 live-query accuracy cases (tasks 2.1/2.6) run against a deterministic
mock that mirrors expected LLM output.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.intent_parser import (
    IntentParserAgent,
    ParsedIntent,
    _parse_json,
)
from src.state.graph_state import initial_state


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_llm_response(data: dict) -> MagicMock:
    """Create a mock LLM response that returns a JSON string."""
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(data)
    mock_resp.usage_metadata = {"input_tokens": 50, "output_tokens": 30}
    return mock_resp


def _mock_agent(llm_response: dict, clarification_response: str = "") -> IntentParserAgent:
    """Return an IntentParserAgent whose LLM is fully mocked."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            _make_llm_response(llm_response),
            MagicMock(
                content=clarification_response,
                usage_metadata={"input_tokens": 20, "output_tokens": 15},
            ),
        ]
    )
    return IntentParserAgent(llm=mock_llm)


# ─── 2.2 ParsedIntent dataclass ───────────────────────────────────────────────

def test_parsed_intent_from_llm_response():
    data = {
        "query_type": "descriptive",
        "target_metrics": ["revenue"],
        "dimensions": ["region"],
        "filters": [{"column": "year", "op": ">=", "value": 2023}],
        "time_range": None,
        "aggregation": "sum",
        "sort_by": "revenue",
        "limit": 10,
        "confidence": 0.9,
        "ambiguities": [],
    }
    intent = ParsedIntent.from_llm_response(data, raw_query="test")
    assert intent.query_type == "descriptive"
    assert intent.target_metrics == ["revenue"]
    assert intent.confidence == 0.9
    assert intent.raw_query == "test"


def test_parsed_intent_to_dict_roundtrip():
    data = {
        "query_type": "comparative",
        "target_metrics": ["ctr"],
        "dimensions": ["campaign"],
        "filters": [],
        "time_range": None,
        "aggregation": None,
        "sort_by": None,
        "limit": None,
        "confidence": 0.85,
        "ambiguities": [],
    }
    intent = ParsedIntent.from_llm_response(data, raw_query="q")
    d = intent.to_dict()
    assert d["query_type"] == "comparative"
    assert d["confidence"] == 0.85


def test_parsed_intent_defaults_on_missing_fields():
    intent = ParsedIntent.from_llm_response({}, raw_query="empty")
    assert intent.query_type == "exploratory"
    assert intent.target_metrics == []
    assert intent.confidence == 0.5
    assert intent.ambiguities == []


# ─── 2.2 Parser invocation ───────────────────────────────────────────────────

def test_parser_returns_parsed_intent_in_state():
    agent = _mock_agent({
        "query_type": "descriptive",
        "target_metrics": ["sales"],
        "dimensions": ["region"],
        "filters": [],
        "time_range": None,
        "aggregation": "sum",
        "sort_by": None,
        "limit": None,
        "confidence": 0.95,
        "ambiguities": [],
    })
    state = initial_state("Show total sales by region")
    result = asyncio.run(agent.process(state))
    assert result["parsed_intent"]["query_type"] == "descriptive"
    assert result["parsed_intent"]["target_metrics"] == ["sales"]
    assert result["clarification_needed"] is False


def test_parser_logs_decision_trace():
    agent = _mock_agent({
        "query_type": "descriptive",
        "target_metrics": ["revenue"],
        "dimensions": [],
        "filters": [],
        "time_range": None,
        "aggregation": None,
        "sort_by": None,
        "limit": None,
        "confidence": 0.9,
        "ambiguities": [],
    })
    state = initial_state("Show revenue")
    result = asyncio.run(agent.process(state))
    trace = result["decision_trace"]
    assert len(trace) == 1
    assert trace[0]["agent_name"] == "intent_parser"
    assert trace[0]["action"] == "parse_intent"


def test_parser_tracks_token_usage():
    agent = _mock_agent({
        "query_type": "descriptive",
        "target_metrics": ["revenue"],
        "dimensions": [],
        "filters": [],
        "time_range": None,
        "aggregation": None,
        "sort_by": None,
        "limit": None,
        "confidence": 0.9,
        "ambiguities": [],
    })
    state = initial_state("Show revenue")
    result = asyncio.run(agent.process(state))
    assert "intent_parser" in result["token_usage"]
    assert result["token_usage"]["intent_parser"]["total_tokens"] == 80  # 50 + 30


def test_parser_preserves_other_agents_token_usage():
    """A single graph run passes state through many agents; parse_intent's
    write must not wipe out an earlier agent's entry already in state."""
    agent = _mock_agent({
        "query_type": "descriptive", "target_metrics": ["revenue"], "dimensions": [],
        "filters": [], "time_range": None, "aggregation": None, "sort_by": None,
        "limit": None, "confidence": 0.9, "ambiguities": [],
    })
    state = initial_state("Show revenue")
    state["token_usage"] = {"other_agent": {"input_tokens": 1, "output_tokens": 1,
                                             "total_tokens": 2, "cost_usd": 0.0, "calls": 1}}
    result = asyncio.run(agent.process(state))
    assert "other_agent" in result["token_usage"]
    assert "intent_parser" in result["token_usage"]


def test_parser_does_not_leak_token_usage_across_queries():
    """Roadmap Tier 3 #18: IntentParserAgent is a module-level singleton
    (src/graph/nodes.py._get_intent_parser), reused for every query in the
    process. Confirmed live that the pre-fix version's second query showed
    query1 + query2's tokens combined. Two independent process() calls on
    the same agent instance must each report only their own query's cost.

    Uses a reusable return_value mock (not _mock_agent's single-use
    side_effect queue, which is consumed after one process() call and is
    meant to model the parse-then-clarification sequence within *one*
    query, not two independent queries)."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response({
        "query_type": "descriptive", "target_metrics": ["revenue"], "dimensions": [],
        "filters": [], "time_range": None, "aggregation": None, "sort_by": None,
        "limit": None, "confidence": 0.9, "ambiguities": [],
    }))
    agent = IntentParserAgent(llm=mock_llm)

    state_a = initial_state("Show revenue")
    result_a = asyncio.run(agent.process(state_a))

    state_b = initial_state("Show costs")
    result_b = asyncio.run(agent.process(state_b))

    assert result_a["token_usage"]["intent_parser"]["total_tokens"] == 80
    assert result_b["token_usage"]["intent_parser"]["total_tokens"] == 80  # not 160
    assert result_b["token_usage"]["intent_parser"]["calls"] == 1  # not 2


def test_parser_adds_to_conversation_history():
    agent = _mock_agent({
        "query_type": "descriptive",
        "target_metrics": ["orders"],
        "dimensions": [],
        "filters": [],
        "time_range": None,
        "aggregation": "count",
        "sort_by": None,
        "limit": None,
        "confidence": 0.9,
        "ambiguities": [],
    })
    state = initial_state("How many orders?")
    result = asyncio.run(agent.process(state))
    history = result["conversation_history"]
    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "How many orders?"}


# ─── 2.3 Ambiguity detection ─────────────────────────────────────────────────

def test_low_confidence_triggers_clarification():
    agent = _mock_agent(
        {
            "query_type": "exploratory",
            "target_metrics": [],
            "dimensions": [],
            "filters": [],
            "time_range": None,
            "aggregation": None,
            "sort_by": None,
            "limit": None,
            "confidence": 0.5,
            "ambiguities": ["unclear metric", "unclear dimension"],
        },
        clarification_response="Which metric would you like to analyze?",
    )
    state = initial_state("Tell me something interesting")
    result = asyncio.run(agent.process(state))
    assert result["clarification_needed"] is True
    assert result["clarification_question"] == "Which metric would you like to analyze?"


def test_high_confidence_no_clarification():
    agent = _mock_agent({
        "query_type": "descriptive",
        "target_metrics": ["revenue"],
        "dimensions": ["quarter"],
        "filters": [],
        "time_range": None,
        "aggregation": "sum",
        "sort_by": None,
        "limit": None,
        "confidence": 0.95,
        "ambiguities": [],
    })
    state = initial_state("Total revenue per quarter")
    result = asyncio.run(agent.process(state))
    assert result["clarification_needed"] is False
    assert result["clarification_question"] is None


def test_ambiguities_present_triggers_clarification():
    """Ambiguities list alone (even with confidence >= 0.7) triggers clarification."""
    agent = _mock_agent(
        {
            "query_type": "descriptive",
            "target_metrics": ["metric_x"],
            "dimensions": [],
            "filters": [],
            "time_range": None,
            "aggregation": None,
            "sort_by": None,
            "limit": None,
            "confidence": 0.75,
            "ambiguities": ["metric_x is not in the schema"],
        },
        clarification_response="Which column did you mean by 'metric_x'?",
    )
    state = initial_state("Show metric_x")
    result = asyncio.run(agent.process(state))
    assert result["clarification_needed"] is True


# ─── 2.4 Clarification flow ───────────────────────────────────────────────────

def test_clarification_node_surfaces_question():
    agent = IntentParserAgent(llm=MagicMock())
    state = initial_state("test")
    state["clarification_question"] = "What time range are you interested in?"
    result = asyncio.run(agent.generate_clarification_question(state))
    history = result["conversation_history"]
    assert any(m["content"] == "What time range are you interested in?" for m in history)
    assert result["clarification_needed"] is False
    assert result["clarification_question"] is None


def test_clarification_node_fallback_message():
    """When no question was stored, a default message is used."""
    agent = IntentParserAgent(llm=MagicMock())
    state = initial_state("test")
    state["clarification_question"] = None
    result = asyncio.run(agent.generate_clarification_question(state))
    history = result["conversation_history"]
    assert len(history) == 1
    assert "details" in history[0]["content"].lower()


def test_clarification_node_logs_trace():
    agent = IntentParserAgent(llm=MagicMock())
    state = initial_state("test")
    state["clarification_question"] = "Please clarify."
    result = asyncio.run(agent.generate_clarification_question(state))
    assert result["decision_trace"][-1]["action"] == "ask_clarification"


# ─── 2.5 Schema-aware parsing ────────────────────────────────────────────────

def test_schema_injected_into_prompt():
    """System prompt must include schema when schema_summary is set."""
    captured_messages = []

    async def capturing_invoke(messages):
        captured_messages.extend(messages)
        return _make_llm_response({
            "query_type": "descriptive",
            "target_metrics": ["revenue"],
            "dimensions": ["region"],
            "filters": [],
            "time_range": None,
            "aggregation": "sum",
            "sort_by": None,
            "limit": None,
            "confidence": 0.92,
            "ambiguities": [],
        })

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=capturing_invoke)
    agent = IntentParserAgent(llm=mock_llm)

    state = initial_state("Show revenue by region")
    state["schema_summary"] = "Table: sales | Columns: revenue (float), region (str)"
    asyncio.run(agent.process(state))

    system_content = captured_messages[0].content
    assert "sales" in system_content
    assert "revenue" in system_content
    assert "region" in system_content


def test_no_schema_uses_base_prompt():
    """Without schema_summary the system prompt should not mention 'Available Data Schema'."""
    captured_messages = []

    async def capturing_invoke(messages):
        captured_messages.extend(messages)
        return _make_llm_response({
            "query_type": "exploratory",
            "target_metrics": [],
            "dimensions": [],
            "filters": [],
            "time_range": None,
            "aggregation": None,
            "sort_by": None,
            "limit": None,
            "confidence": 0.8,
            "ambiguities": [],
        })

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=capturing_invoke)
    agent = IntentParserAgent(llm=mock_llm)

    state = initial_state("Explore the data")
    state["schema_summary"] = ""
    asyncio.run(agent.process(state))

    system_content = captured_messages[0].content
    assert "Available Data Schema" not in system_content


# ─── Multi-turn conversation history (roadmap #17) ────────────────────────────

def test_conversation_history_injected_into_prompt():
    """A follow-up query must see prior turns in the system prompt so
    references like "that" or "add X too" can be resolved."""
    captured_messages = []

    async def capturing_invoke(messages):
        captured_messages.extend(messages)
        return _make_llm_response({
            "query_type": "descriptive", "target_metrics": ["revenue"],
            "dimensions": ["region", "quarter"], "filters": [],
            "time_range": None, "aggregation": "sum", "sort_by": None,
            "limit": None, "confidence": 0.9, "ambiguities": [],
        })

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=capturing_invoke)
    agent = IntentParserAgent(llm=mock_llm)

    state = initial_state("Now break that down by quarter too")
    state["conversation_history"] = [
        {"role": "user", "content": "Show revenue by region"},
        {"role": "assistant", "content": "query_type=descriptive; target_metrics=['revenue']; "
                                          "dimensions=['region']; key_findings=North leads at $450K"},
    ]
    asyncio.run(agent.process(state))

    system_content = captured_messages[0].content
    assert "### Conversation History (most recent last)" in system_content
    assert "Show revenue by region" in system_content
    assert "North leads at $450K" in system_content


def test_no_conversation_history_omits_section():
    """A fresh conversation (no prior turns) should not inject an empty
    history section into the prompt."""
    captured_messages = []

    async def capturing_invoke(messages):
        captured_messages.extend(messages)
        return _make_llm_response({
            "query_type": "descriptive", "target_metrics": ["revenue"],
            "dimensions": [], "filters": [], "time_range": None,
            "aggregation": "sum", "sort_by": None, "limit": None,
            "confidence": 0.9, "ambiguities": [],
        })

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=capturing_invoke)
    agent = IntentParserAgent(llm=mock_llm)

    state = initial_state("Show revenue by region")
    asyncio.run(agent.process(state))

    # The base prompt's rules describe how to use history *if* a section is
    # given; only the actual injected "### Conversation History" section
    # header should be absent when there's no history to inject.
    assert "### Conversation History (most recent last)" not in captured_messages[0].content


def test_format_history_caps_to_max_messages():
    from src.agents.intent_parser import _format_history, _MAX_HISTORY_MESSAGES
    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    formatted = _format_history(history)
    assert "turn 19" in formatted  # most recent kept
    assert "turn 0" not in formatted  # oldest dropped
    assert formatted.count("turn ") == _MAX_HISTORY_MESSAGES


def test_format_history_empty_returns_empty_string():
    from src.agents.intent_parser import _format_history
    assert _format_history(None) == ""
    assert _format_history([]) == ""


# ─── 2.6 Accuracy on 5 sample queries ────────────────────────────────────────
# These use deterministic mocks to validate the full parsing pipeline.

_SAMPLE_QUERIES = [
    # (query, expected_type, expected_metrics, expected_dimensions)
    (
        "What is the total revenue by region for Q1 2024?",
        "descriptive",
        ["revenue"],
        ["region"],
    ),
    (
        "Why did sales drop in the North region last quarter?",
        "diagnostic",
        ["sales"],
        ["region"],
    ),
    (
        "Compare average order value across all product categories",
        "comparative",
        ["order_value"],
        ["product_category"],
    ),
    (
        "Show me the distribution of customer ages",
        "descriptive",
        ["age"],
        [],
    ),
    (
        "Forecast monthly revenue for the next 6 months",
        "predictive",
        ["revenue"],
        [],
    ),
]

_SAMPLE_RESPONSES = [
    {"query_type": "descriptive", "target_metrics": ["revenue"], "dimensions": ["region"],
     "filters": [{"column": "quarter", "op": "=", "value": "Q1 2024"}],
     "time_range": None, "aggregation": "sum", "sort_by": None, "limit": None,
     "confidence": 0.95, "ambiguities": []},
    {"query_type": "diagnostic", "target_metrics": ["sales"], "dimensions": ["region"],
     "filters": [], "time_range": {"start": "last quarter", "end": "last quarter"},
     "aggregation": None, "sort_by": None, "limit": None,
     "confidence": 0.88, "ambiguities": []},
    {"query_type": "comparative", "target_metrics": ["order_value"],
     "dimensions": ["product_category"], "filters": [],
     "time_range": None, "aggregation": "avg", "sort_by": None, "limit": None,
     "confidence": 0.92, "ambiguities": []},
    {"query_type": "descriptive", "target_metrics": ["age"], "dimensions": [],
     "filters": [], "time_range": None, "aggregation": None, "sort_by": None,
     "limit": None, "confidence": 0.9, "ambiguities": []},
    {"query_type": "predictive", "target_metrics": ["revenue"], "dimensions": [],
     "filters": [], "time_range": {"start": "now", "end": "+6 months"},
     "aggregation": None, "sort_by": None, "limit": None,
     "confidence": 0.87, "ambiguities": []},
]


@pytest.mark.parametrize("idx", range(5))
def test_sample_query_accuracy(idx):
    query, exp_type, exp_metrics, exp_dims = _SAMPLE_QUERIES[idx]
    mock_resp = _SAMPLE_RESPONSES[idx]

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(mock_resp))
    agent = IntentParserAgent(llm=mock_llm)

    state = initial_state(query)
    result = asyncio.run(agent.process(state))

    pi = result["parsed_intent"]
    assert pi["query_type"] == exp_type, f"[{idx}] type mismatch"
    for m in exp_metrics:
        assert m in pi["target_metrics"], f"[{idx}] missing metric: {m}"
    for d in exp_dims:
        assert d in pi["dimensions"], f"[{idx}] missing dimension: {d}"


# ─── _parse_json helper ───────────────────────────────────────────────────────

def test_parse_json_clean():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_fences():
    text = '```json\n{"a": 1}\n```'
    assert _parse_json(text) == {"a": 1}


def test_parse_json_embedded_in_text():
    text = 'Here is the result: {"a": 1} done.'
    assert _parse_json(text) == {"a": 1}


def test_parse_json_raises_on_empty():
    with pytest.raises(ValueError):
        _parse_json("no json here at all")


# ─── Error handling ───────────────────────────────────────────────────────────

def test_empty_query_sets_error():
    agent = IntentParserAgent(llm=MagicMock())
    state = initial_state("")
    result = asyncio.run(agent.process(state))
    assert result["current_phase"] == "error"
    assert result["error"] is not None


def test_llm_failure_sets_error():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unreachable"))
    agent = IntentParserAgent(llm=mock_llm)
    state = initial_state("Show me sales")
    result = asyncio.run(agent.process(state))
    assert result["current_phase"] == "error"
    assert "LLM unreachable" in result["error"]
