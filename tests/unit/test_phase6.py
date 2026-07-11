"""
Phase 6 tests — Visualization Agent.
Run with: pytest tests/unit/test_phase6.py -v
"""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ─── 6.1 Chart recommender ────────────────────────────────────────────────────

def test_recommend_bar_for_grouped_data():
    from src.tools.chart_tool import recommend_chart
    result = [{"region": "North", "sales": 100}, {"region": "South", "sales": 200}]
    spec = recommend_chart(result, method="groupby", intent_type="descriptive")
    assert spec is not None
    assert spec.chart_type == "bar"
    assert spec.x_axis == "region"
    assert spec.y_axis == "sales"


def test_recommend_horizontal_bar_for_many_categories():
    from src.tools.chart_tool import recommend_chart
    result = [{"category": f"Cat{i}", "value": i * 10} for i in range(8)]
    spec = recommend_chart(result, method="groupby", intent_type="descriptive")
    assert spec is not None
    assert spec.chart_type == "horizontal_bar"


def test_recommend_line_for_time_series():
    from src.tools.chart_tool import recommend_chart
    result = {"trend": "upward", "n_periods": 12, "date_col": "month", "value_col": "revenue"}
    spec = recommend_chart(result, method="time_series")
    assert spec is not None
    assert spec.chart_type == "line"
    assert spec.x_axis == "month"


def test_recommend_heatmap_for_correlation():
    from src.tools.chart_tool import recommend_chart
    result = {"matrix": {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}, "method": "pearson"}
    spec = recommend_chart(result, method="correlation")
    assert spec is not None
    assert spec.chart_type == "heatmap"


def test_recommend_box_for_anomaly():
    from src.tools.chart_tool import recommend_chart
    result = {"n_outliers": 5, "method": "zscore", "column": "price"}
    spec = recommend_chart(result, method="anomaly_detection")
    assert spec is not None
    assert spec.chart_type == "box"


def test_recommend_scatter_for_regression():
    from src.tools.chart_tool import recommend_chart
    result = {"r_squared": 0.85, "coefficients": {"x1": 2.3}, "target": "y"}
    spec = recommend_chart(result, method="linear_regression")
    assert spec is not None
    assert spec.chart_type == "scatter"


def test_recommend_scatter_for_two_numeric_cols():
    from src.tools.chart_tool import recommend_chart
    result = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
    spec = recommend_chart(result, method="correlation")
    assert spec is not None
    assert spec.chart_type == "scatter"


def test_recommend_returns_none_for_empty():
    from src.tools.chart_tool import recommend_chart
    assert recommend_chart(None) is None
    assert recommend_chart({}) is None


def test_recommend_segment_comparison():
    from src.tools.chart_tool import recommend_chart
    result = {
        "segments": [{"segment": "A", "value": 100}, {"segment": "B", "value": 200}],
        "top_segment": "B",
        "value_col": "value",
        "segment_col": "segment",
    }
    spec = recommend_chart(result, method="comparison")
    assert spec is not None
    assert spec.chart_type in {"bar", "horizontal_bar"}


# ─── ChartSpec ────────────────────────────────────────────────────────────────

def test_chart_spec_to_dict():
    from src.tools.chart_tool import ChartSpec
    spec = ChartSpec(chart_type="bar", title="Test Chart", x_axis="x", y_axis="y")
    d = spec.to_dict()
    assert d["chart_type"] == "bar"
    assert d["title"] == "Test Chart"
    assert d["x_axis"] == "x"


# ─── 6.2 Static chart generation ──────────────────────────────────────────────

def test_generate_static_bar_chart():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(
        chart_type="bar",
        title="Sales by Region",
        x_axis="region",
        y_axis="sales",
        data=[{"region": "North", "sales": 100}, {"region": "South", "sales": 200}],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, output_dir=tmpdir)
        assert os.path.exists(path)
        assert path.endswith(".png")


def test_generate_static_line_chart():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(
        chart_type="line",
        title="Revenue over Time",
        x_axis="month",
        y_axis="revenue",
        data=[{"month": f"2024-{m:02d}", "revenue": m * 1000} for m in range(1, 7)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_scatter_chart():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(
        chart_type="scatter",
        title="X vs Y",
        x_axis="x",
        y_axis="y",
        data=[{"x": i * 1.0, "y": i * 2.0} for i in range(10)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_histogram():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    spec = ChartSpec(chart_type="histogram", title="Value Distribution", y_axis="value")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, df=df, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_box_chart():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    df = pd.DataFrame({"price": [10, 20, 30, 40, 200]})
    spec = ChartSpec(chart_type="box", title="Price Distribution", y_axis="price")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, df=df, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_heatmap():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(
        chart_type="heatmap",
        title="Correlation",
        matrix={"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_horizontal_bar():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(
        chart_type="horizontal_bar",
        title="Revenue by Country",
        x_axis="country",
        y_axis="revenue",
        data=[{"country": f"Country{i}", "revenue": i * 50} for i in range(8)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_static_chart(spec, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_static_chart_creates_dir():
    from src.tools.chart_tool import ChartSpec, generate_static_chart
    spec = ChartSpec(chart_type="bar", title="Test", x_axis="x", y_axis="y",
                     data=[{"x": "a", "y": 1}])
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "nested", "charts")
        path = generate_static_chart(spec, output_dir=new_dir)
        assert os.path.exists(path)


# ─── 6.3 Interactive chart generation (Plotly) ────────────────────────────────

def test_generate_interactive_bar_returns_json():
    from src.tools.chart_tool import ChartSpec, generate_interactive_chart
    spec = ChartSpec(
        chart_type="bar",
        title="Sales by Region",
        x_axis="region",
        y_axis="sales",
        data=[{"region": "North", "sales": 100}, {"region": "South", "sales": 200}],
    )
    result = generate_interactive_chart(spec)
    assert "json" in result
    assert "figure" in result
    assert len(result["json"]) > 10
    # Verify it's valid JSON
    parsed = json.loads(result["json"])
    assert "data" in parsed


def test_generate_interactive_line():
    from src.tools.chart_tool import ChartSpec, generate_interactive_chart
    spec = ChartSpec(
        chart_type="line",
        title="Trend",
        x_axis="month",
        y_axis="value",
        data=[{"month": f"M{i}", "value": i * 10} for i in range(6)],
    )
    result = generate_interactive_chart(spec)
    assert result["json"]


def test_generate_interactive_heatmap():
    from src.tools.chart_tool import ChartSpec, generate_interactive_chart
    spec = ChartSpec(
        chart_type="heatmap",
        title="Correlation",
        matrix={"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}},
    )
    result = generate_interactive_chart(spec)
    assert result["json"]


def test_generate_interactive_pie():
    from src.tools.chart_tool import ChartSpec, generate_interactive_chart
    spec = ChartSpec(
        chart_type="pie",
        title="Share by Region",
        x_axis="region",
        y_axis="sales",
        data=[{"region": "North", "sales": 300}, {"region": "South", "sales": 700}],
    )
    result = generate_interactive_chart(spec)
    assert result["json"]


def test_generate_interactive_includes_spec():
    from src.tools.chart_tool import ChartSpec, generate_interactive_chart
    spec = ChartSpec(chart_type="bar", title="Test", x_axis="x", y_axis="y",
                     data=[{"x": "a", "y": 1}])
    result = generate_interactive_chart(spec)
    assert result["spec"]["chart_type"] == "bar"


# ─── 6.4 Dashboard generation ─────────────────────────────────────────────────

def test_generate_dashboard_two_charts():
    from src.tools.chart_tool import ChartSpec, generate_dashboard
    specs = [
        ChartSpec(chart_type="bar", title="Chart 1", x_axis="x", y_axis="y",
                  data=[{"x": "a", "y": 1}, {"x": "b", "y": 2}]),
        ChartSpec(chart_type="bar", title="Chart 2", x_axis="x", y_axis="y",
                  data=[{"x": "c", "y": 3}, {"x": "d", "y": 4}]),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_dashboard(specs, output_dir=tmpdir)
        assert os.path.exists(path)
        assert path.endswith(".png")


def test_generate_dashboard_three_charts():
    from src.tools.chart_tool import ChartSpec, generate_dashboard
    specs = [
        ChartSpec(chart_type="bar", title=f"Chart {i}", x_axis="x", y_axis="y",
                  data=[{"x": "a", "y": i}])
        for i in range(3)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_dashboard(specs, output_dir=tmpdir)
        assert os.path.exists(path)


def test_generate_dashboard_empty_returns_empty():
    from src.tools.chart_tool import generate_dashboard
    result = generate_dashboard([])
    assert result == ""


# ─── 6.5 VizAgent unit tests ──────────────────────────────────────────────────

def test_viz_agent_skips_failed_results():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    agent = VizAgent(llm=MagicMock(), charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = [{"failed": True, "method": "groupby", "result": None, "step": 1}]

    result = asyncio.run(agent.process(state))
    assert result["charts"] == []


def test_viz_agent_generates_chart_for_valid_result():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Sales are highest in the North region."
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = VizAgent(llm=mock_llm, charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = [{
        "step": 1,
        "method": "groupby",
        "tool": "pandas_transform",
        "result": [{"region": "North", "sales": 100}, {"region": "South", "sales": 200}],
        "result_summary": "North had most sales",
        "confidence": 1.0,
        "warnings": [],
        "failed": False,
    }]

    result = asyncio.run(agent.process(state))
    assert len(result["charts"]) >= 1
    chart = result["charts"][0]
    assert chart["chart_type"] in {"bar", "horizontal_bar", "scatter", "line"}
    assert "image_path" in chart
    assert "plotly_json" in chart
    assert "caption" in chart


def test_viz_agent_generates_dashboard_for_multiple_results():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Chart caption."
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = VizAgent(llm=mock_llm, charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = [
        {
            "step": 1,
            "method": "groupby",
            "tool": "pandas_transform",
            "result": [{"region": "North", "sales": 100}, {"region": "South", "sales": 200}],
            "result_summary": "By region",
            "confidence": 1.0,
            "warnings": [],
            "failed": False,
        },
        {
            "step": 2,
            "method": "groupby",
            "tool": "pandas_transform",
            "result": [{"month": "Jan", "orders": 50}, {"month": "Feb", "orders": 60}],
            "result_summary": "By month",
            "confidence": 1.0,
            "warnings": [],
            "failed": False,
        },
    ]

    result = asyncio.run(agent.process(state))
    # 2 individual charts + 1 dashboard
    assert len(result["charts"]) == 3
    dashboard = result["charts"][-1]
    assert dashboard["chart_type"] == "dashboard"


def test_viz_agent_caption_fallback_on_llm_error():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    agent = VizAgent(llm=mock_llm, charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = [{
        "step": 1,
        "method": "groupby",
        "tool": "pandas_transform",
        "result": [{"category": "A", "count": 5}, {"category": "B", "count": 10}],
        "result_summary": "Category distribution",
        "confidence": 1.0,
        "warnings": [],
        "failed": False,
    }]

    result = asyncio.run(agent.process(state))
    assert len(result["charts"]) >= 1
    # Caption should be rule-based fallback, not empty
    assert len(result["charts"][0]["caption"]) > 0


def test_viz_agent_logs_decision():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Caption text."
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = VizAgent(llm=mock_llm, charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = []

    result = asyncio.run(agent.process(state))
    assert any(t["action"] == "generate_viz" for t in result["decision_trace"])


def test_viz_agent_no_results():
    from src.agents.viz_agent import VizAgent
    from src.state.graph_state import initial_state

    agent = VizAgent(llm=MagicMock(), charts_dir=tempfile.mkdtemp())
    state = initial_state("test")
    state["analysis_results"] = []

    result = asyncio.run(agent.process(state))
    assert result["charts"] == []


# ─── 6.5 Rule-based caption ───────────────────────────────────────────────────

def test_rule_based_caption_bar():
    from src.agents.viz_agent import _rule_based_caption
    from src.tools.chart_tool import ChartSpec
    spec = ChartSpec(chart_type="bar", title="Sales by Region")
    ar = {"result_summary": "North leads."}
    caption = _rule_based_caption(spec, ar)
    assert "Bar chart" in caption
    assert "North leads" in caption


def test_rule_based_caption_unknown_type():
    from src.agents.viz_agent import _rule_based_caption
    from src.tools.chart_tool import ChartSpec
    spec = ChartSpec(chart_type="treemap", title="Tree Map")
    ar = {"result_summary": ""}
    caption = _rule_based_caption(spec, ar)
    assert len(caption) > 0


# ─── generate_viz_node integration ────────────────────────────────────────────

def test_generate_viz_node_wires_viz_agent():
    from src.graph.nodes import generate_viz_node
    from src.state.graph_state import initial_state
    import src.graph.nodes as _nodes

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Caption."
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    # asyncio.run() closes the event loop; create a fresh one for the sync node wrapper
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with patch("src.agents.viz_agent._build_llm", return_value=mock_llm):
            _nodes._viz_agent = None
            state = initial_state("Show sales by region")
            state["analysis_results"] = []
            result = generate_viz_node(state)
            _nodes._viz_agent = None
    finally:
        loop.close()

    assert "charts" in result
    assert isinstance(result["charts"], list)
    assert result["current_phase"] == "synthesize"
