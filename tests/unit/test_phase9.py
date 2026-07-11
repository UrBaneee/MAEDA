"""
Phase 9 tests — Evaluation Module.
Run with: pytest tests/unit/test_phase9.py -v
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.state.graph_state import initial_state


# ─── MetricScore ──────────────────────────────────────────────────────────────

def test_metric_score_to_dict():
    from src.eval.metrics import MetricScore
    ms = MetricScore("answer_relevance", 0.85, "pass", "Good answer")
    d = ms.to_dict()
    assert d["metric"] == "answer_relevance"
    assert d["score"] == 0.85
    assert d["label"] == "pass"


def test_metric_score_label_thresholds():
    from src.eval.metrics import _label
    assert _label(0.9) == "pass"
    assert _label(0.5) == "warn"
    assert _label(0.2) == "fail"


# ─── 9.4 Factual accuracy ────────────────────────────────────────────────────

def test_factual_accuracy_passes_overlap():
    from src.eval.metrics import score_factual_accuracy
    report = "Revenue was 100 in Q1 and 200 in Q2, totalling 300."
    results = [{"result_summary": "Q1=100, Q2=200", "failed": False}]
    ms = score_factual_accuracy(report, results)
    assert ms.score > 0.5
    assert ms.metric == "factual_accuracy"


def test_factual_accuracy_passes_no_numbers():
    from src.eval.metrics import score_factual_accuracy
    report = "Sales grew significantly."
    results = []
    ms = score_factual_accuracy(report, results)
    assert ms.score >= 0.5


def test_factual_accuracy_empty_report():
    from src.eval.metrics import score_factual_accuracy
    ms = score_factual_accuracy("", [])
    assert ms.score == 0.0
    assert ms.label == "fail"


def test_factual_accuracy_with_ground_truth():
    from src.eval.metrics import score_factual_accuracy
    report = "Revenue: 1000, customers: 50"
    results = []
    ground_truth = {"revenue": 1000, "customers": 50}
    ms = score_factual_accuracy(report, results, ground_truth)
    assert ms.score > 0.5


# ─── 9.5 Agent performance metrics ──────────────────────────────────────────

def test_intent_accuracy_high_confidence():
    from src.eval.metrics import score_intent_accuracy
    intent = {"query_type": "descriptive", "confidence": 0.95, "target_metrics": ["sales"]}
    ms = score_intent_accuracy(intent, "descriptive", ["sales"])
    assert ms.score > 0.7


def test_intent_accuracy_type_mismatch():
    from src.eval.metrics import score_intent_accuracy
    intent = {"query_type": "diagnostic", "confidence": 0.8, "target_metrics": []}
    ms = score_intent_accuracy(intent, "descriptive", [])
    assert ms.score < 0.9  # loses points for wrong type


def test_intent_accuracy_empty_intent():
    from src.eval.metrics import score_intent_accuracy
    ms = score_intent_accuracy({})
    assert ms.score == 0.0
    assert ms.label == "fail"


def test_tool_selection_all_success():
    from src.eval.metrics import score_tool_selection
    results = [
        {"method": "groupby", "failed": False},
        {"method": "correlation", "failed": False},
    ]
    ms = score_tool_selection(results)
    assert ms.score == 1.0
    assert ms.label == "pass"


def test_tool_selection_partial_failure():
    from src.eval.metrics import score_tool_selection
    results = [
        {"method": "groupby", "failed": False},
        {"method": "timeseries", "failed": True},
    ]
    ms = score_tool_selection(results)
    assert ms.score == 0.5


def test_tool_selection_no_steps():
    from src.eval.metrics import score_tool_selection
    ms = score_tool_selection([])
    assert ms.label == "warn"


def test_plan_efficiency_optimal():
    from src.eval.metrics import score_plan_efficiency
    results = [{"step": i} for i in range(4)]
    ms = score_plan_efficiency(results)
    assert ms.score == 1.0


def test_plan_efficiency_too_many_steps():
    from src.eval.metrics import score_plan_efficiency
    results = [{"step": i} for i in range(15)]
    ms = score_plan_efficiency(results)
    assert ms.score < 0.7


def test_chart_appropriateness_valid_charts():
    from src.eval.metrics import score_chart_appropriateness
    charts = [
        {"chart_type": "bar", "title": "Sales by Region"},
        {"chart_type": "line", "title": "Revenue Trend"},
    ]
    ms = score_chart_appropriateness(charts)
    assert ms.score == 1.0


def test_chart_appropriateness_no_charts():
    from src.eval.metrics import score_chart_appropriateness
    ms = score_chart_appropriateness([])
    assert ms.label == "warn"


# ─── System metrics ──────────────────────────────────────────────────────────

def test_system_metrics_no_errors():
    from src.eval.metrics import score_system_metrics
    state = {"token_usage": {}, "iteration_count": 1, "error": None}
    metrics = score_system_metrics(state)
    error_m = next(m for m in metrics if m.metric == "error_rate")
    assert error_m.score == 1.0


def test_system_metrics_with_error():
    from src.eval.metrics import score_system_metrics
    state = {"token_usage": {}, "iteration_count": 1, "error": "Something went wrong"}
    metrics = score_system_metrics(state)
    error_m = next(m for m in metrics if m.metric == "error_rate")
    assert error_m.score == 0.0
    assert error_m.label == "fail"


def test_system_metrics_safe_refusal_does_not_count_as_error_rate_failure():
    from src.eval.metrics import score_system_metrics
    state = {
        "token_usage": {}, "iteration_count": 1,
        "error": "Hallucinated revenue figure", "error_type": "safe_refusal",
    }
    metrics = score_system_metrics(state)
    error_m = next(m for m in metrics if m.metric == "error_rate")
    refusal_m = next(m for m in metrics if m.metric == "safe_refusal")
    assert error_m.score == 1.0
    assert error_m.label == "pass"
    assert refusal_m.score == 1.0
    assert refusal_m.label == "info"


def test_system_metrics_safe_refusal_absent_when_no_error():
    from src.eval.metrics import score_system_metrics
    state = {"token_usage": {}, "iteration_count": 1, "error": None}
    metrics = score_system_metrics(state)
    refusal_m = next(m for m in metrics if m.metric == "safe_refusal")
    assert refusal_m.score == 0.0


def test_system_metrics_genuine_crash_still_fails_error_rate():
    from src.eval.metrics import score_system_metrics
    state = {
        "token_usage": {}, "iteration_count": 1,
        "error": "No data source provided", "error_type": "pipeline_error",
    }
    metrics = score_system_metrics(state)
    error_m = next(m for m in metrics if m.metric == "error_rate")
    refusal_m = next(m for m in metrics if m.metric == "safe_refusal")
    assert error_m.score == 0.0
    assert error_m.label == "fail"
    assert refusal_m.score == 0.0


def test_system_metrics_retries():
    from src.eval.metrics import score_system_metrics
    state = {"token_usage": {}, "iteration_count": 3, "error": None}
    metrics = score_system_metrics(state)
    retry_m = next(m for m in metrics if m.metric == "retry_count")
    assert retry_m.raw_value == 2  # 3 iterations = 2 retries


# ─── 9.2 / 9.3 LLM-as-judge ─────────────────────────────────────────────────

def test_score_relevance_and_groundedness_with_llm():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "answer_relevance": 0.9,
        "groundedness": 0.85,
        "reasoning": "Report directly answers the question with evidence",
    })
    mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 15}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    from src.eval.metrics import score_relevance_and_groundedness
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        "Show sales by region",
        "# Report\nNorth region: $500K. South: $300K.",
        [{"result_summary": "North=500K South=300K", "failed": False}],
        [],
        llm=mock_llm,
    ))
    assert rel.metric == "answer_relevance"
    assert rel.score == 0.9
    assert gnd.metric == "groundedness"
    assert gnd.score == 0.85


def test_score_relevance_fallback_on_llm_error():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    from src.eval.metrics import score_relevance_and_groundedness
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        "q", "report", [], [], llm=mock_llm
    ))
    assert rel.score == 0.5
    assert gnd.score == 0.5
    assert rel.label == "warn"


def _mock_judge_response(relevance, groundedness, reasoning="r"):
    resp = MagicMock()
    resp.content = json.dumps({
        "answer_relevance": relevance, "groundedness": groundedness, "reasoning": reasoning,
    })
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    return resp


def test_score_relevance_makes_n_samples_judge_calls():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_judge_response(0.8, 0.8))

    from src.eval.metrics import score_relevance_and_groundedness
    asyncio.run(score_relevance_and_groundedness(
        "q", "report", [], [], llm=mock_llm, n_samples=5,
    ))
    assert mock_llm.ainvoke.await_count == 5


def test_score_relevance_aggregates_by_median_not_mean():
    # Median of [0.2, 0.9, 0.9] is 0.9, not the mean (~0.67) — a single
    # noisy low outlier shouldn't drag the score down as much as a mean would.
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_judge_response(0.2, 0.2),
        _mock_judge_response(0.9, 0.9),
        _mock_judge_response(0.9, 0.9),
    ])

    from src.eval.metrics import score_relevance_and_groundedness
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        "q", "report", [], [], llm=mock_llm, n_samples=3,
    ))
    assert rel.score == 0.9
    assert gnd.score == 0.9


def test_score_relevance_flags_high_judge_disagreement():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_judge_response(0.1, 0.1),
        _mock_judge_response(0.9, 0.9),
        _mock_judge_response(0.5, 0.5),
    ])

    from src.eval.metrics import score_relevance_and_groundedness
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        "q", "report", [], [], llm=mock_llm, n_samples=3,
    ))
    assert "disagreement" in rel.reasoning


# ─── 9.1 EvalRunner ──────────────────────────────────────────────────────────

def test_eval_runner_scores_state():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "answer_relevance": 0.88,
        "groundedness": 0.82,
        "reasoning": "Good",
    })
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    from src.eval.runner import EvalRunner
    runner = EvalRunner(llm=mock_llm)
    state = initial_state("Show revenue by region")
    state["report"] = "# Report\n\n## Findings\nNorth: 500K.\n\n## Rec\n- Keep going."
    state["analysis_results"] = [
        {"method": "groupby", "result_summary": "North=500K", "failed": False}
    ]
    state["parsed_intent"] = {
        "query_type": "descriptive", "confidence": 0.9,
        "target_metrics": ["revenue"], "dimensions": ["region"],
    }
    state["rag_context"] = []
    state["charts"] = [{"chart_type": "bar", "title": "Sales"}]

    result = asyncio.run(runner.score(state))
    assert result.aggregate_score > 0.0
    assert result.aggregate_score <= 1.0
    assert any(s.metric == "answer_relevance" for s in result.scores)
    assert any(s.metric == "error_rate" for s in result.scores)


def test_eval_runner_with_test_case():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "answer_relevance": 0.9, "groundedness": 0.9, "reasoning": "Perfect"
    })
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    from src.eval.runner import EvalRunner, GoldenTestCase
    runner = EvalRunner(llm=mock_llm)
    tc = GoldenTestCase(
        id="T01", query="Show sales by region",
        query_type="descriptive", expected_metrics=["sales"],
        expected_dimensions=["region"], ground_truth={},
    )
    state = initial_state("Show sales by region")
    state["report"] = "# Report\n\n## Findings\nNorth leads.\n\n## Rec\n- x"
    state["analysis_results"] = [{"method": "groupby", "result_summary": "North leads", "failed": False}]
    state["parsed_intent"] = {"query_type": "descriptive", "confidence": 0.9, "target_metrics": ["sales"]}
    state["rag_context"] = []
    state["charts"] = []

    result = asyncio.run(runner.score(state, test_case=tc))
    assert result.test_case_id == "T01"
    assert result.aggregate_score > 0.5


def test_eval_runner_reuses_existing_relevance_groundedness_instead_of_rejudging():
    """
    If run_eval_node already scored this state inside the graph,
    EvalRunner.score() must reuse those answer_relevance/groundedness
    values (they don't depend on test_case) instead of calling the judge a
    second time — verified by never wiring up ainvoke at all; a second
    judge call would error since there's no mock configured for it.
    """
    from src.eval.runner import EvalRunner, GoldenTestCase

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=AssertionError("judge should not be called again"))

    runner = EvalRunner(llm=mock_llm)
    tc = GoldenTestCase(
        id="T02", query="Show sales by region", query_type="descriptive",
        expected_metrics=["sales"], expected_dimensions=["region"], ground_truth={},
    )
    state = initial_state("Show sales by region")
    state["report"] = "# Report\n\n## Findings\nNorth leads.\n\n## Rec\n- x"
    state["analysis_results"] = [{"method": "groupby", "result_summary": "North leads", "failed": False}]
    state["parsed_intent"] = {"query_type": "descriptive", "confidence": 0.9, "target_metrics": ["sales"]}
    state["rag_context"] = []
    state["charts"] = []
    # Simulate run_eval_node having already scored this state in-graph.
    state["eval_scores"] = {
        "answer_relevance": {"score": 0.8, "label": "pass", "reasoning": "from graph run"},
        "groundedness": {"score": 0.7, "label": "pass", "reasoning": "from graph run"},
    }

    result = asyncio.run(runner.score(state, test_case=tc))
    assert result.score_by_metric("answer_relevance") == 0.8
    assert result.score_by_metric("groundedness") == 0.7
    mock_llm.ainvoke.assert_not_called()


def test_eval_result_score_by_metric():
    from src.eval.runner import EvalResult
    from src.eval.metrics import MetricScore
    result = EvalResult(
        run_id="r1", query="q",
        scores=[MetricScore("answer_relevance", 0.9, "pass")],
        aggregate_score=0.9,
    )
    assert result.score_by_metric("answer_relevance") == 0.9
    assert result.score_by_metric("nonexistent") is None


def test_eval_result_to_dict():
    from src.eval.runner import EvalResult
    from src.eval.metrics import MetricScore
    result = EvalResult(
        run_id="r1", query="q",
        scores=[MetricScore("error_rate", 1.0, "pass")],
        aggregate_score=0.9,
    )
    d = result.to_dict()
    assert d["run_id"] == "r1"
    assert d["aggregate_score"] == 0.9
    assert len(d["scores"]) == 1


def test_safe_refusal_excluded_from_aggregate_score():
    """safe_refusal is informational — it must not move the aggregate score."""
    from src.eval.runner import _aggregate_score
    from src.eval.metrics import MetricScore

    base_scores = [
        MetricScore("answer_relevance", 0.9, "pass"),
        MetricScore("groundedness", 0.9, "pass"),
        MetricScore("error_rate", 1.0, "pass"),
    ]
    without_refusal = _aggregate_score(base_scores)
    with_refusal_true = _aggregate_score(
        base_scores + [MetricScore("safe_refusal", 1.0, "info")]
    )
    with_refusal_false = _aggregate_score(
        base_scores + [MetricScore("safe_refusal", 0.0, "info")]
    )
    assert with_refusal_true == without_refusal
    assert with_refusal_false == without_refusal


# ─── 9.6 Golden test suite ───────────────────────────────────────────────────

def test_builtin_golden_suite_has_20_cases():
    from src.eval.runner import _builtin_golden_suite
    suite = _builtin_golden_suite()
    assert len(suite) >= 20


def test_builtin_golden_suite_covers_all_query_types():
    from src.eval.runner import _builtin_golden_suite
    suite = _builtin_golden_suite()
    types = {tc.query_type for tc in suite}
    assert "descriptive" in types
    assert "diagnostic" in types
    assert "comparative" in types
    assert "predictive" in types
    assert "exploratory" in types


# Cases where the query asks for something the demo datasets don't contain,
# or asks about the future — these carry a "_note" instead of a checkable
# numeric ground truth. Every other case must have a real, non-empty
# ground_truth backed by an actual computation over data/demo/*.
_KNOWN_UNANSWERABLE_CASES = {"D02", "DG04", "C03", "P01", "P02", "P03"}


def test_golden_suite_ground_truth_backfilled_from_json():
    from src.eval.runner import load_golden_suite
    suite = load_golden_suite()
    for tc in suite:
        if tc.id in _KNOWN_UNANSWERABLE_CASES:
            assert "_note" in tc.ground_truth, f"{tc.id} should document why it has no ground truth"
        else:
            numeric_values = {k: v for k, v in tc.ground_truth.items() if isinstance(v, (int, float))}
            assert numeric_values, f"{tc.id} should have at least one numeric ground_truth fact"


def test_builtin_and_json_golden_suites_have_matching_ground_truth():
    """The JSON file and the builtin fallback must stay in sync."""
    from src.eval.runner import _builtin_golden_suite, load_golden_suite
    json_suite = {tc.id: tc.ground_truth for tc in load_golden_suite()}
    builtin_suite = {tc.id: tc.ground_truth for tc in _builtin_golden_suite()}
    assert json_suite == builtin_suite


def test_golden_test_case_round_trip():
    from src.eval.runner import GoldenTestCase
    tc = GoldenTestCase(
        id="X01", query="test query", query_type="descriptive",
        expected_metrics=["revenue"], expected_dimensions=["region"],
        ground_truth={"total": 1000}, tags=["test"],
    )
    d = tc.to_dict()
    tc2 = GoldenTestCase.from_dict(d)
    assert tc2.id == "X01"
    assert tc2.ground_truth == {"total": 1000}


def test_load_golden_suite_fallback_to_builtin():
    from src.eval.runner import load_golden_suite
    suite = load_golden_suite("/nonexistent/path/test_suite.json")
    assert len(suite) >= 20


# ─── 9.8 Regression detection ────────────────────────────────────────────────

def test_regression_detector_catches_drop():
    from src.eval.runner import EvalResult, detect_regressions
    from src.eval.metrics import MetricScore

    baseline = EvalResult("b1", "q",
        [MetricScore("answer_relevance", 0.9, "pass"),
         MetricScore("groundedness", 0.85, "pass")],
        aggregate_score=0.87)

    current = EvalResult("c1", "q",
        [MetricScore("answer_relevance", 0.7, "warn"),   # -0.2 drop → critical
         MetricScore("groundedness", 0.82, "pass")],
        aggregate_score=0.76)

    alerts = detect_regressions(baseline, current)
    assert any(a.metric == "answer_relevance" for a in alerts)
    rel_alert = next(a for a in alerts if a.metric == "answer_relevance")
    assert rel_alert.severity == "critical"
    assert abs(rel_alert.drop - 0.2) < 0.01


def test_regression_detector_no_alerts_on_stable():
    from src.eval.runner import EvalResult, detect_regressions
    from src.eval.metrics import MetricScore

    baseline = EvalResult("b1", "q",
        [MetricScore("answer_relevance", 0.9, "pass")], aggregate_score=0.9)
    current = EvalResult("c1", "q",
        [MetricScore("answer_relevance", 0.88, "pass")], aggregate_score=0.88)

    alerts = detect_regressions(baseline, current)
    assert alerts == []


def test_regression_detector_warning_threshold():
    from src.eval.runner import EvalResult, detect_regressions
    from src.eval.metrics import MetricScore

    baseline = EvalResult("b1", "q",
        [MetricScore("groundedness", 0.8, "pass")], aggregate_score=0.8)
    current = EvalResult("c1", "q",
        [MetricScore("groundedness", 0.74, "warn")], aggregate_score=0.74)

    alerts = detect_regressions(baseline, current)
    assert any(a.metric == "groundedness" and a.severity == "warning" for a in alerts)


def test_regression_detects_aggregate_drop():
    from src.eval.runner import EvalResult, detect_regressions
    from src.eval.metrics import MetricScore

    baseline = EvalResult("b1", "q", [], aggregate_score=0.85)
    current = EvalResult("c1", "q", [], aggregate_score=0.60)

    alerts = detect_regressions(baseline, current)
    agg_alert = next((a for a in alerts if a.metric == "aggregate_score"), None)
    assert agg_alert is not None
    assert agg_alert.severity == "critical"


# ─── run_eval_node integration ───────────────────────────────────────────────

def test_run_eval_node_populates_eval_scores():
    import src.graph.nodes as _nodes
    from src.graph.nodes import run_eval_node

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "answer_relevance": 0.85, "groundedness": 0.80, "reasoning": "Good"
    })
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with patch("src.eval.metrics._build_eval_llm", return_value=mock_llm):
            _nodes._eval_runner = None
            state = initial_state("Show sales")
            state["report"] = "# Report\n\n## Findings\n- Sales up.\n\n## Rec\n- Keep going."
            state["analysis_results"] = [
                {"method": "groupby", "result_summary": "Sales up 10%", "failed": False}
            ]
            state["parsed_intent"] = {"query_type": "descriptive", "confidence": 0.9,
                                       "target_metrics": ["sales"]}
            state["rag_context"] = []
            state["charts"] = []
            result = run_eval_node(state)
            _nodes._eval_runner = None
    finally:
        loop.close()

    assert "eval_scores" in result
    assert "_aggregate" in result["eval_scores"]
    assert result["current_phase"] == "complete"
    assert result["eval_scores"]["_aggregate"] > 0.0
