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


def test_factual_accuracy_matches_thousands_separator():
    """A naive digit regex splits '$1,363,760.55' into '1'/'363'/'760.55',
    none of which equal the raw ground-truth value — this must not happen."""
    from src.eval.metrics import score_factual_accuracy
    report = "Total revenue was $1,363,760.55 across all regions."
    ground_truth = {"north_region_revenue": 1363760.55}
    ms = score_factual_accuracy(report, [], ground_truth)
    assert ms.score == 1.0


def test_factual_accuracy_tolerates_llm_rounding():
    """LLM-written numbers are often rounded ($1,363,761 instead of
    1363760.55, or 0.35 instead of 0.3536) — exact string equality
    shouldn't zero these out."""
    from src.eval.metrics import score_factual_accuracy
    report = "Revenue was about $1,363,761. The correlation was 0.35."
    ground_truth = {"revenue": 1363760.55, "correlation": 0.3536}
    ms = score_factual_accuracy(report, [], ground_truth)
    assert ms.score == 1.0


def test_factual_accuracy_rejects_wrong_number_within_loose_absolute_tolerance():
    """A flat absolute tolerance would let a wrong correlation coefficient
    (0.7 vs true 0.3536) slip through; the tolerance must scale with the
    expected value's magnitude instead."""
    from src.eval.metrics import score_factual_accuracy
    report = "The correlation between spend and revenue was 0.7."
    ground_truth = {"correlation": 0.3536}
    ms = score_factual_accuracy(report, [], ground_truth)
    assert ms.score == 0.0


def test_numbers_match_helper():
    from src.eval.metrics import _numbers_match
    assert _numbers_match(1363761.0, 1363760.55)  # rounding
    assert _numbers_match(0.35, 0.3536)            # small-magnitude rounding
    assert not _numbers_match(0.7, 0.3536)         # genuinely wrong


def test_extract_numbers_strips_thousands_separators():
    from src.eval.metrics import _extract_numbers
    assert _extract_numbers("Revenue: $1,363,760.55") == {1363760.55}
    assert _extract_numbers("Counts: 118, 302") == {118.0, 302.0}


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


def test_step_success_rate_all_success():
    """This is what "tool_selection" used to mean (eval v2 Step 2d renamed
    it -- it never actually measured tool choice, only whether a step
    threw)."""
    from src.eval.metrics import score_step_success_rate
    results = [
        {"method": "groupby", "failed": False},
        {"method": "correlation", "failed": False},
    ]
    ms = score_step_success_rate(results)
    assert ms.score == 1.0
    assert ms.label == "pass"


def test_step_success_rate_partial_failure():
    from src.eval.metrics import score_step_success_rate
    results = [
        {"method": "groupby", "failed": False},
        {"method": "timeseries", "failed": True},
    ]
    ms = score_step_success_rate(results)
    assert ms.score == 0.5


def test_step_success_rate_no_steps():
    from src.eval.metrics import score_step_success_rate
    ms = score_step_success_rate([])
    assert ms.label == "warn"


def test_tool_selection_no_expected_tools_is_lenient():
    """No expected_tools authored for this case (e.g. an ad-hoc query with
    no GoldenTestCase) -- can't score tool choice without ground truth, so
    this must not penalize, unlike a real mismatch below."""
    from src.eval.metrics import score_tool_selection
    ms = score_tool_selection([{"tool": "pandas_transform"}], expected_tools=None)
    assert ms.score == 0.8
    assert ms.label == "pass"


def test_tool_selection_explicitly_no_tool_expected():
    """expected_tools=[] (authored) is distinct from expected_tools=None
    (not authored) -- an explicitly-no-specific-tool-expected case always
    passes regardless of what was actually used."""
    from src.eval.metrics import score_tool_selection
    ms = score_tool_selection([{"tool": "sql_query"}], expected_tools=[])
    assert ms.score == 1.0


def test_tool_selection_matches_expected_tool():
    from src.eval.metrics import score_tool_selection
    results = [{"tool": "time_series"}, {"tool": "pandas_transform"}]
    ms = score_tool_selection(results, expected_tools=["time_series"])
    assert ms.score == 1.0


def test_tool_selection_wrong_tool_scores_zero():
    """The old tool_selection (now step_success_rate) would score this
    1.0 -- the step didn't throw. The rebuilt metric must catch that the
    planner picked a tool with zero overlap with what was expected."""
    from src.eval.metrics import score_tool_selection
    results = [{"tool": "sql_query", "failed": False}]
    ms = score_tool_selection(results, expected_tools=["anomaly_detection"])
    assert ms.score == 0.0
    assert ms.label == "fail"


def test_tool_selection_hitting_just_one_of_several_acceptable_tools_scores_full():
    """expected_tools=[A, B] means 'either is fine', not 'both required' --
    using just one acceptable tool alongside an unrelated supporting step
    must not be penalized as merely half right."""
    from src.eval.metrics import score_tool_selection
    results = [{"tool": "pandas_transform"}, {"tool": "sql_query"}]
    ms = score_tool_selection(results, expected_tools=["pandas_transform", "time_series"])
    assert ms.score == 1.0


def test_tool_selection_no_steps_executed_with_expected_tools_scores_zero():
    from src.eval.metrics import score_tool_selection
    ms = score_tool_selection([], expected_tools=["sql_query"])
    assert ms.score == 0.0


def test_chart_appropriateness_no_expected_types_is_lenient():
    from src.eval.metrics import score_chart_appropriateness
    ms = score_chart_appropriateness([{"chart_type": "bar"}], expected_chart_types=None)
    assert ms.score == 0.8
    assert ms.label == "pass"


def test_chart_appropriateness_explicitly_no_chart_expected():
    from src.eval.metrics import score_chart_appropriateness
    ms = score_chart_appropriateness([], expected_chart_types=[])
    assert ms.score == 1.0


def test_chart_appropriateness_matches_expected_type():
    from src.eval.metrics import score_chart_appropriateness
    charts = [{"chart_type": "bar", "title": "Sales by Region"}]
    ms = score_chart_appropriateness(charts, expected_chart_types=["bar", "horizontal_bar"])
    assert ms.score == 1.0


def test_chart_appropriateness_wrong_type_scores_zero():
    """recommend_chart() (src/tools/chart_tool.py) is rule-based and
    always returns *some* valid type, which is why the old metric ("did
    chart generation not error") was structurally constant at 1.00. The
    rebuilt metric must catch a genuinely inappropriate chart type."""
    from src.eval.metrics import score_chart_appropriateness
    charts = [{"chart_type": "pie", "title": "Revenue Trend"}]
    ms = score_chart_appropriateness(charts, expected_chart_types=["line"])
    assert ms.score == 0.0
    assert ms.label == "fail"


def test_chart_appropriateness_no_charts_generated_but_expected():
    from src.eval.metrics import score_chart_appropriateness
    ms = score_chart_appropriateness([], expected_chart_types=["bar"])
    assert ms.score == 0.0
    assert ms.label == "fail"


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


def test_system_metrics_token_cost_reads_cost_usd_key():
    """CostTracker.to_dict() (src/utils/cost_tracker.py) writes the per-agent
    cost under "cost_usd" — reading any other key name here would silently
    sum to $0 for every run regardless of actual spend."""
    from src.eval.metrics import score_system_metrics
    state = {
        "token_usage": {
            "intent_parser": {"cost_usd": 0.02},
            "analysis_agent": {"cost_usd": 0.03},
        },
        "iteration_count": 1, "error": None,
    }
    metrics = score_system_metrics(state)
    cost_m = next(m for m in metrics if m.metric == "token_cost")
    assert cost_m.raw_value == pytest.approx(0.05)
    assert cost_m.score == pytest.approx(0.95)


# ─── 9.2 / 9.3 LLM-as-judge ─────────────────────────────────────────────────
#
# answer_relevance and groundedness are now two fully independent judge
# calls (eval v2 Step 2b) with different response shapes, so each gets its
# own test block below instead of one shared mock serving both. A shared
# score_relevance_and_groundedness() block at the end covers the
# back-compat wrapper that composes them.

# ── shared judge/human rendering (eval v2 Step 3) ────────────────────────────

def test_render_findings_includes_full_result_for_small_structured_steps():
    """The old rendering only showed result_summary -- a significance
    test's statistic/p-value lived in `result` and was invisible to both
    the judge and human annotators (docs/judge_calibration.md, case C01)."""
    from src.eval.metrics import render_findings
    results = [{
        "method": "comparison_of_sales_performance", "failed": False,
        "result_summary": "comparison: top=4",
        "result": {"significance_test": {"test": "one-way ANOVA", "p_value": None}},
    }]
    text = render_findings(results)
    assert "comparison: top=4" in text
    assert "one-way ANOVA" in text, "the structured result must be visible, not just the summary"


def test_render_findings_caps_large_results_instead_of_dumping_them():
    from src.eval.metrics import render_findings, _STEP_RESULT_CAP
    huge_result = [{"row": i} for i in range(5000)]  # a full-dataframe-sized result
    results = [{"method": "derive", "failed": False, "result_summary": "12240 rows",
                "result": huge_result}]
    text = render_findings(results)
    assert len(text) < len(repr(huge_result))
    assert "more chars truncated" in text


def test_render_findings_skips_failed_steps():
    from src.eval.metrics import render_findings
    results = [{"method": "ok_step", "failed": False, "result_summary": "fine", "result": 1},
               {"method": "bad_step", "failed": True, "result_summary": "should not appear", "result": 2}]
    text = render_findings(results)
    assert "fine" in text
    assert "should not appear" not in text


def test_render_rag_context_includes_full_content_no_truncation():
    from src.eval.metrics import render_rag_context
    long_content = "x" * 500  # longer than the old 100-char cap
    text = render_rag_context([{"content": long_content}])
    assert text == long_content


def test_render_rag_context_empty_list():
    from src.eval.metrics import render_rag_context
    assert render_rag_context([]) == ""


def test_render_data_quality_shows_flagged_issues():
    from src.eval.metrics import render_data_quality
    dq = {"row_count": 12240, "quality_issues": [
        {"issue": "duplicate_rows", "severity": "warning", "detail": "238 fully duplicated rows (1.9%)"},
    ]}
    text = render_data_quality(dq)
    assert "238" in text
    assert "12240" in text


def test_render_data_quality_none_report():
    from src.eval.metrics import render_data_quality
    assert render_data_quality(None) == "None"


def test_render_data_quality_no_issues_flagged():
    from src.eval.metrics import render_data_quality
    text = render_data_quality({"row_count": 100, "quality_issues": []})
    assert "no quality issues" in text.lower()


def test_build_judge_prompt_does_not_truncate():
    """Found via a real annotation session: the judge's old 600/1200-char
    caps cut off the exact groupby results and Automated Caveats section a
    human annotator could see in full, so the two were judging different
    material (docs/judge_calibration.md, case C01)."""
    from src.eval.metrics import _build_judge_prompt
    long_report = "A" * 5000
    findings = [{"method": "step", "failed": False, "result_summary": "B" * 2000, "result": None}]
    prompt = _build_judge_prompt("q", long_report, findings, [])
    assert long_report in prompt, "the full report must reach the judge, not a 1200-char prefix"
    assert "B" * 2000 in prompt, "the full findings must reach the judge, not a 600-char prefix"


def test_build_judge_prompt_includes_data_quality_section():
    from src.eval.metrics import _build_judge_prompt
    dq = {"row_count": 100, "quality_issues": [{"issue": "duplicate_rows", "severity": "warning", "detail": "5 dupes"}]}
    prompt = _build_judge_prompt("q", "report", [], [], data_quality_report=dq)
    assert "### Data Quality" in prompt
    assert "5 dupes" in prompt


class _FakeRateLimitError(Exception):
    """Stands in for openai.RateLimitError / anthropic.RateLimitError --
    both expose a 429 status_code, which is all
    src.utils.retry.is_rate_limit_error checks for, so a lightweight fake
    avoids depending on either SDK's exception class in this test."""
    status_code = 429


def _mock_relevance_response(score, reasoning="r"):
    resp = MagicMock()
    resp.content = json.dumps({"answer_relevance": score, "reasoning": reasoning})
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    return resp


def _mock_groundedness_response(claims, reasoning="r"):
    resp = MagicMock()
    resp.content = json.dumps({"claims": claims, "reasoning": reasoning})
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    return resp


# ── answer_relevance ─────────────────────────────────────────────────────────

def test_score_answer_relevance_with_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_relevance_response(
        0.9, "Directly answers the question"
    ))

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance(
        "Show sales by region",
        "# Report\nNorth region: $500K. South: $300K.",
        [{"result_summary": "North=500K South=300K", "failed": False}],
        [],
        llm=mock_llm,
    ))
    assert rel.metric == "answer_relevance"
    assert rel.score == 0.9
    assert rel.valid is True


def test_score_answer_relevance_fallback_on_llm_error():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm))
    assert rel.score == 0.5
    assert rel.label == "error"
    assert rel.valid is False, "an unrecoverable judge failure must not look like a real 0.5 score"


def test_score_answer_relevance_makes_n_samples_judge_calls():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_relevance_response(0.8))

    from src.eval.metrics import score_answer_relevance
    asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=5))
    assert mock_llm.ainvoke.await_count == 5


def test_score_answer_relevance_aggregates_by_median_not_mean():
    # Median of [0.2, 0.9, 0.9] is 0.9, not the mean (~0.67) — a single
    # noisy low outlier shouldn't drag the score down as much as a mean would.
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_relevance_response(0.2),
        _mock_relevance_response(0.9),
        _mock_relevance_response(0.9),
    ])

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert rel.score == 0.9


def test_score_answer_relevance_flags_high_judge_disagreement():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_relevance_response(0.1),
        _mock_relevance_response(0.9),
        _mock_relevance_response(0.5),
    ])

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert "disagreement" in rel.reasoning


def test_score_answer_relevance_missing_field_is_a_failure_not_a_05_default():
    """eval v2 Step 2a: a response missing 'answer_relevance' entirely used
    to silently become 0.5 via raw.get(key, 0.5) -- indistinguishable from
    the judge genuinely scoring 0.5. It must count as a failed sample
    instead, so 2/3 good samples still produce a real median."""
    mock_llm = MagicMock()
    ok_response = _mock_relevance_response(0.9)
    malformed_response = MagicMock()
    malformed_response.content = json.dumps({"reasoning": "forgot the score"})  # no answer_relevance key
    malformed_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(side_effect=[ok_response, malformed_response, ok_response])

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert rel.score == 0.9
    assert rel.valid is True


def test_answer_relevance_retries_on_rate_limit_then_succeeds(monkeypatch):
    """Found running scripts/measure_noise.py against a real (rate-limited)
    API account: a single transient 429 used to propagate straight through
    to the judge's fallback, silently degrading the result to 0.5 --
    indistinguishable from the judge genuinely scoring something mediocre.
    It must retry instead."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _FakeRateLimitError("429 rate limit exceeded"),
        _mock_relevance_response(0.9),
    ])
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", AsyncMock())

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=1))
    assert rel.score == 0.9
    assert mock_llm.ainvoke.await_count == 2  # 1 rate-limited attempt + 1 retry that succeeded


def test_answer_relevance_gives_up_after_max_retries_and_falls_back(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=_FakeRateLimitError("429 rate limit exceeded"))
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", AsyncMock())

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=1))
    assert rel.score == 0.5
    assert rel.valid is False
    assert "LLM unavailable" in rel.reasoning


def test_answer_relevance_non_rate_limit_error_is_not_retried(monkeypatch):
    """A permanently-broken response (malformed JSON, etc.) shouldn't spin
    through retries meant for transient rate limits -- it should fail fast."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("malformed response"))
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", sleep_mock)

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=1))
    assert rel.score == 0.5
    assert mock_llm.ainvoke.await_count == 1  # no retry attempted
    sleep_mock.assert_not_awaited()


def test_answer_relevance_partial_sample_failure_scores_from_survivors_not_fallback():
    """One of three samples fails permanently (not rate-limited, so not
    retried) -- previously this discarded the whole batch via a bare
    asyncio.gather; the two surviving samples must still produce a real
    score instead of collapsing to the 0.5 fallback."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_relevance_response(0.9),
        RuntimeError("malformed response"),
        _mock_relevance_response(0.9),
    ])

    from src.eval.metrics import score_answer_relevance
    rel = asyncio.run(score_answer_relevance("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert rel.score == 0.9
    assert rel.valid is True


# ── groundedness ─────────────────────────────────────────────────────────────

def test_score_groundedness_computes_score_from_claim_list():
    """eval v2 Step 2c: groundedness is supported_count/total_claims,
    computed in code -- not a bare 0-1 number asked from the judge."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_groundedness_response([
        {"claim": "North region revenue $1,363,760.55", "supported": True, "evidence": "findings"},
        {"claim": "growth driven by a new product line", "supported": False, "evidence": None},
    ]))

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm))
    assert gnd.metric == "groundedness"
    assert gnd.score == 0.5  # 1/2 claims supported
    assert gnd.valid is True


def test_score_groundedness_all_claims_supported_scores_one():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_groundedness_response([
        {"claim": "a", "supported": True, "evidence": "x"},
        {"claim": "b", "supported": True, "evidence": "y"},
    ]))

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm))
    assert gnd.score == 1.0


def test_score_groundedness_no_checkable_claims_is_lenient_not_zero():
    """No claims to check is neither grounded nor ungrounded -- same
    lenient-default convention as score_factual_accuracy's 'no numbers to
    cross-check' case, not a penalty."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_groundedness_response([]))

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm))
    assert gnd.score == 0.8
    assert gnd.valid is True


def test_score_groundedness_reasoning_is_the_claim_list_not_generic_prose():
    """The old shared-judge design let groundedness's reasoning describe
    relevance instead (verified 20/20 cases identical in the audit that
    started this rewrite) -- the claim list itself must now be what's
    reported, so score and reasoning can't drift apart."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=_mock_groundedness_response([
        {"claim": "North region revenue is $1,363,760.55", "supported": True, "evidence": "findings: North=1363760.55"},
        {"claim": "driven by a new premium product line", "supported": False, "evidence": None},
    ]))

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm))
    assert "North region revenue" in gnd.reasoning
    assert "premium product line" in gnd.reasoning
    assert "1/2" in gnd.reasoning


def test_score_groundedness_fallback_on_llm_error():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm))
    assert gnd.score == 0.5
    assert gnd.label == "error"
    assert gnd.valid is False


def test_score_groundedness_missing_claims_field_is_a_failure():
    """eval v2 Step 2a: a response missing 'claims' entirely must count as
    a failed sample, not silently become an empty (lenient) claim list."""
    mock_llm = MagicMock()
    malformed = MagicMock()
    malformed.content = json.dumps({"reasoning": "forgot the claims list"})
    malformed.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),
        malformed,
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),
    ])

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert gnd.score == 1.0  # median of the 2 successful samples, malformed one excluded
    assert gnd.valid is True


def test_score_groundedness_aggregates_by_median_across_samples():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_groundedness_response([{"claim": "a", "supported": False, "evidence": None}]),  # 0.0
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),      # 1.0
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),      # 1.0
    ])

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert gnd.score == 1.0  # median of [0.0, 1.0, 1.0]


def test_groundedness_retries_on_rate_limit_then_succeeds(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _FakeRateLimitError("429 rate limit exceeded"),
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),
    ])
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", AsyncMock())

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm, n_samples=1))
    assert gnd.score == 1.0
    assert mock_llm.ainvoke.await_count == 2


def test_groundedness_partial_sample_failure_scores_from_survivors_not_fallback():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),
        RuntimeError("malformed response"),
        _mock_groundedness_response([{"claim": "a", "supported": True, "evidence": "x"}]),
    ])

    from src.eval.metrics import score_groundedness
    gnd = asyncio.run(score_groundedness("q", "report", [], [], llm=mock_llm, n_samples=3))
    assert gnd.score == 1.0
    assert gnd.valid is True


# ── back-compat wrapper: score_relevance_and_groundedness ───────────────────

def _routing_mock_llm(relevance_response, groundedness_response):
    """A single mock LLM that returns different canned responses depending
    on which of the two independent system prompts it was called with --
    lets the wrapper test prove the two calls are genuinely independent
    (different prompt, different response shape) rather than assuming it."""
    from src.config.agent_prompts import EVAL_GROUNDEDNESS_SYSTEM, EVAL_RELEVANCE_SYSTEM

    async def _ainvoke(messages):
        system_content = messages[0].content
        if system_content == EVAL_RELEVANCE_SYSTEM:
            return relevance_response
        if system_content == EVAL_GROUNDEDNESS_SYSTEM:
            return groundedness_response
        raise AssertionError(f"unexpected system prompt: {system_content!r}")

    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=_ainvoke)
    return mock


def test_score_relevance_and_groundedness_composes_two_independent_calls():
    mock_llm = _routing_mock_llm(
        _mock_relevance_response(0.9, "answers the question directly"),
        _mock_groundedness_response(
            [{"claim": "a", "supported": True, "evidence": "x"},
             {"claim": "b", "supported": False, "evidence": None}],
            "unused -- reasoning is rendered from the claim list",
        ),
    )

    from src.eval.metrics import score_relevance_and_groundedness
    rel, gnd = asyncio.run(score_relevance_and_groundedness(
        "q", "report", [], [], llm=mock_llm, n_samples=1,
    ))
    assert rel.metric == "answer_relevance"
    assert rel.score == 0.9
    assert gnd.metric == "groundedness"
    assert gnd.score == 0.5
    assert rel.reasoning != gnd.reasoning, (
        "the two scores must never share one reasoning string again -- "
        "that was the original bug (20/20 cases identical)"
    )


# ─── 9.1 EvalRunner ──────────────────────────────────────────────────────────

def test_eval_runner_scores_state():
    mock_llm = _routing_mock_llm(
        _mock_relevance_response(0.88, "Good"),
        _mock_groundedness_response([{"claim": "North=500K", "supported": True, "evidence": "findings"}]),
    )

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
    mock_llm = _routing_mock_llm(
        _mock_relevance_response(0.9, "Perfect"),
        _mock_groundedness_response([{"claim": "North leads", "supported": True, "evidence": "findings"}]),
    )

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


def test_invalid_metric_excluded_from_aggregate_score():
    """eval v2 Step 2a: a metric that failed to score (judge unreachable)
    must not be averaged in as a fake 0.5 placeholder -- it should be
    excluded from the weighted average entirely, same treatment as
    safe_refusal but for a different reason (no real measurement exists,
    rather than the metric being intentionally informational-only)."""
    from src.eval.runner import _aggregate_score
    from src.eval.metrics import MetricScore

    base_scores = [
        MetricScore("answer_relevance", 0.9, "pass"),
        MetricScore("factual_accuracy", 1.0, "pass"),
    ]
    without_failure = _aggregate_score(base_scores)
    with_failed_groundedness = _aggregate_score(
        base_scores + [MetricScore("groundedness", 0.5, "error", "LLM unavailable", valid=False)]
    )
    assert with_failed_groundedness == without_failure, (
        "a valid=False placeholder score must not move the aggregate at all"
    )


# ─── 9.6 Golden test suite ───────────────────────────────────────────────────

def test_golden_suite_has_20_cases():
    from src.eval.runner import load_golden_suite
    suite = load_golden_suite()
    assert len(suite) >= 20


def test_golden_suite_covers_all_query_types():
    from src.eval.runner import load_golden_suite
    suite = load_golden_suite()
    types = {tc.query_type for tc in suite}
    assert "descriptive" in types
    assert "diagnostic" in types
    assert "comparative" in types
    assert "predictive" in types
    assert "exploratory" in types


def test_load_golden_suite_fails_loudly_if_json_missing():
    """Eval v2 Step 4: the old silent fallback to a hardcoded Python copy
    is gone -- a missing/misconfigured suite file must raise, not quietly
    substitute a copy that can drift out of sync unnoticed."""
    from src.eval.runner import load_golden_suite
    with pytest.raises(FileNotFoundError):
        load_golden_suite(path="tests/eval/does_not_exist.json")


def test_golden_suite_data_mismatch_cases_are_tagged():
    """Eval v2 Step 4: KNOWN_DATA_MISMATCH (a hardcoded set in run_eval.py)
    was replaced by a "data_mismatch" tag authored directly on each case."""
    from src.eval.runner import load_golden_suite
    suite = {tc.id: tc for tc in load_golden_suite()}
    for cid in ("DG04", "C03", "P03"):
        assert "data_mismatch" in suite[cid].tags, f"{cid} should carry the data_mismatch tag"


# Cases where the query asks for something the demo datasets don't contain,
# or asks about the future — these carry a "_note" instead of a checkable
# numeric ground truth. Every other case must have a real, non-empty
# ground_truth backed by an actual computation over data/demo/*.
# D02 was here too until Phase B #1/#2 (cross-table joins) made it fully
# answerable via a real orders-JOIN-products query — see docs/eval_report.md.
_KNOWN_UNANSWERABLE_CASES = {
    "DG04", "C03", "P01", "P02", "P03",
    # Eval v2 Step 4: 20 new cases added alongside the original 20 (see
    # docs/eval_v2_plan.md Step 4) -- same two "no checkable ground truth"
    # reasons as the originals: genuinely predictive (P04, P05, P07),
    # absent data (P06), and a deliberately ambiguous query with no
    # metric/dimension to check at all (E05).
    "P04", "P05", "P06", "P07", "E05",
    # Eval v2 Step 4, second batch of 20 (2026-07-31): same reasons again --
    # genuinely predictive (P08, P09, P11), absent data (DG12), ambiguous (E13).
    "P08", "P09", "P11", "DG12", "E13",
    # Eval v2 Step 4, third batch of 20 (2026-08-01): same reasons again --
    # genuinely predictive (P12, P13, P14), absent data (DG16), ambiguous (E19).
    "P12", "P13", "P14", "DG16", "E19",
    # Eval v2 Step 4, fourth batch of 20 (2026-08-01), suite now 100 cases --
    # genuinely predictive (P15, P16, P17), absent data (DG20). No new
    # ambiguous case this batch (3 already in the suite).
    "P15", "P16", "P17", "DG20",
}


def test_golden_suite_ground_truth_backfilled_from_json():
    from src.eval.runner import load_golden_suite
    suite = load_golden_suite()
    for tc in suite:
        if tc.id in _KNOWN_UNANSWERABLE_CASES:
            assert "_note" in tc.ground_truth, f"{tc.id} should document why it has no ground truth"
        else:
            numeric_values = {k: v for k, v in tc.ground_truth.items() if isinstance(v, (int, float))}
            assert numeric_values, f"{tc.id} should have at least one numeric ground_truth fact"


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
        result = asyncio.run(run_eval_node(state))
        _nodes._eval_runner = None

    assert "eval_scores" in result
    assert "_aggregate" in result["eval_scores"]
    assert result["current_phase"] == "complete"
    assert result["eval_scores"]["_aggregate"] > 0.0


# ─── _parse_json ─────────────────────────────────────────────────────────────

def test_parse_json_extracts_object():
    from src.eval.metrics import _parse_json
    assert _parse_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_parse_json_strips_markdown_fences():
    from src.eval.metrics import _parse_json
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_no_longer_falls_back_to_a_bare_list():
    """Found live (eval v2 Step 2c): a truncated groundedness response
    (claims list cut off mid-response by max_tokens) could make the
    top-level object parse fail and an old `[...]` fallback then
    successfully parse just the claims array as a bare list -- which
    crashed callers expecting a dict with "'list' object has no attribute
    'get'" instead of a clean, catchable parse error. Every prompt in this
    module asks for a top-level object; a response with no object at all
    (no curly braces whatsoever) must fail the same clear way truncation
    does, not silently fall through to a list-shaped result."""
    from src.eval.metrics import _parse_json
    with pytest.raises(ValueError):
        _parse_json('[1, 2, 3]')


def test_parse_json_raises_on_truncated_response():
    from src.eval.metrics import _parse_json
    truncated = '{"claims": [{"claim": "a", "supported": true, "evidence": "part'
    with pytest.raises(ValueError):
        _parse_json(truncated)


def test_parse_json_raises_on_garbage():
    from src.eval.metrics import _parse_json
    with pytest.raises(ValueError):
        _parse_json("not json at all")
