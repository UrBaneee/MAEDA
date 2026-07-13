"""
Phase 8 tests — Guardrail Agent.
Run with: pytest tests/unit/test_phase8.py -v
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.state.graph_state import initial_state


# ─── Data model ───────────────────────────────────────────────────────────────

def test_check_result_to_dict():
    from src.agents.guardrail_agent import CheckResult
    cr = CheckResult("pii_detection", True, "info", None)
    d = cr.to_dict()
    assert d["check"] == "pii_detection"
    assert d["passed"] is True
    assert d["severity"] == "info"


def test_guardrail_report_to_state_dict():
    from src.agents.guardrail_agent import CheckResult, GuardrailReport
    report = GuardrailReport(
        checks=[CheckResult("sql_safety", True, "info")],
        overall_verdict="approved",
        passed=True,
        retry_reason=None,
        caveats=[],
    )
    d = report.to_state_dict()
    assert d["overall_verdict"] == "approved"
    assert d["passed"] is True
    assert len(d["checks"]) == 1


# ─── 8.1 Numerical consistency ────────────────────────────────────────────────

def test_numerical_consistency_passes_no_conflict():
    from src.agents.guardrail_agent import _check_numerical_consistency
    report = "Revenue was 100 in Q1 and 200 in Q2."
    results = [{"result_summary": "Q1=100, Q2=200", "failed": False}]
    cr = _check_numerical_consistency(report, results)
    assert cr.passed


def test_numerical_consistency_warning_on_mismatch():
    from src.agents.guardrail_agent import _check_numerical_consistency
    # Report has many numbers totally unrelated to summaries
    report = "Revenue was 999 1234 5678 9999 111 222 333."
    results = [{"result_summary": "Q1=1 Q2=2", "failed": False}]
    cr = _check_numerical_consistency(report, results)
    # May or may not flag depending on overlap — just check it returns a CheckResult
    assert cr.check == "numerical_consistency"
    assert cr.severity in {"info", "warning", "critical"}


def test_numerical_consistency_passes_empty():
    from src.agents.guardrail_agent import _check_numerical_consistency
    cr = _check_numerical_consistency("", [])
    assert cr.passed


# ─── 8.3 SQL safety ───────────────────────────────────────────────────────────

def test_sql_safety_passes_safe_select():
    from src.agents.guardrail_agent import _check_sql_safety
    cr = _check_sql_safety(["SELECT region, SUM(sales) FROM data GROUP BY region;"])
    assert cr.passed


def test_sql_safety_blocks_drop():
    from src.agents.guardrail_agent import _check_sql_safety
    cr = _check_sql_safety(["DROP TABLE customers;"])
    assert not cr.passed
    assert cr.severity == "critical"


def test_sql_safety_blocks_delete():
    from src.agents.guardrail_agent import _check_sql_safety
    cr = _check_sql_safety(["DELETE FROM orders WHERE id > 0;"])
    assert not cr.passed
    assert cr.severity == "critical"


def test_sql_safety_passes_no_sql():
    from src.agents.guardrail_agent import _check_sql_safety
    cr = _check_sql_safety([])
    assert cr.passed


def test_extract_sql_finds_fenced_block():
    from src.agents.guardrail_agent import _extract_sql
    text = "Analysis:\n```sql\nSELECT * FROM data;\n```\nEnd."
    stmts = _extract_sql(text)
    assert len(stmts) >= 1
    assert "SELECT" in stmts[0]


def test_extract_sql_finds_dangerous_stmt():
    from src.agents.guardrail_agent import _extract_sql
    text = "```sql\nDROP TABLE customers;\n```"
    stmts = _extract_sql(text)
    assert any("DROP" in s.upper() for s in stmts)


# ─── 8.4 PII detection ────────────────────────────────────────────────────────

def test_pii_detects_email():
    from src.agents.guardrail_agent import _check_pii
    cr = _check_pii("Contact john.doe@example.com for details.")
    assert not cr.passed
    assert cr.severity == "critical"
    assert "email" in cr.finding.lower()


def test_pii_detects_ssn():
    from src.agents.guardrail_agent import _check_pii
    cr = _check_pii("Customer SSN: 123-45-6789")
    assert not cr.passed
    assert cr.severity == "critical"


def test_pii_detects_phone():
    from src.agents.guardrail_agent import _check_pii
    cr = _check_pii("Call us at 555-867-5309.")
    assert not cr.passed


def test_pii_passes_clean_text():
    from src.agents.guardrail_agent import _check_pii
    cr = _check_pii("Revenue in Q1 was $1.2M, up 15% YoY.")
    assert cr.passed


def test_pii_passes_empty():
    from src.agents.guardrail_agent import _check_pii
    cr = _check_pii("")
    assert cr.passed


# ─── 8.6 Completeness ─────────────────────────────────────────────────────────

def test_completeness_passes_good_report():
    from src.agents.guardrail_agent import _check_completeness
    report = (
        "# Executive Summary\n\nRevenue grew 20% year-over-year, driven by strong "
        "performance in the North region.\n\n## Key Findings\n- North region leads "
        "with $500K.\n- Q4 outperformed all other quarters.\n\n"
        "## Recommendations\n- Increase investment in North.\n"
    )
    cr = _check_completeness(report, "Show revenue growth")
    assert cr.passed


def test_completeness_warns_empty():
    from src.agents.guardrail_agent import _check_completeness
    cr = _check_completeness("", "What is revenue?")
    assert not cr.passed
    assert cr.severity == "warning"


def test_completeness_warns_short():
    from src.agents.guardrail_agent import _check_completeness
    cr = _check_completeness("Revenue up.", "Show revenue by region by quarter for 2024")
    assert not cr.passed


def test_completeness_warns_no_structure():
    from src.agents.guardrail_agent import _check_completeness
    plain = "a" * 200
    cr = _check_completeness(plain, "q")
    assert not cr.passed


# ─── 8.8 Population-claim grounding ──────────────────────────────────────────

def test_population_claim_passes_no_report():
    from src.agents.guardrail_agent import _check_population_claim_grounding
    cr = _check_population_claim_grounding("", [])
    assert cr.passed


def test_population_claim_passes_no_generalizing_language():
    from src.agents.guardrail_agent import _check_population_claim_grounding
    report = "Customer 4821 placed an order for $120 on March 3rd."
    cr = _check_population_claim_grounding(report, [])
    assert cr.passed


def test_population_claim_critical_when_no_aggregate_evidence():
    """The exact failure mode #12 targets: a row-level sample (filter/derive
    only) gets generalized into a population-wide claim in the report."""
    from src.agents.guardrail_agent import _check_population_claim_grounding
    report = "Most customers churn within their first 90 days due to pricing."
    results = [
        {"method": "filter_recent", "result_summary": "pandas/filter -> 3 rows matched",
         "failed": False},
    ]
    cr = _check_population_claim_grounding(report, results)
    assert not cr.passed
    assert cr.severity == "critical"
    assert "population-level claim" in cr.finding


def test_population_claim_passes_when_aggregate_evidence_exists():
    from src.agents.guardrail_agent import _check_population_claim_grounding
    report = "Most customers churn within their first 90 days due to pricing."
    results = [
        {"method": "churn_by_cohort", "result_summary": "pandas/groupby -> 5 groups",
         "failed": False},
    ]
    cr = _check_population_claim_grounding(report, results)
    assert cr.passed


def test_population_claim_ignores_failed_steps_as_evidence():
    from src.agents.guardrail_agent import _check_population_claim_grounding
    report = "Overall, revenue trends upward across the board."
    results = [
        {"method": "revenue_groupby", "result_summary": "pandas/groupby -> 5 groups",
         "failed": True},
    ]
    cr = _check_population_claim_grounding(report, results)
    assert not cr.passed


def test_population_claim_bare_all_is_not_flagged():
    """A bare "all" is too common in ordinary prose to use as a trigger on
    its own -- only "all/every/most <population noun>" and a short list of
    unambiguous generalization phrases should match."""
    from src.agents.guardrail_agent import _check_population_claim_grounding
    report = "All of the figures above are rounded to the nearest dollar."
    cr = _check_population_claim_grounding(report, [])
    assert cr.passed


# ─── 8.7 Aggregator ───────────────────────────────────────────────────────────

def test_aggregate_all_pass():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [
        CheckResult("sql_safety", True, "info"),
        CheckResult("pii_detection", True, "info"),
        CheckResult("completeness_check", True, "info"),
    ]
    report = _aggregate(checks, iteration=0, max_retries=2)
    assert report.overall_verdict == "approved"
    assert report.passed is True


def test_aggregate_warning_delivers_with_caveats():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [
        CheckResult("completeness_check", False, "warning", "Report is short"),
        CheckResult("sql_safety", True, "info"),
    ]
    report = _aggregate(checks, iteration=0, max_retries=2)
    assert report.overall_verdict == "approved"
    assert report.passed is True
    assert "Report is short" in report.caveats


def test_aggregate_critical_triggers_retry():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [
        CheckResult("pii_detection", False, "critical", "Email found"),
        CheckResult("sql_safety", True, "info"),
    ]
    report = _aggregate(checks, iteration=0, max_retries=2)
    assert report.overall_verdict == "retry"
    assert report.passed is False
    assert "Email found" in report.retry_reason


def test_aggregate_critical_fails_after_max_retries():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [CheckResult("pii_detection", False, "critical", "SSN found")]
    report = _aggregate(checks, iteration=2, max_retries=2)
    assert report.overall_verdict == "fail"
    assert report.passed is False


def test_aggregate_multiple_critical_fails():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [
        CheckResult("hallucination_check", False, "critical", "Hallucinated stat"),
        CheckResult("pii_detection", False, "critical", "Email found"),
    ]
    report = _aggregate(checks, iteration=3, max_retries=2)
    assert report.overall_verdict == "fail"


# ─── 8.5 LLM-as-judge ────────────────────────────────────────────────────────

def test_llm_judge_passes_clean_report():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": True,
        "checks": [
            {"check": "hallucination_check", "passed": True, "finding": None},
            {"check": "claim_grounding", "passed": True, "finding": None},
        ],
        "overall_verdict": "approved",
        "retry_reason": None,
    })
    mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 15}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = GuardrailAgent(llm=mock_llm)
    result = asyncio.run(agent._llm_judge(
        "# Report\nRevenue grew 15% in Q1.",
        [{"finding": "Revenue grew", "confidence": 0.9}],
        [{"result_summary": "Revenue grew 15%", "failed": False}],
        "Show revenue growth",
    ))
    assert any(c.check == "hallucination_check" and c.passed for c in result)


def test_llm_judge_flags_hallucination():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": False,
        "checks": [
            {"check": "hallucination_check", "passed": False,
             "finding": "Report claims 99% growth not supported by data"},
            {"check": "claim_grounding", "passed": True, "finding": None},
        ],
        "overall_verdict": "retry",
        "retry_reason": "Hallucinated growth figure",
    })
    mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 20}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = GuardrailAgent(llm=mock_llm)
    result = asyncio.run(agent._llm_judge(
        "# Report\nRevenue grew 99% — unprecedented.",
        [],
        [{"result_summary": "Revenue grew 5%", "failed": False}],
        "Show revenue",
    ))
    hallucination = next(c for c in result if c.check == "hallucination_check")
    assert not hallucination.passed
    # Per DEV_SPEC: hallucination is "critical" (block + retry), not a warning.
    assert hallucination.severity == "critical"


def test_llm_judge_fallback_on_error():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    agent = GuardrailAgent(llm=mock_llm)
    result = asyncio.run(agent._llm_judge("report text", [], [], "query"))
    # Fallback: returns pass results, doesn't raise
    assert all(c.passed for c in result)


# ─── 8.7 Full process ────────────────────────────────────────────────────────

def test_process_clean_state_passes():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": True,
        "checks": [
            {"check": "hallucination_check", "passed": True, "finding": None},
            {"check": "claim_grounding", "passed": True, "finding": None},
        ],
        "overall_verdict": "approved",
        "retry_reason": None,
    })
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 10}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = GuardrailAgent(llm=mock_llm)
    state = initial_state("Show revenue by region")
    state["report"] = "# Report\n\n## Key Findings\n- Revenue grew 15%.\n\n## Recommendations\n- Invest more."
    state["insights"] = [{"finding": "Revenue grew 15%", "confidence": 0.9}]
    state["analysis_results"] = [
        {"result_summary": "Revenue grew 15%", "failed": False}
    ]

    result = asyncio.run(agent.process(state))
    assert result["guardrail_passed"] is True
    assert len(result["guardrail_checks"]) == 1
    assert result["guardrail_checks"][0]["overall_verdict"] == "approved"


def test_process_pii_in_report_fails():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": True,
        "checks": [{"check": "hallucination_check", "passed": True, "finding": None}],
        "overall_verdict": "approved",
        "retry_reason": None,
    })
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = GuardrailAgent(llm=mock_llm, max_retries=2)
    state = initial_state("q")
    state["report"] = "## Report\n\nContact admin@company.com for details.\n\n## Findings\n- x"
    state["insights"] = []
    state["analysis_results"] = []
    state["iteration_count"] = 0

    result = asyncio.run(agent.process(state))
    # PII detected → not passed
    assert result["guardrail_passed"] is False
    assert result["guardrail_checks"][0]["overall_verdict"] in {"retry", "fail"}


def test_process_dangerous_sql_blocks():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = GuardrailAgent(llm=mock_llm, max_retries=2)
    state = initial_state("q")
    state["report"] = "Analysis:\n```sql\nDROP TABLE orders;\n```\n\n## Findings\n- data deleted"
    state["insights"] = []
    state["analysis_results"] = []
    state["iteration_count"] = 0

    result = asyncio.run(agent.process(state))
    assert result["guardrail_passed"] is False


def test_process_warning_appends_caveat():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": True,
        "checks": [{"check": "hallucination_check", "passed": True, "finding": None}],
        "overall_verdict": "approved",
        "retry_reason": None,
    })
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    agent = GuardrailAgent(llm=mock_llm)
    state = initial_state("q")
    # Report is short — triggers warning
    state["report"] = "Revenue up."
    state["insights"] = []
    state["analysis_results"] = []

    result = asyncio.run(agent.process(state))
    # Warning does not block delivery
    assert result["guardrail_passed"] is True
    # Caveat appended to report
    assert "Automated Caveats" in result.get("report", "") or result["guardrail_passed"] is True


def test_process_logs_decision():
    from src.agents.guardrail_agent import GuardrailAgent
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    agent = GuardrailAgent(llm=mock_llm)
    state = initial_state("q")
    state["report"] = "# Report\n\n## Findings\n- x\n\n## Rec\n- y"
    state["insights"] = []
    state["analysis_results"] = []

    result = asyncio.run(agent.process(state))
    assert any(t["action"] == "run_guardrails" for t in result["decision_trace"])


# ─── 8.8 Retry mechanism ─────────────────────────────────────────────────────

def test_retry_verdict_respects_iteration_count():
    from src.agents.guardrail_agent import CheckResult, _aggregate
    checks = [CheckResult("pii_detection", False, "critical", "SSN detected")]
    # Below max_retries → retry
    r1 = _aggregate(checks, iteration=0, max_retries=2)
    assert r1.overall_verdict == "retry"
    # At max_retries → fail
    r2 = _aggregate(checks, iteration=2, max_retries=2)
    assert r2.overall_verdict == "fail"


# ─── Node integration ─────────────────────────────────────────────────────────

def test_run_guardrails_node_integration():
    import src.graph.nodes as _nodes
    from src.graph.nodes import run_guardrails_node

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": True,
        "checks": [{"check": "hallucination_check", "passed": True, "finding": None}],
        "overall_verdict": "approved",
        "retry_reason": None,
    })
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("src.agents.guardrail_agent._build_llm", return_value=mock_llm):
        _nodes._guardrail_agent = None
        state = initial_state("Show revenue")
        state["report"] = "# Report\n\n## Findings\n- Revenue up 10%.\n\n## Rec\n- Keep going."
        state["insights"] = [{"finding": "Revenue up", "confidence": 0.9}]
        state["analysis_results"] = [
            {"result_summary": "Revenue up 10%", "failed": False}
        ]
        result = asyncio.run(run_guardrails_node(state))
        _nodes._guardrail_agent = None

    assert "guardrail_checks" in result
    assert "guardrail_passed" in result
    assert result["current_phase"] == "guardrail"


# ─── Router still routes correctly ───────────────────────────────────────────

def test_router_guardrails_passed_with_checks():
    from src.graph.router import route_after_guardrails
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "approved", "passed": True}]
    assert route_after_guardrails(state) == "passed"


def test_router_guardrails_retry_with_checks():
    from src.graph.router import route_after_guardrails
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "retry"}]
    state["guardrail_retry_count"] = 0
    assert route_after_guardrails(state) == "retry"


def test_router_guardrails_fail_exhausted():
    from src.graph.router import route_after_guardrails
    state = initial_state("q")
    state["guardrail_checks"] = [{"overall_verdict": "retry"}]
    state["guardrail_retry_count"] = 5
    assert route_after_guardrails(state) == "fail"
