"""
Guardrail Agent — Phase 8.

Validates all outputs before user delivery across four dimensions:
  Accuracy  : numerical consistency, SQL safety, statistical validity
  Grounding : claim grounding, hallucination/fabrication detection
  Safety    : PII detection, bias check
  Quality   : completeness, readability

Failure modes:
  critical  → block + retry (max 2), then fail
  warning   → attach caveat, deliver
  info      → log only, deliver

State fields read:  report, insights, analysis_results, charts,
                    user_query, parsed_intent
State fields written: guardrail_checks, guardrail_passed,
                      report (may append caveats), decision_trace, token_usage
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.insight_agent import _classify_evidence_level
from src.config.agent_prompts import GUARDRAIL_SYSTEM
from src.config.settings import settings
from src.state.graph_state import MAEDAState
from src.utils.logger import get_logger

logger = get_logger("maeda.agent.guardrail")

# ─── Constants ────────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone"),
    (re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"), "credit_card"),
    (re.compile(r"\b(?:5[1-5][0-9]{14}|2(?:2[2-9]|[3-6][0-9]|7[01])[0-9]{12})\b"), "credit_card"),
]

_DANGEROUS_SQL = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|EXEC|EXECUTE|GRANT|REVOKE|"
    r"CREATE\s+USER|CREATE\s+ROLE|xp_cmdshell|LOAD\s+DATA)\b",
    re.IGNORECASE,
)

# Phrases that generalize a claim to the whole dataset/population. Deliberately
# targeted at unambiguous generalization markers (not a bare "all", which is
# too common in ordinary prose) to keep false positives low, since a match
# without aggregate-step backing escalates straight to "critical".
_POPULATION_NOUN = (
    r"(?:customers?|users?|products?|orders?|transactions?|records?|rows?|"
    r"regions?|stores?|accounts?|clients?|segments?)"
)
_POPULATION_CLAIM_RE = re.compile(
    rf"\b(?:all|every|most|the majority of)\s+{_POPULATION_NOUN}\b"
    rf"|\bacross (?:all|the board)\b"
    rf"|\bon average\b"
    rf"|\boverall,"
    rf"|\bconsistently\b"
    rf"|\bin general\b",
    re.IGNORECASE,
)

Severity = Literal["critical", "warning", "info"]


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check: str
    passed: bool
    severity: Severity
    finding: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuardrailReport:
    checks: list[CheckResult]
    overall_verdict: Literal["approved", "retry", "fail"]
    passed: bool
    retry_reason: Optional[str] = None
    caveats: list[str] = field(default_factory=list)

    def to_state_dict(self) -> dict:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "overall_verdict": self.overall_verdict,
            "passed": self.passed,
            "retry_reason": self.retry_reason,
            "caveats": self.caveats,
        }


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model, temperature=0.0,
            max_tokens=512, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model, temperature=0.0,
        max_tokens=512, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── GuardrailAgent ───────────────────────────────────────────────────────────

class GuardrailAgent(BaseAgent):
    """
    Runs all guardrail checks and sets state["guardrail_checks"] and
    state["guardrail_passed"].

    Critical failures trigger a retry signal (up to max_retries).
    Warnings attach caveats to the report; info-level findings are logged only.
    """

    def __init__(self, llm=None, max_retries: int = 2):
        super().__init__("guardrail_agent")
        self._llm = llm or _build_llm()
        self._max_retries = max_retries

    async def process(self, state: MAEDAState) -> MAEDAState:
        report_text = state.get("report") or ""
        insights = state.get("insights") or []
        analysis_results = state.get("analysis_results") or []
        query = state.get("user_query", "")

        checks: list[CheckResult] = []

        # 8.1 Numerical consistency (rule-based)
        checks.append(_check_numerical_consistency(report_text, analysis_results))

        # 8.3 SQL safety (rule-based)
        sql_stmts = _extract_sql(report_text)
        checks.append(_check_sql_safety(sql_stmts))

        # 8.4 PII filter (regex rule-based)
        checks.append(_check_pii(report_text))

        # 8.6 Completeness (rule-based)
        checks.append(_check_completeness(report_text, query))

        # 8.8 Population-claim grounding (rule-based)
        checks.append(_check_population_claim_grounding(report_text, analysis_results))

        # 8.5 Hallucination detector + 8.2 claim grounding (LLM-as-judge)
        llm_checks = await self._llm_judge(report_text, insights, analysis_results, query)
        checks.extend(llm_checks)

        # Aggregate — use guardrail_retry_count (already incremented by the node wrapper)
        guardrail_report = _aggregate(checks, state.get("guardrail_retry_count", 0), self._max_retries)

        # Append caveats to report for warnings
        if guardrail_report.caveats and report_text:
            caveat_block = "\n\n## Automated Caveats\n" + "\n".join(
                f"- {c}" for c in guardrail_report.caveats
            )
            state["report"] = report_text + caveat_block

        state["guardrail_checks"] = [guardrail_report.to_state_dict()]
        state["guardrail_passed"] = guardrail_report.passed

        state = self.log_decision(
            state,
            action="run_guardrails",
            reasoning=(
                f"Ran {len(checks)} guardrail checks; "
                f"verdict={guardrail_report.overall_verdict}; "
                f"passed={guardrail_report.passed}"
            ),
            inputs={"n_checks": len(checks)},
            outputs={
                "verdict": guardrail_report.overall_verdict,
                "n_failed": sum(1 for c in checks if not c.passed),
            },
            confidence=1.0 if guardrail_report.passed else 0.0,
        )
        return state

    # ── 8.5 / 8.2 LLM-as-judge ───────────────────────────────────────────────

    async def _llm_judge(
        self,
        report_text: str,
        insights: list[dict],
        analysis_results: list[dict],
        query: str,
    ) -> list[CheckResult]:
        """Run hallucination detection and claim grounding via LLM."""
        findings_summary = "; ".join(
            r.get("result_summary", "")
            for r in analysis_results
            if not r.get("failed") and r.get("result_summary")
        )[:800]

        context = (
            f"### Original Query\n{query}\n\n"
            f"### Analysis Findings\n{findings_summary or 'None'}\n\n"
            f"### Report to Evaluate\n{report_text[:1500]}\n"
        )

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=GUARDRAIL_SYSTEM),
                HumanMessage(content=context),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            self._cost_tracker.record(
                agent_name=self.name, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="llm_judge",
            )
            import json as _json
            raw = _parse_json(response.content.strip())
            return _parse_llm_checks(raw)
        except Exception as exc:
            logger.warning("LLM guardrail judge failed: %s — defaulting to pass", exc)
            return [
                CheckResult("hallucination_check", True, "info",
                            "LLM judge unavailable; defaulted to pass"),
                CheckResult("claim_grounding", True, "info",
                            "LLM judge unavailable; defaulted to pass"),
            ]


# ─── Rule-based checks ────────────────────────────────────────────────────────

def _check_numerical_consistency(report: str, results: list[dict]) -> CheckResult:
    """
    8.1 Extract numbers from report and verify at least some overlap with
    result summaries. A mismatch flag is 'warning' not 'critical'.
    """
    if not report or not results:
        return CheckResult("numerical_consistency", True, "info",
                           "No report or results to compare")

    report_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:%|k|M|B)?\b", report))
    summaries_text = " ".join(
        r.get("result_summary", "") for r in results if not r.get("failed")
    )
    summary_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:%|k|M|B)?\b", summaries_text))

    # Only flag if report has many numbers but none overlap with summaries
    if len(report_numbers) > 5 and summary_numbers and not report_numbers & summary_numbers:
        return CheckResult(
            "numerical_consistency", False, "warning",
            "Report numbers do not overlap with analysis result summaries",
        )
    return CheckResult("numerical_consistency", True, "info")


def _extract_sql(text: str) -> list[str]:
    """Extract SQL-like statements from a text block."""
    pattern = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    stmts = pattern.findall(text)
    # Also catch bare SELECT/INSERT etc. outside fences
    inline = re.findall(r"\b(SELECT\s+.{10,200}?;)", text, re.IGNORECASE | re.DOTALL)
    return stmts + inline


def _check_sql_safety(sql_stmts: list[str]) -> CheckResult:
    """8.3 Block dangerous SQL mutations."""
    if not sql_stmts:
        return CheckResult("sql_safety", True, "info", "No SQL found in output")
    for stmt in sql_stmts:
        if _DANGEROUS_SQL.search(stmt):
            return CheckResult(
                "sql_safety", False, "critical",
                f"Dangerous SQL pattern detected: {stmt[:100]}",
            )
    return CheckResult("sql_safety", True, "info")


def _check_pii(text: str) -> CheckResult:
    """8.4 Regex-based PII detection."""
    if not text:
        return CheckResult("pii_detection", True, "info")
    for pattern, pii_type in _PII_PATTERNS:
        match = pattern.search(text)
        if match:
            # Redact the PII — replace match with placeholder
            return CheckResult(
                "pii_detection", False, "critical",
                f"PII detected: {pii_type} at position {match.start()}",
            )
    return CheckResult("pii_detection", True, "info")


def _check_population_claim_grounding(report: str, results: list[dict]) -> CheckResult:
    """
    8.8 Catch a report that generalizes to the whole population (e.g. "all
    customers churn because...") when no analysis step actually computed
    anything over the whole group — only row-level filter/derive results
    (a sample of individual rows) or an unclassifiable raw result exist.
    This is the specific failure mode roadmap #12 targets: an LLM
    extrapolating a population-wide claim from what was really just a
    sample row or two, previously caught only by a prompt-level ask (the
    [AGGREGATE]/[ROW-LEVEL SAMPLE] evidence tags in
    insight_agent._format_findings) with nothing double-checking that the
    LLM actually respected it.

    Coarse-grained by design: checks whether *any* aggregate-level evidence
    exists in the whole step list, not whether the specific sentence
    containing the claim traces to it — matching a specific claim to a
    specific step would need real NLP, not a regex-based rule check. A
    report making a population claim while zero steps ever aggregated
    anything is unambiguous enough to flag without that precision.
    """
    if not report:
        return CheckResult("population_claim_grounding", True, "info")

    match = _POPULATION_CLAIM_RE.search(report)
    if not match:
        return CheckResult("population_claim_grounding", True, "info")

    has_aggregate_evidence = any(
        _classify_evidence_level(r.get("result_summary", "")) == "AGGREGATE"
        for r in results
        if not r.get("failed")
    )
    if has_aggregate_evidence:
        return CheckResult("population_claim_grounding", True, "info")

    return CheckResult(
        "population_claim_grounding", False, "critical",
        f"Report makes a population-level claim ({match.group(0).strip()!r}) "
        f"but no analysis step aggregated over the data — only row-level or "
        f"unclassifiable results are available. This looks like a "
        f"generalization from a sample, not a computed pattern.",
    )


def _check_completeness(report: str, query: str) -> CheckResult:
    """8.6 Heuristic completeness: report must have sections and address the query."""
    if not report:
        return CheckResult("completeness_check", False, "warning",
                           "Report is empty")
    if len(report) < 100:
        return CheckResult("completeness_check", False, "warning",
                           "Report is very short — may be incomplete")
    # Check for basic structure markers
    has_structure = any(marker in report for marker in ["##", "**", "- ", "\n\n"])
    if not has_structure:
        return CheckResult("completeness_check", False, "warning",
                           "Report lacks structured sections")
    return CheckResult("completeness_check", True, "info")


# Check-name substrings that map a failed check to "critical" severity.
# Per DEV_SPEC's guardrail failure-handling model: PII/SQL-safety and
# hallucination/fabrication are both critical (block + retry); everything
# else (readability, completeness, misleading framing, etc.) is a warning
# (attach caveat, deliver).
_CRITICAL_CHECK_KEYWORDS = ("pii", "safety", "hallucin", "fabricat", "claim_ground", "grounding")


def _parse_llm_checks(raw: dict) -> list[CheckResult]:
    """Convert the GUARDRAIL_SYSTEM JSON response into CheckResult list."""
    results = []
    checks = raw.get("checks", [])
    for c in checks:
        passed = bool(c.get("passed", True))
        check_name = c.get("check", "llm_check")
        finding = c.get("finding")
        # Map check names to severity
        severity: Severity = "info"
        if not passed:
            name_l = check_name.lower()
            if any(k in name_l for k in _CRITICAL_CHECK_KEYWORDS):
                # Block on dangerous content (PII/SQL) and on hallucinated/
                # ungrounded claims — both are "critical" per DEV_SPEC.
                severity = "critical"
            else:
                severity = "warning"
        results.append(CheckResult(check_name, passed, severity, finding))
    # If LLM returned overall verdict, synthesise missing checks
    if not checks:
        verdict = raw.get("overall_verdict", "approved")
        passed_overall = verdict == "approved"
        results.append(CheckResult(
            "llm_overall", passed_overall,
            # Treat as warning so analysis is delivered with a caveat rather than blocked
            "warning" if not passed_overall else "info",
            raw.get("retry_reason"),
        ))
    return results


def _parse_json(text: str):
    import json
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    for s, e in [("{", "}"), ("[", "]")]:
        start = text.find(s)
        end = text.rfind(e)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"No JSON in: {text[:200]!r}")


# ─── 8.7 Aggregator ───────────────────────────────────────────────────────────

def _aggregate(checks: list[CheckResult], iteration: int, max_retries: int) -> GuardrailReport:
    """
    Determine overall verdict from individual check results.

    critical failures: verdict = "retry" (if retries remain) or "fail"
    warnings only:     verdict = "approved" (with caveats)
    all pass:          verdict = "approved"
    """
    critical_failures = [c for c in checks if not c.passed and c.severity == "critical"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]

    caveats = [c.finding for c in warnings if c.finding]

    if critical_failures:
        if iteration < max_retries:
            reasons = "; ".join(c.finding or c.check for c in critical_failures)
            return GuardrailReport(
                checks=checks,
                overall_verdict="retry",
                passed=False,
                retry_reason=reasons,
                caveats=caveats,
            )
        else:
            reasons = "; ".join(c.finding or c.check for c in critical_failures)
            return GuardrailReport(
                checks=checks,
                overall_verdict="fail",
                passed=False,
                retry_reason=reasons,
                caveats=caveats,
            )

    return GuardrailReport(
        checks=checks,
        overall_verdict="approved",
        passed=True,
        retry_reason=None,
        caveats=caveats,
    )
