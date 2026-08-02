"""
Eval metrics — Phase 9, rebuilt in eval v2 Step 2 (docs/eval_v2_plan.md).

Quality metrics (weighted into runner._aggregate_score):
  9.2 answer_relevance   — LLM judge: does the report answer the question?
  9.3 groundedness       — LLM judge: claim list, supported_count/total_claims
                            computed in code (Step 2c), not asked as a bare score
  9.4 factual_accuracy   — rule-based: numerical values vs. ground truth
  intent_accuracy        — rule-based: did intent parser get type/metrics right?
  tool_selection         — did the planner pick the right tool(s)? scored
                            against GoldenTestCase.expected_tools (Step 2d) --
                            distinct from step_success_rate below
  chart_appropriateness  — did chart generation pick the right type(s)?
                            scored against GoldenTestCase.expected_chart_types
                            (Step 2d) -- distinct from "did it error"

Operational/diagnostic metrics (reported, weight=0 in the aggregate --
see runner._aggregate_score's comment for why folding these into a quality
average dilutes it rather than measuring anything):
  step_success_rate      — did analysis steps execute without throwing?
                            (this is what "tool_selection" used to mean,
                            before Step 2d split "didn't throw" from
                            "was the right choice")
  total_latency, token_cost, retry_count, error_rate, safe_refusal

`completeness` and `plan_efficiency` (formerly listed here) were removed
in Step 2d: completeness was documented but never implemented, and
plan_efficiency's "1-6 steps = full score" measured nothing a genuinely
better metric wouldn't already imply.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.agent_prompts import EVAL_GROUNDEDNESS_SYSTEM, EVAL_RELEVANCE_SYSTEM
from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import call_with_rate_limit_retry

logger = get_logger("maeda.eval.metrics")


@dataclass
class MetricScore:
    metric: str
    score: float            # 0.0–1.0  (system metrics like latency stored as raw value)
    label: str              # "pass" | "warn" | "fail" | "error"
    reasoning: str = ""
    raw_value: Optional[Any] = None
    # False means this metric could not actually be scored this run (judge
    # unreachable after retries, malformed response) -- score/label are a
    # placeholder, not a real measurement. Eval v2 Step 2a: previously a
    # judge-parse failure and a judge genuinely scoring 0.5 were the exact
    # same value in the data (0.5 was the single most common score),
    # indistinguishable after the fact. _aggregate_score (runner.py) skips
    # valid=False entries entirely instead of averaging in a placeholder.
    valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _label(score: float, warn: float = 0.6, fail: float = 0.4) -> str:
    if score >= warn:
        return "pass"
    if score >= fail:
        return "warn"
    return "fail"


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_eval_llm(max_tokens: int = 256):
    """
    The eval judge deliberately does NOT reuse settings.llm_provider/llm_model
    (the agent pipeline's own model) — a judge sharing weights/training with
    what it's scoring is a self-preference risk. resolved_eval_provider/
    resolved_eval_model prefer a different provider (if a real key exists for
    one) and a stronger model, falling back to the agent's own settings only
    when nothing else is configured.

    max_tokens defaults to 256 (enough for a float + a short reasoning
    string, what answer_relevance needs) but score_groundedness passes a
    much higher value: found live against real cached reports (eval v2
    Step 2c) that a claim list for a report with several claims routinely
    exceeds 256 tokens, silently truncating the JSON mid-response --
    _parse_json's bracket-matching fallback would then occasionally parse
    the truncated `"claims": [...]` fragment as a bare list, crashing with
    "'list' object has no attribute 'get'" instead of a clean parse error.
    """
    if settings.resolved_eval_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.resolved_eval_model, temperature=0.0,
            max_tokens=max_tokens, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.resolved_eval_model, temperature=0.0,
        max_tokens=max_tokens, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── 9.2 / 9.3 LLM-as-judge ─────────────────────────────────────────────────
#
# answer_relevance and groundedness are scored by two fully independent LLM
# calls (eval v2 Step 2b) -- they used to share one call and one reasoning
# string, so groundedness's reasoning sometimes described relevance instead
# (verified 20/20 cases identical in the audit that started this rewrite;
# see docs/eval_v2_plan.md). groundedness (Step 2c) also no longer asks the
# judge for a bare 0-1 score: it asks for a structured claim list and
# computes supported_count/total_claims in code, so the reasoning shown is
# literally the claims the score was computed from, not free-text prose
# that can drift from the number.

_STEP_RESULT_CAP = 1500  # see render_findings


def render_findings(analysis_results: list[dict]) -> str:
    """One line per successful step: its result_summary, plus a compact
    rendering of the structured `result` field beyond what result_summary
    already captures.

    Found via a real judge-calibration annotation session (eval v2 Step 3,
    see docs/judge_calibration.md, case C01): significance-test output
    (an ANOVA's statistic/p-value) lived only in a step's `result` dict,
    never in `result_summary` -- invisible to the judge AND to human
    annotators, so neither could confirm or refute the report's own
    citation of it. Capped per step so a full-dataframe-sized `result`
    (e.g. a `derive` step returning all 12,240 rows) doesn't blow up the
    prompt; small structured results (comparison/statistical-test steps)
    render in full, well under the cap.

    Also used by scripts/export_for_annotation.py, so the judge and the
    human annotator see byte-for-byte the same findings -- they used to
    diverge (the export script had its own, differently-formatted
    rendering), which would have made measured judge/human agreement
    compare two different inputs, not the same one.
    """
    lines = []
    for r in analysis_results:
        if r.get("failed"):
            continue
        summary = r.get("result_summary", "")
        label = r.get("method") or r.get("tool") or "step"
        line = f"[{label}] {summary}" if summary else f"[{label}] (no summary)"
        result_repr = repr(r.get("result"))
        if result_repr != "None":
            line += f"\n    full result: {result_repr[:_STEP_RESULT_CAP]}"
            if len(result_repr) > _STEP_RESULT_CAP:
                line += f" ...({len(result_repr) - _STEP_RESULT_CAP} more chars truncated)"
        lines.append(line)
    return "\n".join(lines) or "(no successful analysis steps)"


def render_rag_context(rag_context: list[dict]) -> str:
    """Full content, every chunk -- previously capped at 3 chunks / 100
    chars each in the judge's view while the human annotator's export saw
    all of it uncapped, the same divergence render_findings fixes above."""
    if not rag_context:
        return ""
    parts = []
    for c in rag_context:
        content = c.get("content", "") if isinstance(c, dict) else str(c)
        parts.append(content)
    return "\n---\n".join(parts)


def render_data_quality(data_quality_report: Optional[dict]) -> str:
    """The report's own data-quality caveats (e.g. "238 duplicated rows")
    cite this, but it was never shown to the judge OR the human annotator
    at all before eval v2 Step 3 -- both had to take such claims on faith.
    Shows the row count and the flagged issues (what a report can actually
    cite), not the full per-column profile (sample values etc., not
    typically cited and unbounded for a wide dataset)."""
    if not data_quality_report:
        return "None"
    issues = data_quality_report.get("quality_issues") or []
    row_count = data_quality_report.get("row_count")
    header = f"{row_count} rows" if row_count is not None else ""
    if not issues:
        return (header + " — no quality issues flagged") if header else "No quality issues flagged"
    issue_lines = [
        f"- {i.get('issue', '?')} ({i.get('severity', '?')}): {i.get('detail', '')}"
        for i in issues
    ]
    return (header + "\n" if header else "") + "\n".join(issue_lines)


def _build_judge_prompt(
    query: str,
    report: str,
    analysis_results: list[dict],
    rag_context: list[dict],
    data_quality_report: Optional[dict] = None,
) -> str:
    """No truncation on findings or report (removed in eval v2 Step 3,
    found via a real annotation session where the judge's 600/1200-char
    caps cut off the exact groupby results and Automated Caveats section a
    human annotator could see in full -- the two were being asked to judge
    different material). At this account's current OpenAI tier, the cost
    of showing the whole thing is a few cents across the full suite (see
    the token-cost analysis in this session's conversation) -- not a
    reason to keep judge and human looking at different inputs."""
    return (
        f"### User Query\n{query}\n\n"
        f"### Analysis Findings\n{render_findings(analysis_results)}\n\n"
        f"### Data Quality\n{render_data_quality(data_quality_report)}\n\n"
        f"### RAG Context\n{render_rag_context(rag_context) or 'None'}\n\n"
        f"### Report\n{report}\n"
    )


async def _run_judge_samples(sample_fn, n: int) -> tuple[list[dict], list[BaseException]]:
    """Run n independent judge samples via sample_fn() (an async, zero-arg
    callable -- bind the llm/prompt with a closure at the call site),
    tolerating partial failure. return_exceptions=True is deliberate: with
    a plain asyncio.gather, one sample raising (rate limit exhausted,
    malformed JSON) used to abort the whole batch and throw away every
    other sample that had already succeeded. Never raises; callers decide
    what an all-failed batch means."""
    raw_results = await asyncio.gather(
        *[sample_fn() for _ in range(n)], return_exceptions=True,
    )
    samples = [r for r in raw_results if not isinstance(r, BaseException)]
    errors = [r for r in raw_results if isinstance(r, BaseException)]
    return samples, errors


def _render_claims(claims: list[dict]) -> str:
    if not claims:
        return "No checkable claims in report."
    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
    parts = []
    for c in claims:
        mark = "✓" if c.get("supported") else "✗"
        evidence = c.get("evidence")
        suffix = f" (evidence: {evidence})" if c.get("supported") and evidence else \
                 "" if c.get("supported") else " (no evidence found)"
        parts.append(f"{mark} {c.get('claim', '')}{suffix}")
    return f"{supported}/{len(claims)} claims supported. " + " | ".join(parts)


async def score_answer_relevance(
    query: str,
    report: str,
    analysis_results: list[dict],
    rag_context: list[dict],
    llm=None,
    n_samples: Optional[int] = None,
    data_quality_report: Optional[dict] = None,
) -> MetricScore:
    """Does the report directly answer the user's question -- independent
    of whether its claims are grounded (score_groundedness scores that)."""
    _llm = llm or _build_eval_llm()
    n = max(1, n_samples if n_samples is not None else settings.eval_judge_samples)
    prompt = _build_judge_prompt(query, report, analysis_results, rag_context, data_quality_report)

    async def _one_sample() -> dict:
        async def _call():
            response = await _llm.ainvoke([
                SystemMessage(content=EVAL_RELEVANCE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            raw = _parse_json(response.content.strip())
            if "answer_relevance" not in raw:
                # A missing field is a malformed response, not "the judge
                # scored it 0.5" -- raising here (instead of raw.get(key,
                # 0.5)) lets this sample be excluded/retried like any other
                # failure rather than silently injecting a fake mid-point
                # score (eval v2 Step 2a).
                raise ValueError(f"Judge response missing 'answer_relevance': {raw!r}")
            return {"score": float(raw["answer_relevance"]), "reasoning": raw.get("reasoning", "")}
        return await call_with_rate_limit_retry(_call, logger=logger, label="Relevance judge call")

    samples, errors = await _run_judge_samples(_one_sample, n)
    if not samples:
        last_err = errors[-1] if errors else RuntimeError("unknown judge failure")
        logger.warning("All %d answer_relevance sample(s) failed: %s", n, last_err)
        return MetricScore("answer_relevance", 0.5, "error", f"LLM unavailable: {last_err}", valid=False)
    if errors:
        logger.warning(
            "%d/%d answer_relevance sample(s) failed (%s); scoring from the %d that succeeded",
            len(errors), n, errors[-1], len(samples),
        )

    scores = [s["score"] for s in samples]
    score = statistics.median(scores)
    reasoning = samples[0]["reasoning"]
    if len(samples) > 1:
        spread = max(scores) - min(scores)
        if spread >= 0.3:
            reasoning += f" [judge disagreement across {len(samples)} samples: spread={spread:.2f}]"
    return MetricScore("answer_relevance", score, _label(score), reasoning)


async def score_groundedness(
    query: str,
    report: str,
    analysis_results: list[dict],
    rag_context: list[dict],
    llm=None,
    n_samples: Optional[int] = None,
    data_quality_report: Optional[dict] = None,
) -> MetricScore:
    """Is every claim in the report traceable to the analysis findings or
    RAG context -- independent of whether the report answers the question
    (score_answer_relevance scores that). Score is computed in code as
    supported_count/total_claims from a judge-extracted claim list, not
    asked from the model as a bare 0-1 number (eval v2 Step 2c)."""
    # 1024, not the 256-token default -- a claim list for a report making
    # several claims routinely needs more room than a bare float ever did;
    # found live via truncated/unparseable responses (see _build_eval_llm).
    _llm = llm or _build_eval_llm(max_tokens=1024)
    n = max(1, n_samples if n_samples is not None else settings.eval_judge_samples)
    prompt = _build_judge_prompt(query, report, analysis_results, rag_context, data_quality_report)

    async def _one_sample() -> dict:
        async def _call():
            response = await _llm.ainvoke([
                SystemMessage(content=EVAL_GROUNDEDNESS_SYSTEM),
                HumanMessage(content=prompt),
            ])
            raw = _parse_json(response.content.strip())
            claims = raw.get("claims")
            if not isinstance(claims, list):
                raise ValueError(f"Judge response missing 'claims' list: {raw!r}")
            supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
            total = len(claims)
            # No checkable claims is neither grounded nor ungrounded -- a
            # lenient default, same convention and same value as
            # score_factual_accuracy's "no numbers to cross-check" case
            # below.
            score = (supported / total) if total else 0.8
            return {"score": score, "claims": claims, "reasoning": raw.get("reasoning", "")}
        return await call_with_rate_limit_retry(_call, logger=logger, label="Groundedness judge call")

    samples, errors = await _run_judge_samples(_one_sample, n)
    if not samples:
        last_err = errors[-1] if errors else RuntimeError("unknown judge failure")
        logger.warning("All %d groundedness sample(s) failed: %s", n, last_err)
        return MetricScore("groundedness", 0.5, "error", f"LLM unavailable: {last_err}", valid=False)
    if errors:
        logger.warning(
            "%d/%d groundedness sample(s) failed (%s); scoring from the %d that succeeded",
            len(errors), n, errors[-1], len(samples),
        )

    scores = [s["score"] for s in samples]
    score = statistics.median(scores)
    # The sample whose own score is closest to the reported median supplies
    # the claim list shown as "the reasoning" -- so the reasoning can never
    # describe a different result than the score (the old shared-prompt
    # design let exactly this happen; see module comment above).
    closest = min(samples, key=lambda s: abs(s["score"] - score))
    reasoning = _render_claims(closest["claims"])
    if len(samples) > 1:
        spread = max(scores) - min(scores)
        if spread >= 0.3:
            reasoning += f" [judge disagreement across {len(samples)} samples: spread={spread:.2f}]"
    return MetricScore("groundedness", score, _label(score), reasoning)


async def score_relevance_and_groundedness(
    query: str,
    report: str,
    analysis_results: list[dict],
    rag_context: list[dict],
    llm=None,
    n_samples: Optional[int] = None,
    data_quality_report: Optional[dict] = None,
) -> tuple[MetricScore, MetricScore]:
    """Back-compat convenience wrapper composing the two independent judge
    calls above -- kept so callers that just want "both scores together"
    (scripts/measure_noise.py, tests/integration/test_eval_judge_calibration.py)
    don't need to change. src/eval/runner.py's EvalRunner calls
    score_answer_relevance/score_groundedness directly instead, since it
    needs to reuse-check each one independently."""
    rel, gnd = await asyncio.gather(
        score_answer_relevance(query, report, analysis_results, rag_context, llm=llm, n_samples=n_samples,
                               data_quality_report=data_quality_report),
        score_groundedness(query, report, analysis_results, rag_context, llm=llm, n_samples=n_samples,
                           data_quality_report=data_quality_report),
    )
    return rel, gnd


# ─── 9.4 Factual accuracy ─────────────────────────────────────────────────────

_THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> set[float]:
    """Extract numeric values from text, tolerant of thousands separators.

    LLM-written numbers commonly use thousands separators ($1,363,760.55);
    a naive \\b\\d+(?:\\.\\d+)?\\b regex splits those into unrelated
    fragments ("1", "363", "760.55") that never match a ground-truth value.
    Strip the separating commas first so the whole number extracts as one.
    """
    cleaned = _THOUSANDS_SEP_RE.sub("", text)
    numbers = set()
    for m in _NUMBER_RE.finditer(cleaned):
        try:
            numbers.add(float(m.group()))
        except ValueError:
            continue
    return numbers


def _numbers_match(a: float, b: float) -> bool:
    """Tolerant numeric equality — allows LLM rounding, not exact string match.

    1% relative tolerance (with a small absolute floor for near-zero values
    like correlation coefficients) covers "$1,363,761" vs "1363760.55" or
    "0.35" vs "0.3536" without being loose enough to pass a genuinely wrong
    number.
    """
    tolerance = max(abs(b) * 0.01, 0.005)
    return abs(a - b) <= tolerance


def score_factual_accuracy(
    report: str,
    analysis_results: list[dict],
    ground_truth: Optional[dict] = None,
) -> MetricScore:
    """
    Check that numbers appearing in the report have at least some overlap
    with the analysis result summaries (proxy for factual accuracy).
    If ground_truth is provided, check exact values.
    """
    if not report:
        return MetricScore("factual_accuracy", 0.0, "fail", "Empty report")

    report_nums = _extract_numbers(report)
    summaries = " ".join(
        r.get("result_summary", "") for r in analysis_results if not r.get("failed")
    )
    summary_nums = _extract_numbers(summaries)

    if ground_truth:
        # Check against explicit ground truth values
        expected = {v for v in ground_truth.values() if isinstance(v, (int, float))}
        if expected:
            matched = sum(
                1 for exp in expected
                if any(_numbers_match(rn, exp) for rn in report_nums)
            )
            overlap = matched / len(expected)
            return MetricScore("factual_accuracy", overlap, _label(overlap),
                               f"Ground truth overlap: {overlap:.0%}")

    if not summary_nums:
        return MetricScore("factual_accuracy", 0.8, "pass", "No numbers to cross-check")

    if not report_nums:
        return MetricScore("factual_accuracy", 0.5, "warn", "Report contains no numbers")

    matched = sum(
        1 for sn in summary_nums if any(_numbers_match(rn, sn) for rn in report_nums)
    )
    overlap = matched / max(len(summary_nums), 1)
    score = min(1.0, overlap * 2)  # generous: even 50% overlap → full score
    return MetricScore("factual_accuracy", score, _label(score),
                       f"{matched}/{len(summary_nums)} numbers overlap")


# ─── 9.5 Agent performance ────────────────────────────────────────────────────

def score_intent_accuracy(
    parsed_intent: dict,
    expected_query_type: Optional[str] = None,
    expected_metrics: Optional[list] = None,
) -> MetricScore:
    if not parsed_intent:
        return MetricScore("intent_accuracy", 0.0, "fail", "No parsed intent")

    score = 0.0
    reasons = []

    # Confidence from intent parser is a direct signal
    confidence = float(parsed_intent.get("confidence", 0.5))
    score += confidence * 0.5

    if expected_query_type:
        if parsed_intent.get("query_type") == expected_query_type:
            score += 0.3
            reasons.append(f"query_type correct: {expected_query_type}")
        else:
            reasons.append(
                f"query_type mismatch: got {parsed_intent.get('query_type')} "
                f"expected {expected_query_type}"
            )

    if expected_metrics:
        got = set(parsed_intent.get("target_metrics") or [])
        exp = set(expected_metrics)
        if exp:
            overlap = len(got & exp) / len(exp)
            score += overlap * 0.2
            reasons.append(f"metrics overlap: {overlap:.0%}")

    score = min(1.0, score)
    return MetricScore("intent_accuracy", score, _label(score), "; ".join(reasons))


def score_step_success_rate(analysis_results: list[dict]) -> MetricScore:
    """Did analysis steps execute without throwing? Operational/diagnostic
    signal, not a quality signal (see runner._aggregate_score) -- this is
    what "tool_selection" used to mean before eval v2 Step 2d split "didn't
    throw" from "was the right tool," which is why it was constant at 1.00
    across every case in the last audited run: a step using the WRONG tool
    can still execute cleanly, so this alone never measured tool choice."""
    if not analysis_results:
        return MetricScore("step_success_rate", 0.5, "warn", "No analysis steps executed")
    failed = sum(1 for r in analysis_results if r.get("failed"))
    success_rate = 1.0 - (failed / len(analysis_results))
    return MetricScore("step_success_rate", success_rate, _label(success_rate),
                       f"{failed}/{len(analysis_results)} steps failed")


def score_tool_selection(analysis_results: list[dict], expected_tools: Optional[list[str]] = None) -> MetricScore:
    """Did the planner pick an appropriate tool for this query, scored
    against GoldenTestCase.expected_tools (eval v2 Step 2d)?

    Binary hit, not a fractional |actual ∩ expected| / |expected| overlap:
    expected_tools is authored as a set of acceptable ALTERNATIVES ("either
    time_series or pandas_transform would be reasonable here"), not a
    checklist every entry must appear in. A multi-step plan also routinely
    mixes a directly-relevant tool with supporting steps (e.g. a
    pandas_transform to derive a column before a statistical_test) --
    penalizing that mix as "only half right" would punish a normal,
    correct plan. What matters is whether at least one step used an
    acceptable tool at all.
    """
    if expected_tools is None:
        return MetricScore("tool_selection", 0.8, "pass", "No expected_tools defined for this case")
    expected = set(expected_tools)
    if not expected:
        return MetricScore("tool_selection", 1.0, "pass", "No specific tool expected for this case")
    actual = {r.get("tool") for r in analysis_results if r.get("tool")}
    if not actual:
        return MetricScore("tool_selection", 0.0, "fail",
                           f"No steps executed; expected one of {sorted(expected)}")
    score = 1.0 if (actual & expected) else 0.0
    return MetricScore("tool_selection", score, _label(score),
                       f"used {sorted(actual)}, expected one of {sorted(expected)}")


def score_chart_appropriateness(charts: list[dict], expected_chart_types: Optional[list[str]] = None) -> MetricScore:
    """Were the chart TYPES actually appropriate for this query, scored
    against GoldenTestCase.expected_chart_types (eval v2 Step 2d)? Not
    derived from the same data-shape rules src.tools.chart_tool's
    recommend_chart() uses internally -- that would grade the function
    against itself, which is exactly why this metric was structurally
    constant at 1.00 before: recommend_chart is rule-based and always
    returns *some* valid type, so "did chart generation not error" could
    never fail.

    Binary hit, same reasoning as score_tool_selection above:
    expected_chart_types is a set of acceptable alternatives ("bar or
    horizontal_bar are both fine here"), and an extra chart beyond the
    expected one isn't inherently wrong.
    """
    if expected_chart_types is None:
        return MetricScore("chart_appropriateness", 0.8, "pass", "No expected_chart_types defined for this case")
    expected = set(expected_chart_types)
    actual = {c.get("chart_type") for c in charts if c.get("chart_type")}
    if not expected:
        # Explicitly authored as "no chart needed" -- an unrequested extra
        # chart isn't really wrong, just unnecessary, so it isn't scored
        # as harshly as a genuinely missing expected chart below.
        score = 1.0 if not actual else 0.8
        return MetricScore("chart_appropriateness", score, _label(score),
                           f"no chart type expected; {len(actual)} generated")
    if not actual:
        return MetricScore("chart_appropriateness", 0.0, "fail",
                           f"No charts generated; expected one of {sorted(expected)}")
    score = 1.0 if (actual & expected) else 0.0
    return MetricScore("chart_appropriateness", score, _label(score),
                       f"used {sorted(actual)}, expected one of {sorted(expected)}")


# ─── System metrics ───────────────────────────────────────────────────────────

def score_system_metrics(state: dict, start_time: Optional[float] = None) -> list[MetricScore]:
    metrics = []

    # Token cost — CostTracker.to_dict() (src/utils/cost_tracker.py) writes
    # the per-agent cost under "cost_usd", not "total_cost". Reading the
    # wrong key here silently summed to 0 for every run, so token_cost
    # scored 1.0 regardless of what a run actually spent.
    token_usage = state.get("token_usage") or {}
    total_cost = sum(
        v.get("cost_usd", 0) for v in token_usage.values() if isinstance(v, dict)
    )
    metrics.append(MetricScore(
        "token_cost", min(1.0, max(0.0, 1.0 - total_cost)),
        "pass" if total_cost < 0.10 else "warn",
        f"${total_cost:.4f}",
        raw_value=total_cost,
    ))

    # Retry count
    retries = max(0, state.get("iteration_count", 1) - 1)
    retry_score = 1.0 if retries == 0 else (0.7 if retries == 1 else 0.3)
    metrics.append(MetricScore("retry_count", retry_score, _label(retry_score),
                               f"{retries} retries", raw_value=retries))

    # Error rate — a guardrail-blocked "safe refusal" (state["error_type"] ==
    # "safe_refusal") is the pipeline correctly declining to deliver an
    # ungrounded/unsafe report, not a system failure. Only a genuine crash
    # (data connection failure, unhandled exception, etc.) should count
    # against error_rate; refusals are tracked separately below so the two
    # aren't conflated in regression detection or the aggregate score.
    is_safe_refusal = state.get("error_type") == "safe_refusal"
    has_crash = bool(state.get("error")) and not is_safe_refusal
    metrics.append(MetricScore("error_rate", 0.0 if has_crash else 1.0,
                               "fail" if has_crash else "pass",
                               state.get("error") or "No errors"))
    metrics.append(MetricScore("safe_refusal", 1.0 if is_safe_refusal else 0.0,
                               "info",
                               state.get("error") or "" if is_safe_refusal else "No refusal"))

    # Latency (if start_time provided)
    if start_time:
        latency = time.time() - start_time
        lat_score = 1.0 if latency < 30 else (0.7 if latency < 60 else 0.3)
        metrics.append(MetricScore("total_latency", lat_score, _label(lat_score),
                                   f"{latency:.1f}s", raw_value=latency))

    return metrics


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Every judge/metric prompt in this file asks for a top-level JSON
    *object* -- the caller always does raw.get(...). This used to also try
    matching a bare `[...]` array as a fallback; found live (eval v2 Step
    2c) that a truncated groundedness response (claims list cut off
    mid-response by max_tokens, see _build_eval_llm) could make the `{...}`
    parse fail and the `[...]` fallback then successfully parse just the
    claims fragment as a bare list -- which crashed callers with
    "'list' object has no attribute 'get'" instead of a clean, catchable
    parse error. Only ever matching `{...}` means any malformed/truncated
    response fails the same clear way."""
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    raise ValueError(f"No JSON object in: {text[:200]!r}")
