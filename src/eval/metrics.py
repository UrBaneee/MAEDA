"""
Eval metrics — Phase 9.

Output quality metrics (LLM-as-judge + rule-based):
  9.2 answer_relevance   — does output answer the user's question?
  9.3 groundedness       — every claim traceable to data or RAG source?
  9.4 factual_accuracy   — numerical values correct against ground truth?
  9.5 completeness       — analysis covers all query aspects?

Agent performance metrics (rule-based):
  intent_accuracy        — did intent parser correctly parse the query?
  tool_selection         — appropriate tools chosen?
  plan_efficiency        — plan steps reasonable count?
  chart_appropriateness  — chart types match data?

System metrics (derived from state):
  total_latency          — from state timestamps if available
  token_cost             — from cost_tracker
  retry_count            — iteration_count
  error_rate             — 1 unless a genuine pipeline crash occurred
  safe_refusal           — 1 if guardrail correctly blocked an unsafe/
                            ungrounded output (informational — excluded
                            from the weighted aggregate score, see
                            runner._aggregate_score)
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

from src.config.agent_prompts import EVAL_RELEVANCE_SYSTEM
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("maeda.eval.metrics")


@dataclass
class MetricScore:
    metric: str
    score: float            # 0.0–1.0  (system metrics like latency stored as raw value)
    label: str              # "pass" | "warn" | "fail"
    reasoning: str = ""
    raw_value: Optional[Any] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _label(score: float, warn: float = 0.6, fail: float = 0.4) -> str:
    if score >= warn:
        return "pass"
    if score >= fail:
        return "warn"
    return "fail"


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_eval_llm():
    """
    The eval judge deliberately does NOT reuse settings.llm_provider/llm_model
    (the agent pipeline's own model) — a judge sharing weights/training with
    what it's scoring is a self-preference risk. resolved_eval_provider/
    resolved_eval_model prefer a different provider (if a real key exists for
    one) and a stronger model, falling back to the agent's own settings only
    when nothing else is configured.
    """
    if settings.resolved_eval_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.resolved_eval_model, temperature=0.0,
            max_tokens=256, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.resolved_eval_model, temperature=0.0,
        max_tokens=256, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── 9.2 / 9.3 LLM-as-judge ─────────────────────────────────────────────────

async def _judge_once(_llm, prompt: str) -> dict:
    """Single judge call, returning the parsed {"answer_relevance", "groundedness", "reasoning"} dict."""
    response = await _llm.ainvoke([
        SystemMessage(content=EVAL_RELEVANCE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw = _parse_json(response.content.strip())
    return {
        "answer_relevance": float(raw.get("answer_relevance", 0.5)),
        "groundedness": float(raw.get("groundedness", 0.5)),
        "reasoning": raw.get("reasoning", ""),
    }


async def score_relevance_and_groundedness(
    query: str,
    report: str,
    analysis_results: list[dict],
    rag_context: list[dict],
    llm=None,
    n_samples: Optional[int] = None,
) -> tuple[MetricScore, MetricScore]:
    """
    Scores both answer_relevance and groundedness via n_samples independent
    judge calls (default settings.eval_judge_samples), aggregated by median.
    A single judge call is noisy enough that the same case can score 0.5 on
    one run and 1.0 on the next; the median of several calls is far more
    stable while still being cheap relative to the rest of the pipeline.
    """
    _llm = llm or _build_eval_llm()
    n = n_samples if n_samples is not None else settings.eval_judge_samples
    n = max(1, n)

    findings = "; ".join(
        r.get("result_summary", "") for r in analysis_results if not r.get("failed")
    )[:600]
    rag_text = " | ".join(
        (c.get("content", "") if isinstance(c, dict) else str(c))[:100]
        for c in rag_context[:3]
    )

    prompt = (
        f"### User Query\n{query}\n\n"
        f"### Analysis Findings\n{findings or 'None'}\n\n"
        f"### RAG Context\n{rag_text or 'None'}\n\n"
        f"### Report\n{report[:1200]}\n"
    )

    try:
        samples = await asyncio.gather(*[_judge_once(_llm, prompt) for _ in range(n)])
        rel = statistics.median(s["answer_relevance"] for s in samples)
        gnd = statistics.median(s["groundedness"] for s in samples)
        reasoning = samples[0]["reasoning"]
        if n > 1:
            rel_spread = max(s["answer_relevance"] for s in samples) - min(s["answer_relevance"] for s in samples)
            gnd_spread = max(s["groundedness"] for s in samples) - min(s["groundedness"] for s in samples)
            if rel_spread >= 0.3 or gnd_spread >= 0.3:
                reasoning += (
                    f" [judge disagreement across {n} samples: "
                    f"relevance spread={rel_spread:.2f}, groundedness spread={gnd_spread:.2f}]"
                )
        return (
            MetricScore("answer_relevance", rel, _label(rel), reasoning),
            MetricScore("groundedness", gnd, _label(gnd), reasoning),
        )
    except Exception as exc:
        logger.warning("LLM eval judge failed: %s — using heuristic fallback", exc)
        return (
            MetricScore("answer_relevance", 0.5, "warn", f"LLM unavailable: {exc}"),
            MetricScore("groundedness", 0.5, "warn", f"LLM unavailable: {exc}"),
        )


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


def score_tool_selection(analysis_results: list[dict]) -> MetricScore:
    """Were appropriate tools used? Proxy: no failed steps."""
    if not analysis_results:
        return MetricScore("tool_selection", 0.5, "warn", "No analysis steps executed")

    failed = sum(1 for r in analysis_results if r.get("failed"))
    success_rate = 1.0 - (failed / len(analysis_results))
    return MetricScore("tool_selection", success_rate, _label(success_rate),
                       f"{failed}/{len(analysis_results)} steps failed")


def score_plan_efficiency(analysis_results: list[dict]) -> MetricScore:
    """Was the plan efficient? Proxy: 1–6 steps is optimal."""
    n = len(analysis_results)
    if n == 0:
        return MetricScore("plan_efficiency", 0.5, "warn", "No steps executed")
    if 1 <= n <= 6:
        score = 1.0
    elif n <= 10:
        score = 0.7
    else:
        score = 0.4
    return MetricScore("plan_efficiency", score, _label(score), f"{n} analysis steps")


def score_chart_appropriateness(charts: list[dict]) -> MetricScore:
    """Were chart types reasonable? Proxy: charts generated without errors."""
    if not charts:
        return MetricScore("chart_appropriateness", 0.5, "warn", "No charts generated")
    valid = [c for c in charts if c.get("chart_type") and c.get("chart_type") != "error"]
    score = len(valid) / max(len(charts), 1)
    return MetricScore("chart_appropriateness", score, _label(score),
                       f"{len(valid)}/{len(charts)} charts valid")


# ─── System metrics ───────────────────────────────────────────────────────────

def score_system_metrics(state: dict, start_time: Optional[float] = None) -> list[MetricScore]:
    metrics = []

    # Token cost
    token_usage = state.get("token_usage") or {}
    total_cost = sum(
        v.get("total_cost", 0) for v in token_usage.values() if isinstance(v, dict)
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
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    for s, e in [("{", "}"), ("[", "]")]:
        start, end = text.find(s), text.rfind(e)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"No JSON in: {text[:200]!r}")
