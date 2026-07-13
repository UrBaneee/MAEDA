"""
Insight Agent — Phase 7.

Responsibilities:
  7.1 Build a focused retrieval query from analysis results + parsed intent.
  7.2 (Handled upstream) RAG chunks arrive via state["rag_context"].
  7.3 Generate grounded insights by combining quantitative findings with
      domain knowledge from RAG-MCP-Server.
  7.4 Confidence scoring — each insight gets a 0–1 score.
  7.5 Report generator — markdown report from insights.
  7.6 Source attribution — every domain claim tracks its RAG source.

State fields read:  parsed_intent, analysis_results, intermediate_data,
                    rag_context, rag_sources, charts, data_quality_report
State fields written: insights, report, decision_trace, token_usage
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import INSIGHT_GENERATOR_SYSTEM, REPORT_WRITER_SYSTEM
from src.config.settings import settings
from src.state.graph_state import MAEDAState
from src.utils.logger import get_logger

logger = get_logger("maeda.agent.insight")


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Insight:
    finding: str
    evidence: list[str]
    confidence: float
    domain_context: str
    impact: Literal["high", "medium", "low"]
    recommendation: str
    sources: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Insight":
        # LLM returns a simpler structure; enrich with defaults where needed
        evidence = d.get("evidence", "")
        if isinstance(evidence, str):
            evidence = [evidence] if evidence else []
        return cls(
            finding=d.get("finding", ""),
            evidence=evidence,
            confidence=float(d.get("confidence", 0.7)),
            domain_context=d.get("domain_context", ""),
            impact=_score_to_impact(float(d.get("confidence", 0.7))),
            recommendation=d.get("recommendation", ""),
            sources=d.get("sources", []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _score_to_impact(confidence: float) -> Literal["high", "medium", "low"]:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    # Was hardcoded to temperature=0.3, bypassing settings.llm_temperature —
    # the only agent introducing real sampling variance. Reproduced a case
    # where that variance made the same real, non-empty analysis result
    # produce zero insights (and a report falsely claiming "no data") on one
    # call but three grounded insights on a replay of the identical prompt.
    # Insight generation should be as deterministic as every other agent.
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model, temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens_per_call, api_key=settings.anthropic_api_key or "sk-no-key",
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model, temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens_per_call, api_key=settings.openai_api_key or "sk-no-key",
    )


# ─── InsightAgent ─────────────────────────────────────────────────────────────

class InsightAgent(BaseAgent):
    """
    Generates grounded insights and a markdown report.

    Two entry points (matching graph nodes):
      - retrieve_query(state) → returns the synthesized retrieval query string
      - generate(state)       → populates state["insights"] and state["report"]
    """

    def __init__(self, llm=None):
        super().__init__("insight_agent")
        self._llm = llm or _build_llm()

    async def process(self, state: MAEDAState) -> MAEDAState:
        return await self.generate(state)

    # ── 7.1 Retrieval query builder ───────────────────────────────────────────

    def build_retrieval_query(self, state: MAEDAState) -> str:
        """
        Synthesize a focused RAG retrieval query from analysis results + intent.
        Falls back to user_query if no analysis context is available.
        """
        intent = state.get("parsed_intent") or {}
        results = state.get("analysis_results") or []

        parts: list[str] = []

        # Include the original user intent
        query_type = intent.get("query_type", "")
        metrics = intent.get("target_metrics") or []
        dimensions = intent.get("dimensions") or []
        if metrics:
            parts.append(f"{query_type} analysis of {', '.join(metrics)}")
        if dimensions:
            parts.append(f"by {', '.join(dimensions)}")

        # Append top findings from successful analysis steps
        summaries = [
            r["result_summary"]
            for r in results
            if not r.get("failed") and r.get("result_summary")
        ][:3]
        if summaries:
            parts.append("Key findings: " + "; ".join(summaries))

        query = " — ".join(parts) if parts else state.get("user_query", "")
        logger.debug("Retrieval query: %s", query)
        return query

    # ── 7.3 / 7.4 Insight generator with confidence scoring ──────────────────

    async def generate(self, state: MAEDAState) -> MAEDAState:
        """Generate insights by combining analysis results with RAG context."""
        analysis_results = state.get("analysis_results") or []
        rag_chunks = state.get("rag_context") or []
        rag_sources = state.get("rag_sources") or []

        successful = [r for r in analysis_results if not r.get("failed")]

        # Build the context prompt
        findings_text = _format_findings(analysis_results)
        rag_text = _format_rag_chunks(rag_chunks, rag_sources)

        prompt = (
            f"### Analysis Findings\n{findings_text}\n\n"
            f"### Domain Knowledge (from RAG)\n{rag_text}\n\n"
            f"### Original Query\n{state.get('user_query', '')}\n\n"
            "Generate actionable insights combining the quantitative findings "
            "with the domain context."
        )

        insights: list[Insight] = []
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=INSIGHT_GENERATOR_SYSTEM),
                HumanMessage(content=prompt),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            self.track_cost(
                state, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="generate_insights",
            )

            raw = _parse_json(response.content.strip())
            if not isinstance(raw, list):
                raw = [raw]
            # 7.4 Confidence scoring + 7.6 Source attribution
            for item in raw:
                insight = Insight.from_dict(item)
                # Attach RAG source references
                if rag_sources and not insight.sources:
                    insight.sources = [
                        s.get("source_file", "") for s in rag_sources[:3] if s.get("source_file")
                    ]
                insights.append(insight)

        except Exception as exc:
            logger.warning("Insight generation LLM failed: %s — using rule-based fallback", exc)
            insights = _rule_based_insights(successful, rag_chunks, rag_sources)

        state["insights"] = [i.to_dict() for i in insights]

        # 7.5 Report generation
        state["report"] = await self._generate_report(state, insights)

        # Append this turn's resolved intent + top findings to conversation
        # history so a follow-up query ("now break that down by quarter")
        # can be resolved against exactly what was asked/found this turn —
        # see IntentParserAgent._parse()'s consumption of this field.
        state["conversation_history"] = [
            *state.get("conversation_history", []),
            {"role": "assistant", "content": _format_assistant_turn_summary(state, insights)},
        ]

        state = self.log_decision(
            state,
            action="generate_insights",
            reasoning=(
                f"Generated {len(insights)} insights from "
                f"{len(successful)} analysis steps and {len(rag_chunks)} RAG chunks"
            ),
            inputs={"n_analysis_steps": len(successful), "n_rag_chunks": len(rag_chunks)},
            outputs={"n_insights": len(insights)},
            confidence=_avg_confidence(insights),
        )
        return state

    # ── 7.5 Report generator ──────────────────────────────────────────────────

    async def _generate_report(self, state: MAEDAState, insights: list[Insight]) -> str:
        """Produce a markdown report from insights + analysis context."""
        insights_text = json.dumps([i.to_dict() for i in insights], indent=2)
        charts_info = _format_charts_summary(state.get("charts") or [])
        quality_note = _format_quality_note(state.get("data_quality_report") or {})

        prompt = (
            f"### Insights\n{insights_text}\n\n"
            f"### Charts Generated\n{charts_info}\n\n"
            f"### Data Quality\n{quality_note}\n\n"
            f"### Original Query\n{state.get('user_query', '')}\n\n"
            "Write the full markdown report."
        )

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=REPORT_WRITER_SYSTEM),
                HumanMessage(content=prompt),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            self.track_cost(
                state, model=settings.llm_model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                call_label="generate_report",
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Report generation LLM failed: %s — using rule-based fallback", exc)
            return _rule_based_report(state, insights)


# ─── Helpers ─────────────────────────────────────────────────────────────────

# result_summary is written by the tool dispatchers as "pandas/{op} → ..." /
# "stats/{test} ..." / "anomaly/{method}: ..." / "timeseries: ..." /
# "comparison: ...". Prefixes that compute over the whole group (groupby,
# pivot, statistical tests, anomaly detection, trend/forecast, segment
# comparison) support a population-level claim; "filter"/"derive" preserve
# individual rows and only ever support example-level claims.
_AGGREGATE_SUMMARY_PREFIXES = (
    "pandas/groupby", "pandas/pivot", "stats/", "anomaly/", "timeseries:", "comparison:",
)
_ROW_LEVEL_SUMMARY_PREFIXES = ("pandas/filter", "pandas/derive")


def _classify_evidence_level(result_summary: str) -> str:
    s = (result_summary or "").strip()
    if s.startswith(_AGGREGATE_SUMMARY_PREFIXES):
        return "AGGREGATE"
    if s.startswith(_ROW_LEVEL_SUMMARY_PREFIXES):
        return "ROW-LEVEL SAMPLE"
    return "UNKNOWN"  # e.g. raw SQL — cannot tell if it grouped/aggregated


def _format_assistant_turn_summary(state: MAEDAState, insights: list[Insight]) -> str:
    """
    Compact, structured recap of this turn for conversation_history — the
    resolved intent's fields verbatim (so a follow-up query has exact
    values to carry forward, not prose to re-parse) plus up to 2 key
    findings. Intentionally not the full report: history is capped to the
    last few messages (see IntentParserAgent._MAX_HISTORY_MESSAGES), so
    each entry needs to stay compact as a conversation grows.
    """
    intent = state.get("parsed_intent") or {}
    parts = [
        f"query_type={intent.get('query_type')}",
        f"target_metrics={intent.get('target_metrics') or []}",
        f"dimensions={intent.get('dimensions') or []}",
    ]
    if intent.get("filters"):
        parts.append(f"filters={intent['filters']}")
    if intent.get("time_range"):
        parts.append(f"time_range={intent['time_range']}")
    findings = [ins.finding for ins in insights[:2] if ins.finding]
    if findings:
        parts.append("key_findings=" + " / ".join(findings))
    return "; ".join(parts)


def _format_findings(results: list[dict]) -> str:
    if not results:
        return "No analysis results available."
    lines = []
    for r in results:
        method = r.get("method", "")
        step = r.get("step", "?")
        if r.get("failed"):
            reason = r.get("result_summary", "") or "; ".join(r.get("warnings") or [])
            lines.append(f"- Step {step} ({method}): FAILED — {reason}")
            continue
        summary = r.get("result_summary", "")
        level = _classify_evidence_level(summary)
        lines.append(f"- Step {step} ({method}) [{level}]: {summary}")
        detail = _extract_result_detail(r.get("result"))
        if detail:
            lines.append(f"  Data: {detail}")
    return "\n".join(lines)


def _extract_result_detail(result: Any) -> str:
    """
    Surface concrete numbers and caveats from a step's raw result payload.
    result_summary alone is a one-line paraphrase that drops exactly the
    figures (forecast values, trend stats, outlier values, tool-emitted
    caveats) an insight needs to cite — without them the Insight Agent has
    nothing real to ground a finding in and tends to invent
    plausible-sounding numbers.
    """
    # groupby/filter/sql_query/pivot results are a list of row records —
    # surface a bounded sample explicitly rather than dropping them (they
    # previously only reached the Insight Agent via the possibly-truncated
    # result_summary preview).
    if isinstance(result, list):
        if not result:
            return "0 rows"
        return f"{len(result)} rows, sample={json.dumps(result[:5], default=str)}"

    if not isinstance(result, dict):
        return ""
    parts = []
    if result.get("trend"):
        parts.append(f"trend={result['trend']}")
    if result.get("forecast"):
        fc = result["forecast"]
        parts.append(f"forecast_periods={fc.get('periods', [])[:3]}")
        if fc.get("caveat"):
            parts.append(f"forecast_caveat={fc['caveat']!r}")
    if result.get("caveat") and "forecast" not in result:
        parts.append(f"caveat={result['caveat']!r}")
    if result.get("significance_test"):
        parts.append(f"significance_test={result['significance_test']}")
    if "n_outliers" in result:
        parts.append(
            f"anomaly(method={result.get('method')}, column={result.get('column')}): "
            f"n_outliers={result.get('n_outliers')}, outlier_pct={result.get('outlier_pct')}, "
            f"sample_outlier_values={result.get('outlier_values', [])[:5]}"
        )
    if not parts:
        # Small scalar/dict result: include verbatim (bounded) so numbers
        # are available even for step types without a special-cased field.
        rendered = json.dumps(result, default=str)
        if len(rendered) < 400:
            parts.append(rendered)
        else:
            parts.append(rendered[:400] + "...(truncated)")
    return " | ".join(parts)


def _format_rag_chunks(chunks: list[dict], sources: list[dict]) -> str:
    if not chunks:
        return "No domain knowledge retrieved."
    lines = []
    for i, chunk in enumerate(chunks[:5]):
        content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        source = sources[i].get("source_file", "") if i < len(sources) else ""
        src_tag = f" [{source}]" if source else ""
        lines.append(f"{i + 1}. {content[:300]}{src_tag}")
    return "\n".join(lines)


def _format_charts_summary(charts: list[dict]) -> str:
    if not charts:
        return "No charts generated."
    return "\n".join(
        f"- {c.get('chart_type', 'chart')}: {c.get('title', '')} — {c.get('caption', '')}"
        for c in charts
        if c.get("chart_type") != "dashboard"
    ) or "Dashboard only."


def _format_quality_note(report: dict) -> str:
    if not report:
        return "No data quality information available."
    # DataQualityReport.to_dict() emits "quality_issues"; the old key "issues"
    # never existed in that dict, so issues silently never reached the report.
    issues = report.get("quality_issues") or report.get("issues") or []
    score = report.get("score")
    critical = report.get("has_critical_issues", False)
    parts = []
    if score is not None:
        parts.append(f"Quality score: {score:.2f}")
    if critical:
        parts.append("Critical issues were detected and cleaned.")
    if issues:
        parts.append(f"Issues: {'; '.join(_format_quality_issue(i) for i in issues[:5])}")
    return "; ".join(parts) if parts else "Data quality checks passed."


def _format_quality_issue(issue: Any) -> str:
    """Render one quality-issue dict as readable text for the report prompt."""
    if not isinstance(issue, dict):
        return str(issue)
    column = issue.get("column")
    kind = issue.get("issue", "unknown")
    detail = issue.get("detail", "")
    prefix = f"{column}: " if column else ""
    return f"{prefix}{kind}" + (f" ({detail})" if detail else "")


def _avg_confidence(insights: list[Insight]) -> float:
    if not insights:
        return 0.5
    return sum(i.confidence for i in insights) / len(insights)


def _parse_json(text: str) -> Any:
    """Extract JSON from LLM response (handles markdown fences)."""
    if "```" in text:
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines)
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON in: {text[:200]!r}")


def _rule_based_insights(
    results: list[dict],
    rag_chunks: list[dict],
    rag_sources: list[dict],
) -> list[Insight]:
    """Fallback: produce one insight per successful analysis step."""
    insights = []
    for r in results[:5]:
        summary = r.get("result_summary", "")
        if not summary:
            continue
        # Try to include a RAG snippet
        domain_ctx = ""
        source_refs: list[str] = []
        if rag_chunks:
            chunk = rag_chunks[0]
            domain_ctx = (chunk.get("content", "") if isinstance(chunk, dict) else str(chunk))[:200]
        if rag_sources:
            source_refs = [s.get("source_file", "") for s in rag_sources[:2] if s.get("source_file")]

        insights.append(Insight(
            finding=summary,
            evidence=[f"Step {r.get('step', '?')}: {summary}"],
            confidence=r.get("confidence", 0.7),
            domain_context=domain_ctx,
            impact=_score_to_impact(r.get("confidence", 0.7)),
            recommendation=f"Review {r.get('method', 'this analysis')} results and take action.",
            sources=source_refs,
        ))
    if not insights:
        insights.append(Insight(
            finding="Analysis completed with no significant findings.",
            evidence=[],
            confidence=0.5,
            domain_context="",
            impact="low",
            recommendation="Review data coverage and query specificity.",
            sources=[],
        ))
    return insights


def _rule_based_report(state: MAEDAState, insights: list[Insight]) -> str:
    """Fallback markdown report when the LLM is unavailable."""
    query = state.get("user_query", "")
    lines = [
        "# MAEDA Analysis Report",
        "",
        "## Executive Summary",
        f"Analysis completed for query: *{query}*",
        "",
        "## Key Findings",
    ]
    for ins in insights:
        lines.append(f"- **{ins.finding}** (confidence: {ins.confidence:.0%}, impact: {ins.impact})")
        if ins.recommendation:
            lines.append(f"  - *Recommendation:* {ins.recommendation}")

    lines += ["", "## Recommendations"]
    for ins in insights:
        if ins.recommendation:
            lines.append(f"- {ins.recommendation}")

    quality = state.get("data_quality_report") or {}
    if quality:
        lines += ["", "## Data Quality Notes", _format_quality_note(quality)]

    if state.get("charts"):
        n = len([c for c in state["charts"] if c.get("chart_type") != "dashboard"])
        lines += ["", "## Visualizations", f"{n} chart(s) generated."]

    return "\n".join(lines)
