"""
Eval runner — Phase 9.

9.1 EvalRunner: scores a completed MAEDAState against all metrics.
9.6 GoldenTestCase: structured test case with expected outputs.
9.8 RegressionDetector: compares two eval runs, alerts on drops > 5%.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.eval.metrics import (
    MetricScore,
    score_answer_relevance,
    score_chart_appropriateness,
    score_factual_accuracy,
    score_groundedness,
    score_intent_accuracy,
    score_step_success_rate,
    score_system_metrics,
    score_tool_selection,
)
from src.utils.logger import get_logger

logger = get_logger("maeda.eval.runner")


# ─── 9.6 Golden test case ────────────────────────────────────────────────────

@dataclass
class GoldenTestCase:
    id: str
    query: str
    query_type: str                 # descriptive|diagnostic|predictive|comparative|exploratory
    expected_metrics: list[str]     # e.g. ["revenue", "sales"]
    expected_dimensions: list[str]  # e.g. ["region", "quarter"]
    ground_truth: dict              # key facts the output must contain
    data_source: Optional[dict] = None
    tags: list[str] = field(default_factory=list)
    # Eval v2 Step 2d (docs/eval_v2_plan.md): ground truth for the rebuilt
    # tool_selection/chart_appropriateness metrics. None = not authored for
    # this case (score_tool_selection/score_chart_appropriateness treat
    # that leniently); [] = explicitly authored as "no specific tool/chart
    # expected" (distinct from "nobody set this field yet"). Values from
    # src.agents.analysis_agent.TOOL_REGISTRY and the chart_type strings
    # src.tools.chart_tool.recommend_chart can return.
    expected_tools: Optional[list[str]] = None
    expected_chart_types: Optional[list[str]] = None
    # Eval v2 Step 4: dev/test split, fixed at authoring time so a holdout
    # set can't be re-randomized (and silently re-peeked-at) across runs.
    split: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenTestCase":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── 9.1 Eval runner ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    run_id: str
    query: str
    scores: list[MetricScore]
    aggregate_score: float
    timestamp: float = field(default_factory=time.time)
    test_case_id: Optional[str] = None
    # 附录 R.3 / 附录 AO.1 / 附录 AP.2: diagnostic-only fields carried over
    # from MAEDAState so a persisted EvalResult can tell whether cleaning
    # ran, was blocked (needs_review), or never triggered for this run --
    # without them, that judgment can only be recovered by string-matching
    # decision_trace, which is exactly the fragile path R.3 was meant to
    # avoid. Optional and appended at the end (not inserted between
    # existing fields) so every existing positional-arg call site in this
    # repo keeps working unchanged, and so that constructing an EvalResult
    # from an older report dict that lacks these keys (via explicit
    # keyword args, as scripts/run_eval.py's _print_regressions does) still
    # works -- they simply default to None. Deliberately NOT folded into
    # _aggregate_score's weighting (see that function's weights dict): like
    # safe_refusal, a run being blocked_needs_review is neither good nor
    # bad on the quality axis this score measures, it just means the run
    # isn't a like-for-like comparison and D0 needs to know that before
    # counting it toward pass@k.
    cleaning_applied_level: Optional[str] = None
    cleaning_stop_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "scores": [s.to_dict() for s in self.scores],
            "aggregate_score": self.aggregate_score,
            "timestamp": self.timestamp,
            "test_case_id": self.test_case_id,
            "cleaning_applied_level": self.cleaning_applied_level,
            "cleaning_stop_reason": self.cleaning_stop_reason,
        }

    def score_by_metric(self, metric: str) -> Optional[float]:
        for s in self.scores:
            if s.metric == metric:
                return s.score
        return None


class EvalRunner:
    """
    Scores a completed MAEDAState against all eval metrics.
    Optionally cross-references a GoldenTestCase for ground-truth comparison.
    """

    def __init__(self, llm=None):
        self._llm = llm  # None → lazy init in metrics module

    async def score(
        self,
        state: dict,
        test_case: Optional[GoldenTestCase] = None,
        start_time: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> EvalResult:
        import uuid
        rid = run_id or str(uuid.uuid4())[:8]
        query = state.get("user_query", "")
        report = state.get("report") or ""
        analysis_results = state.get("analysis_results") or []
        rag_context = state.get("rag_context") or []
        parsed_intent = state.get("parsed_intent") or {}
        charts = state.get("charts") or []
        data_quality_report = state.get("data_quality_report")

        scores: list[MetricScore] = []

        # 9.2 / 9.3 LLM-as-judge. answer_relevance/groundedness don't depend
        # on test_case at all, so if run_eval_node already scored this same
        # state inside the graph (CLAUDE.md #8: eval runs on every
        # execution), reuse those scores instead of re-invoking the judge —
        # with EVAL_JUDGE_SAMPLES=3 (see metrics.py), scoring a harness case
        # twice meant 6 judge calls instead of 3 for zero additional signal.
        # Checked independently (not "both present or neither") since eval
        # v2 Step 2b split these into two unrelated calls -- one being
        # already-persisted and the other missing is a real, if unlikely,
        # state to handle correctly rather than re-invoking both.
        existing = state.get("eval_scores") or {}

        async def _relevance() -> MetricScore:
            if "answer_relevance" in existing:
                e = existing["answer_relevance"]
                return MetricScore("answer_relevance", e["score"], e["label"], e["reasoning"],
                                   valid=e.get("valid", True))
            return await score_answer_relevance(query, report, analysis_results, rag_context, llm=self._llm,
                                                data_quality_report=data_quality_report)

        async def _groundedness() -> MetricScore:
            if "groundedness" in existing:
                e = existing["groundedness"]
                return MetricScore("groundedness", e["score"], e["label"], e["reasoning"],
                                   valid=e.get("valid", True))
            return await score_groundedness(query, report, analysis_results, rag_context, llm=self._llm,
                                            data_quality_report=data_quality_report)

        rel, gnd = await asyncio.gather(_relevance(), _groundedness())
        scores.extend([rel, gnd])

        # 9.4 Factual accuracy
        ground_truth = test_case.ground_truth if test_case else None
        scores.append(score_factual_accuracy(report, analysis_results, ground_truth))

        # 9.5 Agent performance
        expected_type = test_case.query_type if test_case else None
        expected_metrics = test_case.expected_metrics if test_case else None
        expected_tools = test_case.expected_tools if test_case else None
        expected_chart_types = test_case.expected_chart_types if test_case else None
        scores.append(score_intent_accuracy(parsed_intent, expected_type, expected_metrics))
        scores.append(score_tool_selection(analysis_results, expected_tools))
        scores.append(score_step_success_rate(analysis_results))
        scores.append(score_chart_appropriateness(charts, expected_chart_types))

        # System metrics
        scores.extend(score_system_metrics(state, start_time))

        aggregate = _aggregate_score(scores)

        result = EvalResult(
            run_id=rid,
            query=query,
            scores=scores,
            aggregate_score=aggregate,
            test_case_id=test_case.id if test_case else None,
            cleaning_applied_level=state.get("cleaning_applied_level"),
            cleaning_stop_reason=state.get("cleaning_stop_reason"),
        )

        logger.info(
            "Eval run=%s aggregate=%.2f | %s",
            rid, aggregate,
            " ".join(f"{s.metric}={s.score:.2f}" for s in scores[:4]),
        )
        return result


def _aggregate_score(scores: list[MetricScore]) -> float:
    """Weighted average — only metrics with measured discriminative value
    get real weight (eval v2 Step 2d). error_rate/retry_count/token_cost/
    total_latency/step_success_rate were 33% of this table's weight while
    being constant at 1.00 across all 20 cases in the last audited run
    (phase_d_model_tiering.json) -- free points, not signal. They're
    operational/diagnostic: a healthy system's error_rate *should* be
    constantly 1.0, so folding it into a quality average only dilutes it.
    Kept at weight 0 (not deleted) so the exclusion is visible here rather
    than in a second, easy-to-miss set elsewhere -- they're still computed
    and reported, just not averaged in. `completeness` (never implemented)
    and `plan_efficiency` (measured nothing a genuinely better metric
    wouldn't already imply) are gone entirely, not just zero-weighted."""
    weights = {
        "answer_relevance": 3.0,
        "groundedness": 3.0,
        "factual_accuracy": 2.0,
        "intent_accuracy": 1.5,
        "tool_selection": 1.0,
        "chart_appropriateness": 0.5,
        # Operational/diagnostic -- see docstring above.
        "token_cost": 0.0,
        "retry_count": 0.0,
        "error_rate": 0.0,
        "total_latency": 0.0,
        "step_success_rate": 0.0,
        # Informational only — a safe refusal is neither good nor bad in
        # isolation, so it must not move the aggregate score in either
        # direction. It still appears in the report and in regression
        # detection (a refusal rate that changes across runs is worth
        # seeing), just not folded into this weighted average.
        "safe_refusal": 0.0,
    }
    total_w = total_wv = 0.0
    for s in scores:
        if not s.valid:
            # A metric that failed to score (judge unreachable after
            # retries) has no real measurement to average in -- weighting
            # it as a 0.5 placeholder silently distorted the aggregate in
            # whichever direction was wrong for that run (eval v2 Step 2a).
            # Skipped from the weighted average entirely; still reported
            # separately, see run_eval.py's _print_summary.
            continue
        w = weights.get(s.metric, 1.0)
        total_w += w
        total_wv += w * s.score
    return total_wv / total_w if total_w > 0 else 0.0


# ─── 9.8 Regression detector ─────────────────────────────────────────────────

@dataclass
class RegressionAlert:
    metric: str
    baseline: float
    current: float
    drop: float
    severity: str   # "critical" (>20%) | "warning" (>5%)


def detect_regressions(
    baseline: EvalResult,
    current: EvalResult,
    threshold_warn: float = 0.05,
    threshold_critical: float = 0.20,
) -> list[RegressionAlert]:
    """
    Compare two EvalResults. Return alerts for any metric that dropped
    more than threshold_warn (5% by default).
    """
    alerts: list[RegressionAlert] = []
    baseline_map = {s.metric: s.score for s in baseline.scores}
    current_map = {s.metric: s.score for s in current.scores}

    for metric, base_score in baseline_map.items():
        curr_score = current_map.get(metric)
        if curr_score is None:
            continue
        drop = base_score - curr_score
        if drop >= threshold_warn:
            severity = "critical" if drop >= threshold_critical else "warning"
            alerts.append(RegressionAlert(
                metric=metric,
                baseline=base_score,
                current=curr_score,
                drop=drop,
                severity=severity,
            ))
            logger.warning(
                "Regression detected: %s baseline=%.2f current=%.2f drop=%.2f [%s]",
                metric, base_score, curr_score, drop, severity,
            )

    # Also check aggregate
    agg_drop = baseline.aggregate_score - current.aggregate_score
    if agg_drop >= threshold_warn:
        severity = "critical" if agg_drop >= threshold_critical else "warning"
        alerts.append(RegressionAlert(
            metric="aggregate_score",
            baseline=baseline.aggregate_score,
            current=current.aggregate_score,
            drop=agg_drop,
            severity=severity,
        ))

    return alerts


# ─── 9.6 Golden suite loader ─────────────────────────────────────────────────

def load_golden_suite(path: Optional[str] = None) -> list[GoldenTestCase]:
    """
    Load golden test cases from JSON file -- the single source of truth
    (eval v2 Step 4: this used to silently fall back to a hardcoded Python
    copy that could and did drift out of sync with the JSON; now it fails
    loudly instead, since a fallback nobody notices is worse than an error).
    """
    p = Path(path or "tests/eval/test_suite.json")
    if not p.exists():
        raise FileNotFoundError(
            f"Golden suite JSON not found at {p} -- there is no fallback. "
            "If this path is wrong, pass the correct `path=`; the suite is "
            "not hardcoded anywhere else."
        )
    with open(p) as f:
        data = json.load(f)
    return [GoldenTestCase.from_dict(d) for d in data]
