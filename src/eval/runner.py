"""
Eval runner — Phase 9.

9.1 EvalRunner: scores a completed MAEDAState against all metrics.
9.6 GoldenTestCase: structured test case with expected outputs.
9.8 RegressionDetector: compares two eval runs, alerts on drops > 5%.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.eval.metrics import (
    MetricScore,
    score_chart_appropriateness,
    score_factual_accuracy,
    score_intent_accuracy,
    score_plan_efficiency,
    score_relevance_and_groundedness,
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
    data_source: Optional[str] = None
    tags: list[str] = field(default_factory=list)

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

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "scores": [s.to_dict() for s in self.scores],
            "aggregate_score": self.aggregate_score,
            "timestamp": self.timestamp,
            "test_case_id": self.test_case_id,
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

        scores: list[MetricScore] = []

        # 9.2 / 9.3 LLM-as-judge (single call for both)
        rel, gnd = await score_relevance_and_groundedness(
            query, report, analysis_results, rag_context, llm=self._llm
        )
        scores.extend([rel, gnd])

        # 9.4 Factual accuracy
        ground_truth = test_case.ground_truth if test_case else None
        scores.append(score_factual_accuracy(report, analysis_results, ground_truth))

        # 9.5 Agent performance
        expected_type = test_case.query_type if test_case else None
        expected_metrics = test_case.expected_metrics if test_case else None
        scores.append(score_intent_accuracy(parsed_intent, expected_type, expected_metrics))
        scores.append(score_tool_selection(analysis_results))
        scores.append(score_plan_efficiency(analysis_results))
        scores.append(score_chart_appropriateness(charts))

        # System metrics
        scores.extend(score_system_metrics(state, start_time))

        aggregate = _aggregate_score(scores)

        result = EvalResult(
            run_id=rid,
            query=query,
            scores=scores,
            aggregate_score=aggregate,
            test_case_id=test_case.id if test_case else None,
        )

        logger.info(
            "Eval run=%s aggregate=%.2f | %s",
            rid, aggregate,
            " ".join(f"{s.metric}={s.score:.2f}" for s in scores[:4]),
        )
        return result


def _aggregate_score(scores: list[MetricScore]) -> float:
    """Weighted average — quality metrics weighted higher than system metrics."""
    weights = {
        "answer_relevance": 3.0,
        "groundedness": 3.0,
        "factual_accuracy": 2.0,
        "completeness": 1.5,
        "intent_accuracy": 1.5,
        "tool_selection": 1.0,
        "plan_efficiency": 0.5,
        "chart_appropriateness": 0.5,
        "token_cost": 0.3,
        "retry_count": 0.5,
        "error_rate": 2.0,
        "total_latency": 0.3,
    }
    total_w = total_wv = 0.0
    for s in scores:
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
    """Load golden test cases from JSON file. Falls back to built-in suite."""
    p = Path(path or "tests/eval/test_suite.json")
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return [GoldenTestCase.from_dict(d) for d in data]
    return _builtin_golden_suite()


def _builtin_golden_suite() -> list[GoldenTestCase]:
    """20 built-in golden test cases covering all 5 query types."""
    return [
        # ── Descriptive ───────────────────────────────────────────────────────
        GoldenTestCase("D01", "Show total sales by region",
                       "descriptive", ["sales"], ["region"], {}, tags=["descriptive"]),
        GoldenTestCase("D02", "What is the average order value per product category?",
                       "descriptive", ["order_value"], ["category"],
                       {}, tags=["descriptive"]),
        GoldenTestCase("D03", "How many customers do we have per country?",
                       "descriptive", ["customers"], ["country"],
                       {}, tags=["descriptive"]),
        GoldenTestCase("D04", "What are the top 10 products by revenue?",
                       "descriptive", ["revenue"], ["product"],
                       {}, tags=["descriptive", "top_n"]),
        GoldenTestCase("D05", "Show monthly order volume for the last 12 months",
                       "descriptive", ["orders"], ["month"],
                       {}, tags=["descriptive", "time_series"]),

        # ── Diagnostic ────────────────────────────────────────────────────────
        GoldenTestCase("DG01", "Why did revenue drop in Q3?",
                       "diagnostic", ["revenue"], ["quarter"],
                       {}, tags=["diagnostic"]),
        GoldenTestCase("DG02", "What caused the spike in customer churn last month?",
                       "diagnostic", ["churn"], ["month"],
                       {}, tags=["diagnostic"]),
        GoldenTestCase("DG03", "Why is the North region underperforming?",
                       "diagnostic", ["sales"], ["region"],
                       {}, tags=["diagnostic"]),
        GoldenTestCase("DG04", "What factors correlate with high customer lifetime value?",
                       "diagnostic", ["ltv"], [],
                       {}, tags=["diagnostic", "correlation"]),

        # ── Comparative ───────────────────────────────────────────────────────
        GoldenTestCase("C01", "Compare sales performance across Q1, Q2, Q3, Q4",
                       "comparative", ["sales"], ["quarter"],
                       {}, tags=["comparative"]),
        GoldenTestCase("C02", "How does conversion rate differ by marketing channel?",
                       "comparative", ["conversion_rate"], ["channel"],
                       {}, tags=["comparative"]),
        GoldenTestCase("C03", "Compare average order value between new and returning customers",
                       "comparative", ["order_value"], ["customer_type"],
                       {}, tags=["comparative"]),
        GoldenTestCase("C04", "Which product categories have the highest and lowest margins?",
                       "comparative", ["margin"], ["category"],
                       {}, tags=["comparative"]),

        # ── Predictive ────────────────────────────────────────────────────────
        GoldenTestCase("P01", "What will revenue look like next quarter based on current trends?",
                       "predictive", ["revenue"], ["quarter"],
                       {}, tags=["predictive"]),
        GoldenTestCase("P02", "Forecast customer churn for the next 30 days",
                       "predictive", ["churn"], ["day"],
                       {}, tags=["predictive"]),
        GoldenTestCase("P03", "Which customers are most likely to upgrade their plan?",
                       "predictive", ["upgrade_probability"], ["customer"],
                       {}, tags=["predictive"]),

        # ── Exploratory ───────────────────────────────────────────────────────
        GoldenTestCase("E01", "Give me an overview of this dataset",
                       "exploratory", [], [],
                       {}, tags=["exploratory"]),
        GoldenTestCase("E02", "Are there any anomalies or outliers in the sales data?",
                       "exploratory", ["sales"], [],
                       {}, tags=["exploratory", "anomaly"]),
        GoldenTestCase("E03", "What patterns exist in customer purchasing behavior?",
                       "exploratory", [], ["customer"],
                       {}, tags=["exploratory"]),
        GoldenTestCase("E04", "Explore the relationship between marketing spend and revenue",
                       "exploratory", ["revenue", "marketing_spend"], [],
                       {}, tags=["exploratory", "correlation"]),
    ]
