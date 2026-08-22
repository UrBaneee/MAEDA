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
    not_applicable_metric,
    score_answer_relevance,
    score_chart_appropriateness,
    score_factual_accuracy,
    score_groundedness,
    score_intent_accuracy,
    score_step_success_rate,
    score_system_metrics,
    score_tool_selection,
)
from src.state.terminal_state import SUCCESS, resolve_terminal_state
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
    # E3 (附录 CU): None means "this run has no comparable aggregate", which
    # is a different fact from "the aggregate is zero" -- the same
    # distinction src/eval/trials.py::pass_at_k already makes when it
    # returns None rather than 0.0 for "not enough trials", and for the same
    # reason (0.0 would be indistinguishable from a real all-failures
    # result). A run that terminated in anything other than success has most
    # of the weighted metrics marked not_applicable, so whatever weighted
    # average survives is taken over a DIFFERENT metric set than a
    # successful run's and is not comparable with it. Reporting None forces
    # every consumer to decide explicitly what to do with a failed run
    # instead of silently averaging a near-zero into a suite mean.
    aggregate_score: Optional[float]
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
    # E2 (ECOSYSTEM_INTEGRATION_PLAN.md 附录 BQ, 附录 BO.5): same
    # three-step pattern as cleaning_applied_level above -- a top-level
    # scalar, not something D0's trial-variance analysis (src/eval/trials.py)
    # would have to recover by parsing decision_trace's free-text
    # "refine_intent" records. None = refine_intent never ran this round
    # (route_after_schema chose "profile"); True = it ran and the second
    # parse succeeded (state["parsed_intent"] is the refined version);
    # False = it ran but the LLM call raised and the pre-refine intent was
    # kept (src/agents/intent_parser.py's IntentParserAgent.refine).
    intent_refined: Optional[bool] = None
    # 定案 #15 / 阶段 3 收尾执行计划轮次 1 / 附录 CB.3.4: same top-level-scalar,
    # structurally-can't-reach-_aggregate_score pattern as intent_refined
    # above (_aggregate_score only ever consumes `scores: list[MetricScore]`
    # -- these two fields aren't in it, so there's nothing to guard). Unlike
    # intent_refined, these are run-level CONFIGURATION, not a per-run
    # result -- every case/trial of one script invocation carries the same
    # value, sourced from state["cleaner_mode"]/state["rag_mode"]
    # (src/state/graph_state.py, snapshotted from settings at
    # initial_state() time). Kept here in ADDITION to (not instead of)
    # scripts/run_eval.py's report-level `report["arm"]` + per-row
    # `meta["cleaner_mode"]`/`meta["rag_mode"]` -- CB.3.4's original
    # proposal argued for those two locations only and against also
    # duplicating the value into EvalResult (arm is config, not a pipeline
    # *outcome*, and folding it in blurs that distinction); this instance
    # was implemented per the lead's explicit instruction to do both. See
    # 附录 CD for that discrepancy note.
    cleaner_mode: Optional[str] = None
    rag_mode: Optional[str] = None
    # 附录 CK.3: the reason this trial must not be aggregated as an
    # on-arm result (None = it may be). Sourced from
    # state["rag_arm_invalid_reason"], written only by
    # src/graph/nodes.py::retrieve_knowledge_node and only under
    # MAEDA_RAG_MODE=force_on. Same "top-level scalar, structurally
    # cannot reach _aggregate_score" pattern as the fields above -- and
    # for a stronger reason than they have: this row's scores are not
    # merely uninteresting, they describe a run whose on-arm premise
    # didn't hold, so the exclusion happens in
    # src/eval/trials.py::is_applicable rather than by reweighting
    # anything here. The score itself is still computed and still
    # reported; what changes is that trials.py refuses to average it in.
    rag_arm_invalid_reason: Optional[str] = None
    # E3 (附录 CU): how the run ended, one of src/state/terminal_state.py's
    # five values, read off state rather than re-derived here. Same
    # top-level-scalar, structurally-cannot-reach-_aggregate_score pattern
    # as the fields above. `terminal_detail` is the sub-classification --
    # verbatim fallback.py error_class for anything that came from a
    # sub-system call, never a second taxonomy.
    terminal_state: Optional[str] = None
    terminal_detail: Optional[str] = None
    # E3: set when EVAL ITSELF failed for this run (as opposed to the run
    # failing). Distinct from every scores-based signal because when this is
    # set, `scores` is not trustworthy -- there may be none at all.
    eval_error: Optional[str] = None

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
            "intent_refined": self.intent_refined,
            "cleaner_mode": self.cleaner_mode,
            "rag_mode": self.rag_mode,
            "rag_arm_invalid_reason": self.rag_arm_invalid_reason,
            "terminal_state": self.terminal_state,
            "terminal_detail": self.terminal_detail,
            "eval_error": self.eval_error,
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

        # ─── E3 (附录 CU): terminal state → which metrics are applicable ───
        #
        # Read, never re-derived: src/state/terminal_state.py's
        # resolve_terminal_state returns handle_error_node's stored judgment
        # when there is one, and classifies only for a state that never went
        # through a terminal node (e.g. scripts/run_eval.py's
        # graph.ainvoke-raised path).
        terminal_state, terminal_detail = resolve_terminal_state(state)
        na = _not_applicable_metrics(state, terminal_state)

        def _guard(metric: str, fn) -> MetricScore:
            """One place where a metric is either skipped as not_applicable,
            scored, or -- E3's "eval 自身失败也要捕获分类" at metric
            granularity -- turned into a captured scoring failure instead of
            an exception that costs the caller every OTHER metric too."""
            reason = na.get(metric)
            if reason is not None:
                return not_applicable_metric(metric, reason)
            try:
                return fn()
            except Exception as exc:                       # noqa: BLE001
                logger.error("Metric %s raised while scoring: %s", metric, exc)
                return MetricScore(metric, 0.0, "error",
                                   f"scoring raised {type(exc).__name__}: {exc}", valid=False)

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

        # E3: when BOTH judge metrics are not_applicable there is nothing to
        # gather and, crucially, no judge call to pay for. This is what makes
        # putting the failure path through eval free (src/graph/builder.py's
        # handle_error → run_eval edge).
        if "answer_relevance" in na and "groundedness" in na:
            scores.extend([
                not_applicable_metric("answer_relevance", na["answer_relevance"]),
                not_applicable_metric("groundedness", na["groundedness"]),
            ])
        else:
            try:
                rel, gnd = await asyncio.gather(_relevance(), _groundedness())
            except Exception as exc:                       # noqa: BLE001
                # The judge helpers already convert their own failures into
                # valid=False scores; anything that still escapes (a bad
                # provider config, a cached-score dict missing a key) would
                # otherwise abort the whole EvalResult.
                logger.error("Judge metrics raised while scoring: %s", exc)
                detail = f"scoring raised {type(exc).__name__}: {exc}"
                rel = MetricScore("answer_relevance", 0.0, "error", detail, valid=False)
                gnd = MetricScore("groundedness", 0.0, "error", detail, valid=False)
            scores.extend([rel, gnd])

        # 9.4 Factual accuracy
        ground_truth = test_case.ground_truth if test_case else None
        scores.append(_guard("factual_accuracy",
                             lambda: score_factual_accuracy(report, analysis_results, ground_truth)))

        # 9.5 Agent performance
        expected_type = test_case.query_type if test_case else None
        expected_metrics = test_case.expected_metrics if test_case else None
        expected_tools = test_case.expected_tools if test_case else None
        expected_chart_types = test_case.expected_chart_types if test_case else None
        scores.append(_guard("intent_accuracy",
                             lambda: score_intent_accuracy(parsed_intent, expected_type, expected_metrics)))
        scores.append(_guard("tool_selection",
                             lambda: score_tool_selection(analysis_results, expected_tools)))
        scores.append(_guard("step_success_rate",
                             lambda: score_step_success_rate(analysis_results)))
        scores.append(_guard("chart_appropriateness",
                             lambda: score_chart_appropriateness(charts, expected_chart_types)))

        # System metrics. NEVER not_applicable: error_rate and safe_refusal
        # are the metrics a failed run exists to report, and latency/cost/
        # retries are real measurements of the run that actually happened
        # whether or not it produced an answer. This is the "只算适用指标"
        # half of E3 -- a failed run is scored on less, not on nothing.
        try:
            scores.extend(score_system_metrics(state, start_time))
        except Exception as exc:                           # noqa: BLE001
            logger.error("System metrics raised while scoring: %s", exc)
            scores.append(MetricScore("error_rate", 0.0, "error",
                                      f"scoring raised {type(exc).__name__}: {exc}", valid=False))

        # E3: a failed run's surviving metrics are a different (smaller) set
        # than a successful run's, so their weighted average is not on the
        # same scale -- see EvalResult.aggregate_score.
        aggregate = None if terminal_state != SUCCESS else _aggregate_score(scores)

        result = EvalResult(
            run_id=rid,
            query=query,
            scores=scores,
            aggregate_score=aggregate,
            test_case_id=test_case.id if test_case else None,
            cleaning_applied_level=state.get("cleaning_applied_level"),
            cleaning_stop_reason=state.get("cleaning_stop_reason"),
            intent_refined=state.get("intent_refined"),
            cleaner_mode=state.get("cleaner_mode"),
            rag_mode=state.get("rag_mode"),
            rag_arm_invalid_reason=state.get("rag_arm_invalid_reason"),
            terminal_state=terminal_state,
            terminal_detail=terminal_detail,
            eval_error=state.get("eval_error"),
        )

        logger.info(
            "Eval run=%s terminal=%s aggregate=%s | %s",
            rid, terminal_state,
            "n/a" if aggregate is None else f"{aggregate:.2f}",
            " ".join(f"{s.metric}={s.score:.2f}" for s in scores[:4]),
        )
        return result


# ─── E3: which metrics a failed run may still be scored on ───────────────────

# The report-quality metrics. On a run that did not complete these are
# not_applicable UNCONDITIONALLY, even when a `report` string happens to
# exist in state -- a guardrail-refused run has one, and it is precisely the
# text the pipeline decided not to deliver. Asking "how good is this answer"
# about an answer that was never given produces a number that looks like a
# quality measurement and is not one (and, for the two judge metrics, pays
# real money to produce it).
_REPORT_QUALITY_METRICS = ("answer_relevance", "groundedness", "factual_accuracy")

# Every other scorable metric, with the state key holding the artifact it
# scores. On a failed run these ARE still scored whenever that artifact
# exists: a run that parsed an intent correctly and then died at profiling
# gives a real intent_accuracy measurement, and throwing it away would lose
# signal E3 explicitly wants kept ("只算适用指标", not "算零个指标").
_METRIC_ARTIFACT = {
    "intent_accuracy": "parsed_intent",
    "tool_selection": "analysis_results",
    "step_success_rate": "analysis_results",
    "chart_appropriateness": "charts",
}


def _not_applicable_metrics(state: dict, terminal_state: str) -> dict[str, str]:
    """
    metric → reason, for the metrics this run must not be scored on.

    Returns EMPTY for a successful run. That is deliberate and is the reason
    E3 cannot shift any existing 阶段 4 baseline number: on the success path
    every metric is computed exactly as it was before this change, including
    the cases where an artifact is missing (a successful run with no charts
    still gets score_chart_appropriateness' own lenient handling, not a
    not_applicable). E3 asked about failed runs; widening the rule to
    successful ones would be a silent metric-definition change of the kind
    附录 BO.5 exists to warn about.
    """
    if terminal_state == SUCCESS:
        return {}
    na = {m: f"run terminated as {terminal_state}" for m in _REPORT_QUALITY_METRICS}
    for metric, artifact in _METRIC_ARTIFACT.items():
        if not state.get(artifact):
            na[metric] = f"run terminated as {terminal_state} before producing {artifact}"
    return na


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

    # Also check aggregate. E3 (附录 CU): either side may be None now ("this
    # run has no comparable aggregate" -- see EvalResult.aggregate_score).
    # Comparing against a missing aggregate would either raise or, if it were
    # coerced to 0.0, report a spurious 100% regression every time a run
    # failed -- which is the "credible wrong answer" shape 附录 CI.2 warns
    # about, aimed at regression detection instead of at the RAG arm.
    if baseline.aggregate_score is None or current.aggregate_score is None:
        return alerts
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
