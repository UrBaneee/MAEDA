"""
D0 trial statistics — 阶段 3 / 附录 AO/AP/AS/AT/AU/AV, block 3.

Computed against scripts/run_eval.py::run_trials()'s multi-trial report
shape: `per_trial: list[trial][case] -> {"test_case_id", "eval_result",
"meta"}` (`eval_result` is `EvalResult.to_dict()`).

阶段 3's own D0 spec is explicit that pass@k is "只用于二元指标" (binary
metrics only) — this module treats two different "how much did this vary
across trials" questions as genuinely different statistics, not one
generic variance number applied everywhere:

  binary metrics    every metric this harness reports except
                    safe_refusal (see is_binary_metric) — pass_at_k()/
                    pass_hat_k(), the standard unbiased Codex/HumanEval-
                    paper estimators, computed from each MetricScore's
                    existing `label` field (label == "pass" is success)
                    rather than inventing a second binary threshold
                    alongside it (附录 AO.3's recommendation, adopted
                    here as instructed).

  continuous metrics  the underlying 0..1 `score`, independent of label
                    — reuses src/eval/noise.py's summarize(), the exact
                    function docs/noise_floor.md's real 8-trial noise
                    measurement already used, not a second statistics
                    implementation living here.

Trial applicability (附录 R.3 / 附录 AO.1 / 附录 AP.2's whole reason for
existing, plus 附录 CK.3's second condition — see
not_applicable_reason() for both): an inapplicable trial is excluded
from both the binary and continuous aggregation above, not just one of
them — a report generated on data the pipeline itself flagged as needing
manual review, or an on-arm trial whose RAG retrieval silently degraded,
isn't a fair comparison point on ANY metric, not only the ones this
module happens to call "binary".
"""
from __future__ import annotations

from typing import Optional

from src.eval.noise import summarize

# 附录 AO.3: every metric this harness reports uses the pass/warn/fail
# vocabulary (src/eval/metrics.py's MetricScore.label) EXCEPT
# safe_refusal, which is deliberately "info"-only (a refusal is neither
# good nor bad in isolation on its own -- see runner.py's
# _aggregate_score docstring, weight=0 there for the same reason) and so
# has no pass/fail concept to collapse into a binary success/failure
# outcome. Collapsing it anyway would produce a "pass@k" number with no
# meaning (success at what?).
_NON_BINARY_METRICS = frozenset({"safe_refusal"})

# 附录 R.3 / 附录 AP.2: the one cleaning_applied_level value that makes a
# trial's numbers non-comparable across an on/off or trial-to-trial
# comparison. See is_applicable()'s docstring for why this is
# deliberately narrower than a literal "!= full" filter.
_INAPPLICABLE_CLEANING_LEVEL = "blocked_needs_review"

# E3 (执行顺序表轮次 5, 附录 CU) landed and uses this constant literally:
# src/eval/metrics.py::not_applicable_metric imports it as the LABEL for a
# metric a failed run must not be scored on. It is deliberately NOT a
# row-level exclusion for such runs -- see not_applicable_reason below.
NOT_APPLICABLE = "not_applicable"


def is_binary_metric(metric: str) -> bool:
    return metric not in _NON_BINARY_METRICS


def is_applicable(eval_result: dict) -> bool:
    """
    Whether a trial's row counts toward pass@k / continuous-metric
    aggregation at all.

    Deliberately keyed on `cleaning_applied_level == "blocked_needs_review"`
    specifically, not a broader "!= full" reading (the task description
    for this block used "!= full" as shorthand for "isn't a normal run").
    The three values of cleaning_applied_level (阶段 3's D0 row; the
    derivation is nodes.py::_cleaning_applied_level) are:

      "full"                 cleaning triggered and completed a round.
      "blocked_needs_review" cleaning was triggered but a step's risk
                              tier (or the server) blocked it before
                              completing -- the report was generated on
                              data the pipeline itself flagged as needing
                              manual review. THIS is the comparability
                              problem 附录 R.3 exists to let D0 detect.
      "none"                 cleaning never triggered because the data
                              was already clean by the quality gate's own
                              judgment. This is a completely ordinary,
                              fully comparable outcome for any golden
                              case whose data doesn't need cleaning --
                              excluding it under a literal "!= full"
                              reading would silently drop every
                              already-clean case from pass@k's
                              denominator, which is not what 附录 R.3 was
                              written to protect against.

    Only "blocked_needs_review" is excluded here (confirmed by 附录 AX.1
    against a literal "!= full" reading).

    附录 CK.3 adds a SECOND, independent exclusion —
    `rag_arm_invalid_reason` — see not_applicable_reason() below.
    """
    return not_applicable_reason(eval_result) is None


def not_applicable_reason(eval_result: dict) -> Optional[str]:
    """
    Why this trial doesn't count, or None if it does. is_applicable() is
    the boolean view of this function; summarize_case() reports the
    reasons so an exclusion is never just a shrinking denominator.

    Two independent conditions, deliberately kept as one function:

      "blocked_needs_review"   附录 R.3 / AP.2 / AX.1 — the report was
                               generated on data the pipeline itself
                               flagged as needing manual review.
      `rag_arm_invalid_reason` 附录 CK.3 — the run happened under
                               MAEDA_RAG_MODE=force_on, but the retrieval
                               was a fallback, a hard failure, or ran at
                               a weaker tier than the arm asserts
                               (src/graph/nodes.py::retrieve_knowledge_node,
                               settings.maeda_rag_expected_retrieval_mode).
                               Scoring it anyway would average an off-arm
                               run into the on-arm mean, intermittently
                               and invisibly — 附录 CI.2's "a credible
                               wrong answer" aimed straight at TB6's
                               "concede RAG isn't worth it" exit.

    Both are "the pipeline completed, but this run is not a fair
    comparison point", which is why they share one gate instead of the
    second one growing its own parallel filter somewhere else.

    E3 (附录 CU) DELIBERATELY DID NOT ADD A THIRD CONDITION HERE for
    `terminal_state != "success"`, even though that is superficially the
    same shape. A failed run is excluded from the metrics it cannot support
    ONE METRIC AT A TIME, via valid=False (src/eval/metrics.py::
    not_applicable_metric), precisely so that the metrics it CAN support
    still count -- above all error_rate and safe_refusal, whose whole job is
    to count failures. Dropping the whole row here would remove those runs
    from error_rate's own denominator, i.e. the measured failure rate would
    IMPROVE as the system failed more, which is the exact inversion 附录
    CI.2 calls a "credible wrong answer". The two conditions that ARE here
    describe runs that completed normally but are not comparable; a failed
    run is not incomparable, it is a result.
    """
    if eval_result.get("cleaning_applied_level") == _INAPPLICABLE_CLEANING_LEVEL:
        return _INAPPLICABLE_CLEANING_LEVEL
    rag_reason = eval_result.get("rag_arm_invalid_reason")
    if rag_reason:
        return f"rag_arm_invalid: {rag_reason}"
    return None


def pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    """
    Unbiased estimator of "at least one of k randomly-chosen trials (out
    of n total, c of which succeeded) succeeds" — the standard pass@k
    estimator (Chen et al. 2021, "Evaluating Large Language Models
    Trained on Code" — the Codex/HumanEval paper), in the numerically
    stable product form rather than raw binomial coefficients (those
    overflow/lose precision fast once n reaches the dozens, which
    multi-trial D0 runs will realistically do):

        pass@k = 1 - C(n-c, k) / C(n, k)
               = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)

    Returns None (not 0.0) if n < k — "not enough trials to answer this
    question" is a different fact from "the answer is zero", and 0.0
    would be indistinguishable from a real all-failures result.
    """
    if n < k:
        return None
    if n - c < k:  # fewer failures than k -> every k-subset has >=1 success
        return 1.0
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


def pass_hat_k(n: int, c: int, k: int) -> Optional[float]:
    """
    Unbiased estimator of "all k randomly-chosen trials (out of n total,
    c of which succeeded) succeed" — pass^k, the statistic complementary
    to pass@k: pass@k measures best-of-k capability (does it succeed at
    least once given k tries), pass^k measures k-shot reliability (does
    it succeed on EVERY one of k tries). Same derivation/stability
    rationale as pass_at_k:

        pass^k = C(c, k) / C(n, k) = prod_{i=0}^{k-1} (c-i) / (n-i)

    Returns None if n < k (see pass_at_k). Returns 0.0 (not None) if
    c < k — "fewer successes than k total" legitimately means zero
    probability that k of them are all successes; that's a real answer,
    not a missing one.
    """
    if n < k:
        return None
    if c < k:
        return 0.0
    prob_all_succeed = 1.0
    for i in range(k):
        prob_all_succeed *= (c - i) / (n - i)
    return prob_all_succeed


# ─── Report-shaped aggregation ─────────────────────────────────────────────────

def _rows_by_case(per_trial: list[list[dict]]) -> dict[str, list[dict]]:
    """Flatten run_trials()'s trial-major `per_trial` into case-major:
    {case_id: [row_from_trial_0, row_from_trial_1, ...]}."""
    by_case: dict[str, list[dict]] = {}
    for trial_rows in per_trial:
        for row in trial_rows:
            by_case.setdefault(row["test_case_id"], []).append(row)
    return by_case


def _binary_success(score_dict: dict) -> Optional[bool]:
    """None means this metric couldn't actually be scored this trial
    (valid=False -- e.g. judge unreachable after retries, per
    src/eval/metrics.py's MetricScore.valid docstring) -- excluded from
    both n and c, the same spirit as runner.py's _aggregate_score
    skipping invalid entries rather than counting a scoring failure as a
    real "fail"."""
    if not score_dict.get("valid", True):
        return None
    return score_dict.get("label") == "pass"


def summarize_case(case_id: str, rows: list[dict], k_values: list[int]) -> dict:
    """
    `rows`: this one case's row from every trial (any order; typically
    `_rows_by_case(per_trial)[case_id]`).
    """
    n_trials = len(rows)
    applicable_rows = [r for r in rows if is_applicable(r["eval_result"])]
    n_not_applicable = n_trials - len(applicable_rows)
    # 附录 CK.3: now that there are two independent exclusion conditions,
    # a bare count is no longer self-explanatory -- and an exclusion that
    # only shows up as a smaller denominator is the same class of problem
    # as the silent degradation it exists to catch.
    excluded_reasons: dict[str, int] = {}
    for r in rows:
        reason = not_applicable_reason(r["eval_result"])
        if reason:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    all_metrics = sorted({s["metric"] for r in rows for s in r["eval_result"]["scores"]})

    binary: dict[str, dict] = {}
    continuous: dict[str, dict] = {}

    for metric in all_metrics:
        per_trial_scores = []
        for r in applicable_rows:
            for s in r["eval_result"]["scores"]:
                if s["metric"] == metric:
                    per_trial_scores.append(s)
                    break

        summary = summarize(f"{case_id}/{metric}", [s["score"] for s in per_trial_scores])
        continuous[metric] = {
            "n_applicable": len(applicable_rows),
            "n_excluded_not_applicable": n_not_applicable,
            "summary": summary.to_dict() if summary else None,
        }

        if is_binary_metric(metric):
            successes = [_binary_success(s) for s in per_trial_scores]
            n_scored = sum(1 for x in successes if x is not None)
            c = sum(1 for x in successes if x is True)
            binary[metric] = {
                "n_applicable": len(applicable_rows),
                "n_excluded_not_applicable": n_not_applicable,
                "n_scored": n_scored,
                "n_excluded_invalid_score": len(successes) - n_scored,
                "c": c,
                "pass_at_k": {k: pass_at_k(n_scored, c, k) for k in k_values},
                "pass_hat_k": {k: pass_hat_k(n_scored, c, k) for k in k_values},
            }

    # E3 (附录 CU): how this case's trials ended, tallied. Reported, never
    # used to include or exclude anything (see not_applicable_reason's
    # docstring) -- it exists so "阶段 4 这个 case 的 pass@k 低" can be read
    # against "它有 3/8 次是 mcp_error" instead of being attributed to model
    # quality by default. A `None` key is a trial from a report written
    # before E3, which recorded no terminal state at all.
    terminal_states: dict[Optional[str], int] = {}
    for r in rows:
        ts = r["eval_result"].get("terminal_state")
        terminal_states[ts] = terminal_states.get(ts, 0) + 1

    return {
        "test_case_id": case_id,
        "n_trials": n_trials,
        "n_applicable": len(applicable_rows),
        "n_not_applicable": n_not_applicable,
        "not_applicable_reasons": excluded_reasons,
        "terminal_states": terminal_states,
        "binary": binary,
        "continuous": continuous,
    }


def summarize_report(per_trial: list[list[dict]], k_values: Optional[list[int]] = None) -> dict:
    """
    Top-level entry point: `per_trial` is exactly
    scripts/run_eval.py::run_trials()'s return value (or, equivalently,
    `[t["cases"] for t in report["per_trial"]]` from a saved multi-trial
    report JSON — see scripts/summarize_trials.py for that plumbing).

    `k_values` defaults to `[1]` (single-sample pass@1, the only k that
    is always defined once trials >= 1) — pass a case's own trial count
    (or any smaller k) explicitly to get best-of-k / k-shot-reliability
    numbers.
    """
    k_values = list(k_values) if k_values else [1]
    by_case = _rows_by_case(per_trial)
    per_case = {cid: summarize_case(cid, rows, k_values) for cid, rows in by_case.items()}

    suite_binary: dict[str, dict] = {}
    all_binary_metrics = sorted({m for c in per_case.values() for m in c["binary"]})
    for metric in all_binary_metrics:
        suite_binary[metric] = {}
        for k in k_values:
            pk_vals = [
                c["binary"][metric]["pass_at_k"][k] for c in per_case.values()
                if metric in c["binary"] and c["binary"][metric]["pass_at_k"][k] is not None
            ]
            phk_vals = [
                c["binary"][metric]["pass_hat_k"][k] for c in per_case.values()
                if metric in c["binary"] and c["binary"][metric]["pass_hat_k"][k] is not None
            ]
            suite_binary[metric][k] = {
                "mean_pass_at_k": sum(pk_vals) / len(pk_vals) if pk_vals else None,
                "mean_pass_hat_k": sum(phk_vals) / len(phk_vals) if phk_vals else None,
                "n_cases_defined": len(pk_vals),
                "n_cases_total": len(per_case),
            }

    suite_continuous: dict[str, dict] = {}
    all_continuous_metrics = sorted({m for c in per_case.values() for m in c["continuous"]})
    for metric in all_continuous_metrics:
        # Mean-of-per-case-means, not a single pooled summarize() over
        # every (case, trial) value -- docs/noise_floor.md's own
        # measurement explicitly warns pooling mixes real between-case
        # differences with actual noise and is "NOT the noise floor",
        # only a spread sanity check. Reusing that same distinction here
        # rather than re-deriving it.
        case_means = [
            c["continuous"][metric]["summary"]["mean"] for c in per_case.values()
            if metric in c["continuous"] and c["continuous"][metric]["summary"] is not None
        ]
        suite_continuous[metric] = {
            "mean_of_case_means": sum(case_means) / len(case_means) if case_means else None,
            "n_cases_with_variance_data": len(case_means),
            "n_cases_total": len(per_case),
        }

    return {
        "k_values": k_values,
        "n_cases": len(per_case),
        "per_case": per_case,
        "suite_binary": suite_binary,
        "suite_continuous": suite_continuous,
    }
