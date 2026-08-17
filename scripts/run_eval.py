"""
Eval harness — ties together Phase 9 (EvalRunner, golden suite, regression detector)
with the compiled LangGraph pipeline. This is the `tests/eval/run_eval.py` referenced
in DEV_SPEC.md, implemented as a CLI script.

Runs every golden test case end-to-end through graph.invoke(), scores the resulting
MAEDAState with EvalRunner, and writes a timestamped JSON report to logs/eval_runs/
for later regression comparison.

Usage:
    poetry run python scripts/run_eval.py                     # dev split (default)
    poetry run python scripts/run_eval.py --split test         # holdout -- must be explicit
    poetry run python scripts/run_eval.py --limit 3            # smoke test
    poetry run python scripts/run_eval.py --case DG01 --case C02
    poetry run python scripts/run_eval.py --compare logs/eval_runs/eval_1234567890.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from src.eval.metrics import MetricScore
from src.eval.runner import EvalResult, EvalRunner, GoldenTestCase, detect_regressions, load_golden_suite
from src.graph.builder import build_graph
from src.state.graph_state import initial_state

REPORT_DIR = Path("logs/eval_runs")

# Eval v2 Step 4: data source and data-mismatch status used to live here as
# two separate hardcoded dicts (CASE_DATA_SOURCES, KNOWN_DATA_MISMATCH),
# duplicating what's now authored directly on each case in
# tests/eval/test_suite.json (`data_source`, and a `"data_mismatch"` tag) --
# see docs/eval_v2_plan.md Step 4. Consolidated so a new case only needs to
# be added in one place.
#
# Note on which cases carry the data_mismatch tag: the Analysis Planner can
# now write real cross-table JOINs against a SQL source (Phase B #1/#2 —
# see docs/eval_report.md) when it's told the connection string and other
# tables' schemas, so cases needing a join across ecommerce_orders.db
# tables are no longer structurally unanswerable. D02 (order value by
# category) is fully resolved this way. C03 (new vs returning customers)
# still executes a real join but there's no literal "new/returning" flag in
# the data — the Planner substitutes the closest available dimension
# (customer segment) instead, which answers a related but not identical
# question, so it stays flagged. DG04 (customer LTV) and P03 (upgrade
# likelihood) remain genuinely unanswerable: no LTV column exists anywhere,
# and P03 needs plan-change history that was never collected. This is
# intentional: it surfaces real data-coverage gaps as a finding rather than
# papering over them by rewriting the golden queries.


async def run_one_case(tc: GoldenTestCase, graph, eval_runner: EvalRunner) -> tuple[EvalResult, dict]:
    state = initial_state(tc.query)
    if tc.data_source:
        state["data_sources"] = [dict(tc.data_source)]

    t0 = time.time()
    try:
        result_state = await graph.ainvoke(state)
        run_error = result_state.get("error")
    except Exception as exc:
        result_state = state
        run_error = f"graph.ainvoke raised: {exc}"
        result_state["error"] = run_error
        result_state["error_type"] = "pipeline_error"  # an uncaught exception is never a safe refusal
    elapsed = time.time() - t0

    eval_result = await eval_runner.score(result_state, test_case=tc, start_time=t0, run_id=tc.id)
    meta = {
        "elapsed_s": round(elapsed, 2),
        "guardrail_passed": result_state.get("guardrail_passed"),
        "current_phase": result_state.get("current_phase"),
        "error": run_error,
        "error_type": result_state.get("error_type"),
        "mcp_modes": sorted({c.get("mode", "mcp") for c in (result_state.get("mcp_call_log") or [])}),
        "data_mismatch": "data_mismatch" in tc.tags,
        # D0 (阶段 3 / 附录 AQ/AS): eval_runner.score() is called with
        # run_id=tc.id above (the golden case id, e.g. "D01" -- same
        # across every trial of the same case), which is NOT the actual
        # per-invocation graph run_id RunStore persists under
        # (src/state/graph_state.py's initial_state() uuid4). Without
        # this, nothing in the eval report lets a user cross-reference a
        # specific trial's row back to its RunStore-persisted
        # decision_trace/cleaning_applied_level -- the block-0 work
        # (附录 AQ.1) becomes unreachable from the eval harness's own
        # output. eval_result already carries cleaning_applied_level/
        # cleaning_stop_reason directly (no RunStore lookup needed for
        # D0's own pass@k accounting -- see src/eval/trials.py), but the
        # real run_id is still worth surfacing here for manual debugging.
        "run_id": result_state.get("run_id"),
    }
    return eval_result, meta


async def run_trials(
    suite: list[GoldenTestCase], graph, eval_runner: EvalRunner,
    trials: int, concurrency: int,
) -> list[list[dict]]:
    """
    D0 (阶段 3 / 附录 AW block 2): run `suite` `trials` times, every
    (trial, case) pair gated through ONE shared `asyncio.Semaphore
    (concurrency)` -- this is scripts/measure_noise.py's own
    `run_full_mode` concurrency pattern reused verbatim (per 附录 AV's
    instruction to reuse it rather than invent a second one), not a new
    scheduling scheme.

    Returns each trial's rows kept SEPARATE (`list[trial][case] -> dict`,
    never merged or averaged) -- this function does no statistics of any
    kind. pass@k / pass^k / variance across trials live in
    src/eval/trials.py (block 3), a deliberately separate module so this
    one stays a pure runner.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(trial_idx: int, tc: GoldenTestCase):
        async with sem:
            eval_result, meta = await run_one_case(tc, graph, eval_runner)
            return trial_idx, tc.id, eval_result, meta

    tasks = [_one(t, tc) for t in range(trials) for tc in suite]
    results = await asyncio.gather(*tasks)

    per_trial: list[list[dict]] = [[] for _ in range(trials)]
    for trial_idx, cid, eval_result, meta in results:
        per_trial[trial_idx].append(
            {"test_case_id": cid, "eval_result": eval_result.to_dict(), "meta": meta}
        )
    return per_trial


async def main():
    parser = argparse.ArgumentParser(description="Run MAEDA eval harness against the golden suite")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Eval v2 Step 4: which split to run. Defaults to dev -- the holdout "
                             "'test' split must be requested explicitly so it can't be peeked at "
                             "by accident. See data/eval_split_manifest.json. NOTE (D0, 阶段 3): "
                             "the plan's shorthand '--trials/--suite/--concurrency' triad maps onto "
                             "this script as --trials/--concurrency (new, below) PLUS this existing "
                             "--split/--case/--limit trio for case selection -- see 附录 AW block 2 "
                             "for why a separate --suite flag duplicating --split was rejected "
                             "(exactly the flag-overloads-two-dimensions problem 附录 AO.3 flagged, "
                             "just inverted: a second flag with the same job, not one flag with two "
                             "jobs).")
    parser.add_argument("--trials", type=int, default=1,
                        help="D0 (阶段 3): repeat the selected suite this many times. Default 1 -- "
                             "every existing invocation of this script keeps its exact prior "
                             "behavior, output report shape, and API cost unless this is raised "
                             "explicitly (附录 AP.3's rule: multi-trial is an opt-in, never a "
                             "surprise 8x bill). Each trial's rows are kept separate in the output "
                             "report, never averaged -- use src/eval/trials.py to compute "
                             "pass@k/pass^k/variance across them.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent (trial, case) runs in flight, via one shared "
                             "asyncio.Semaphore across every trial x case pair (same pattern as "
                             "scripts/measure_noise.py). Default 1 (fully serial, and takes the "
                             "exact pre-D0 sequential code path with live per-case progress "
                             "output). RAISE THIS WITH CAUTION: the Data Cleaner MCP server applies "
                             "NO internal rate limiting, backoff coordination, or concurrency "
                             "throttle to its own LLM calls (附录 AU.9, verified against "
                             "agent/llm_provider.py) -- every concurrent trial's planner call goes "
                             "straight to the same account, unmediated. Raising --concurrency "
                             "alongside multiple planner_mode=llm trials can trip that account's "
                             "rate limit purely from D0's own request volume, with no other task "
                             "involved, and rate-limit-induced retry delays manufacture score "
                             "differences that are statistically indistinguishable from a real "
                             "on/off effect -- this is not hypothetical: docs/noise_floor.md "
                             "documents this exact mechanism (two concurrent API-heavy jobs sharing "
                             "one rate limit) producing 9 of 28 paired comparisons falsely flagged "
                             "'significant' with zero real difference between the runs being "
                             "compared. Only raise this once you've accounted for that risk for "
                             "this particular run (附录 AV.3/AU.9).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to a prior report JSON to regress against. Only supported for "
                             "single-trial (--trials 1) reports.")
    parser.add_argument("--out", type=str, default=None, help="Output report path (default: timestamped)")
    args = parser.parse_args()

    if args.split == "test":
        print("*** Running against the HOLDOUT TEST SPLIT. ***")
        print("*** This result should be recorded as an official reveal, not used to iterate. ***\n")

    if args.trials < 1:
        print("--trials must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        sys.exit(2)

    suite = [tc for tc in load_golden_suite() if tc.split == args.split]
    if args.case:
        wanted = set(args.case)
        suite = [tc for tc in suite if tc.id in wanted]
    if args.limit:
        suite = suite[: args.limit]

    trial_word = "trial" if args.trials == 1 else "trials"
    print(f"Running eval harness on {len(suite)} golden case(s) [split={args.split}], "
          f"{args.trials} {trial_word}, concurrency={args.concurrency}...\n")

    graph = build_graph()
    eval_runner = EvalRunner()

    if args.trials == 1 and args.concurrency == 1:
        # Unchanged sequential path -- byte-for-byte the same code as
        # before D0 existed, including live per-case progress printed as
        # each ~60-125s pipeline run completes (src/eval/replay_cache.py's
        # docstring gives that per-case latency). Every pre-D0 invocation
        # of this script, and every default invocation post-D0, takes
        # this path. run_trials()/asyncio.gather() below only defers
        # printing until each full trial completes (measure_noise.py's
        # own run_full_mode has the same property), which is an
        # acceptable UX change for an explicitly-opted-into feature but
        # not for the default path.
        rows = []
        for tc in suite:
            print(f"  [{tc.id}] {tc.query!r} ...", end=" ", flush=True)
            eval_result, meta = await run_one_case(tc, graph, eval_runner)
            tags = []
            if meta["data_mismatch"]:
                tags.append("DATA MISMATCH")
            if meta["error_type"] == "safe_refusal":
                tags.append(f"SAFE REFUSAL: {meta['error']}")
            elif meta["error"]:
                tags.append(f"ERROR: {meta['error']}")
            if any(m == "fallback" for m in meta["mcp_modes"]):
                tags.append("fallback")
            suffix = f" [{', '.join(tags)}]" if tags else ""
            print(f"aggregate={eval_result.aggregate_score:.2f}{suffix}")
            rows.append({"test_case_id": tc.id, "eval_result": eval_result.to_dict(), "meta": meta})
        per_trial = [rows]
    else:
        print("(concurrency > 1 and/or trials > 1: no live per-case progress -- results print in "
              "a per-trial summary table once each trial finishes)\n")
        per_trial = await run_trials(suite, graph, eval_runner, args.trials, args.concurrency)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()

    if args.trials == 1:
        # Unchanged report shape -- byte-for-byte the same top-level keys
        # as every report this script produced before D0 existed, so
        # every existing consumer (this script's own --compare, any
        # external tooling reading logs/eval_runs/*.json) keeps working
        # unmodified for the default invocation.
        rows = per_trial[0]
        aggregate_scores = [r["eval_result"]["aggregate_score"] for r in rows]
        overall = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0
        report = {
            "timestamp": timestamp, "n_cases": len(rows),
            "overall_aggregate": overall, "split": args.split, "cases": rows,
        }
        out_path = Path(args.out) if args.out else REPORT_DIR / f"eval_{int(timestamp)}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        _print_summary(rows, overall)
        print(f"\nReport saved to {out_path}")
        if args.compare:
            _print_regressions(Path(args.compare), rows, overall)
        crashed = [r["test_case_id"] for r in rows if r["meta"]["error_type"] == "pipeline_error"]
    else:
        trial_reports = []
        crashed = []
        for trial_idx, rows in enumerate(per_trial):
            aggregate_scores = [r["eval_result"]["aggregate_score"] for r in rows]
            overall = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0
            trial_reports.append({
                "trial_index": trial_idx, "n_cases": len(rows),
                "overall_aggregate": overall, "cases": rows,
            })
            print(f"\n### trial {trial_idx} ###")
            _print_summary(rows, overall)
            crashed.extend(
                f"trial{trial_idx}:{r['test_case_id']}"
                for r in rows if r["meta"]["error_type"] == "pipeline_error"
            )
        report = {
            "timestamp": timestamp, "n_cases": len(suite), "split": args.split,
            "trials": args.trials, "concurrency": args.concurrency,
            "per_trial": trial_reports,
        }
        out_path = (
            Path(args.out) if args.out
            else REPORT_DIR / f"eval_trials{args.trials}_{int(timestamp)}.json"
        )
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to {out_path}")
        print(f"(multi-trial report -- pass it to src/eval/trials.py for pass@k/pass^k/variance "
              f"across the {args.trials} trials)")
        if args.compare:
            print("\n[compare] --compare only supports single-trial (--trials 1) reports -- "
                  "skipped for this multi-trial run. Use src/eval/trials.py's own pass^k/variance "
                  "output to characterize change across the trials of THIS report instead of a "
                  "two-report diff.")

    # A safe_refusal (guardrail correctly blocking a bad output) is a normal,
    # expected outcome for a golden suite run — it must not fail the script.
    # A genuine pipeline_error (uncaught exception, auth failure, etc.) is
    # not: this script previously always exited 0 regardless, so a CI step
    # running it could go green even when every case actually crashed (as
    # happened when the eval smoke case ran without a configured API key —
    # it printed "401 AuthenticationError" and still exited success).
    if crashed:
        print(f"\n{len(crashed)} case(s) hit a genuine pipeline error (not a safe refusal): {crashed}")
        sys.exit(1)


def _print_summary(rows, overall):
    print("\n" + "=" * 92)
    print(f"{'ID':6s} {'aggregate':>9s} {'relevance':>10s} {'grounded':>9s} {'factual':>8s} {'errrate':>8s}  notes")
    print("-" * 92)
    n_refusals = 0
    n_invalid = 0
    for r in rows:
        er = r["eval_result"]
        by = {s["metric"]: s["score"] for s in er["scores"]}
        invalid_metrics = [s["metric"] for s in er["scores"] if not s.get("valid", True)]
        notes = []
        if r["meta"]["data_mismatch"]:
            notes.append("data_mismatch")
        if r["meta"]["error_type"] == "safe_refusal":
            notes.append("safe_refusal")
            n_refusals += 1
        elif r["meta"]["error"]:
            notes.append("error")
        if any(m == "fallback" for m in r["meta"]["mcp_modes"]):
            notes.append("fallback")
        if invalid_metrics:
            notes.append(f"FAILED_TO_SCORE:{'+'.join(invalid_metrics)}")
            n_invalid += len(invalid_metrics)
        print(
            f"{r['test_case_id']:6s} {er['aggregate_score']:9.2f} "
            f"{by.get('answer_relevance', float('nan')):10.2f} "
            f"{by.get('groundedness', float('nan')):9.2f} "
            f"{by.get('factual_accuracy', float('nan')):8.2f} "
            f"{by.get('error_rate', float('nan')):8.2f}  {','.join(notes)}"
        )
    print("-" * 92)
    print(f"{'OVERALL':6s} {overall:9.2f}   safe_refusals={n_refusals}/{len(rows)}")
    if n_invalid:
        print(f"  {n_invalid} metric(s) FAILED TO SCORE this run (judge unreachable after "
              f"retries) -- excluded from their case's aggregate, not silently defaulted.")
    print("=" * 92)


def _print_regressions(baseline_path: Path, rows, overall):
    if not baseline_path.exists():
        print(f"\n[compare] baseline file not found: {baseline_path}")
        return
    baseline_report = json.loads(baseline_path.read_text())
    baseline_by_id = {c["test_case_id"]: c["eval_result"] for c in baseline_report["cases"]}

    print("\n--- Regression check vs", baseline_path, "---")
    any_alert = False
    for r in rows:
        base = baseline_by_id.get(r["test_case_id"])
        if not base:
            continue
        baseline_result = EvalResult(
            run_id=base["run_id"], query=base["query"],
            scores=[MetricScore(**s) for s in base["scores"]],
            aggregate_score=base["aggregate_score"],
        )
        current_result = EvalResult(
            run_id=r["eval_result"]["run_id"], query=r["eval_result"]["query"],
            scores=[MetricScore(**s) for s in r["eval_result"]["scores"]],
            aggregate_score=r["eval_result"]["aggregate_score"],
        )
        for a in detect_regressions(baseline_result, current_result):
            any_alert = True
            print(f"  [{r['test_case_id']}] {a.metric}: {a.baseline:.2f} -> {a.current:.2f} ({a.severity})")
    if not any_alert:
        print("  No regressions detected.")
    print(f"\n  Overall aggregate: baseline={baseline_report['overall_aggregate']:.3f} current={overall:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
