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
    }
    return eval_result, meta


async def main():
    parser = argparse.ArgumentParser(description="Run MAEDA eval harness against the golden suite")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Eval v2 Step 4: which split to run. Defaults to dev -- the holdout "
                             "'test' split must be requested explicitly so it can't be peeked at "
                             "by accident. See data/eval_split_manifest.json.")
    parser.add_argument("--compare", type=str, default=None, help="Path to a prior report JSON to regress against")
    parser.add_argument("--out", type=str, default=None, help="Output report path (default: timestamped)")
    args = parser.parse_args()

    if args.split == "test":
        print("*** Running against the HOLDOUT TEST SPLIT. ***")
        print("*** This result should be recorded as an official reveal, not used to iterate. ***\n")

    suite = [tc for tc in load_golden_suite() if tc.split == args.split]
    if args.case:
        wanted = set(args.case)
        suite = [tc for tc in suite if tc.id in wanted]
    if args.limit:
        suite = suite[: args.limit]

    print(f"Running eval harness on {len(suite)} golden case(s) [split={args.split}]...\n")

    graph = build_graph()
    eval_runner = EvalRunner()

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

    aggregate_scores = [r["eval_result"]["aggregate_score"] for r in rows]
    overall = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0

    report = {
        "timestamp": time.time(),
        "n_cases": len(rows),
        "overall_aggregate": overall,
        "split": args.split,
        "cases": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else REPORT_DIR / f"eval_{int(report['timestamp'])}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    _print_summary(rows, overall)
    print(f"\nReport saved to {out_path}")

    if args.compare:
        _print_regressions(Path(args.compare), rows, overall)

    # A safe_refusal (guardrail correctly blocking a bad output) is a normal,
    # expected outcome for a golden suite run — it must not fail the script.
    # A genuine pipeline_error (uncaught exception, auth failure, etc.) is
    # not: this script previously always exited 0 regardless, so a CI step
    # running it could go green even when every case actually crashed (as
    # happened when the eval smoke case ran without a configured API key —
    # it printed "401 AuthenticationError" and still exited success).
    crashed = [r["test_case_id"] for r in rows if r["meta"]["error_type"] == "pipeline_error"]
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
