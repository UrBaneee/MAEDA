"""
Eval harness — ties together Phase 9 (EvalRunner, golden suite, regression detector)
with the compiled LangGraph pipeline. This is the `tests/eval/run_eval.py` referenced
in DEV_SPEC.md, implemented as a CLI script.

Runs every golden test case end-to-end through graph.invoke(), scores the resulting
MAEDAState with EvalRunner, and writes a timestamped JSON report to logs/eval_runs/
for later regression comparison.

Usage:
    poetry run python scripts/run_eval.py                     # full 20-case suite
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

# Map golden case id -> data source descriptor.
#
# MAEDA's DataConnector loads a single flat DataFrame per query (no cross-table
# joins across the ecommerce_orders.db tables), so cases that would need a join
# (order value by category, new-vs-returning customers, upgrade likelihood) are
# assigned the closest single table and flagged as a known data mismatch below.
# This is intentional: it surfaces the single-table limitation as a finding
# rather than papering over it by rewriting the golden queries.
CASE_DATA_SOURCES: dict[str, dict] = {
    "D01": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "D02": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "products"},
    "D03": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "customers"},
    "D04": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "D05": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "orders"},
    "DG01": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "DG02": {"type": "csv", "path": "data/demo/churn_data.csv"},
    "DG03": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "DG04": {"type": "csv", "path": "data/demo/churn_data.csv"},
    "C01": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "C02": {"type": "csv", "path": "data/demo/marketing_campaigns.csv"},
    "C03": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "orders"},
    "C04": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "products"},
    "P01": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "P02": {"type": "csv", "path": "data/demo/churn_data.csv"},
    "P03": {"type": "csv", "path": "data/demo/churn_data.csv"},
    "E01": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "E02": {"type": "csv", "path": "data/demo/sales_data.csv"},
    "E03": {"type": "sql", "path": "sqlite:///data/demo/ecommerce_orders.db", "table_name": "orders"},
    "E04": {"type": "csv", "path": "data/demo/marketing_campaigns.csv"},
}

KNOWN_DATA_MISMATCH = {"D02", "DG04", "C03", "P03"}


def run_one_case(tc: GoldenTestCase, graph, eval_runner: EvalRunner) -> tuple[EvalResult, dict]:
    state = initial_state(tc.query)
    src = CASE_DATA_SOURCES.get(tc.id)
    if src:
        state["data_sources"] = [dict(src)]

    t0 = time.time()
    try:
        result_state = graph.invoke(state)
        run_error = result_state.get("error")
    except Exception as exc:
        result_state = state
        run_error = f"graph.invoke raised: {exc}"
        result_state["error"] = run_error
        result_state["error_type"] = "pipeline_error"  # an uncaught exception is never a safe refusal
    elapsed = time.time() - t0

    eval_result = asyncio.run(
        eval_runner.score(result_state, test_case=tc, start_time=t0, run_id=tc.id)
    )
    meta = {
        "elapsed_s": round(elapsed, 2),
        "guardrail_passed": result_state.get("guardrail_passed"),
        "current_phase": result_state.get("current_phase"),
        "error": run_error,
        "error_type": result_state.get("error_type"),
        "mcp_modes": sorted({c.get("mode", "mcp") for c in (result_state.get("mcp_call_log") or [])}),
        "data_mismatch": tc.id in KNOWN_DATA_MISMATCH,
    }
    return eval_result, meta


def main():
    parser = argparse.ArgumentParser(description="Run MAEDA eval harness against the golden suite")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--compare", type=str, default=None, help="Path to a prior report JSON to regress against")
    parser.add_argument("--out", type=str, default=None, help="Output report path (default: timestamped)")
    args = parser.parse_args()

    suite = load_golden_suite()
    if args.case:
        wanted = set(args.case)
        suite = [tc for tc in suite if tc.id in wanted]
    if args.limit:
        suite = suite[: args.limit]

    print(f"Running eval harness on {len(suite)} golden case(s)...\n")

    graph = build_graph()
    eval_runner = EvalRunner()

    rows = []
    for tc in suite:
        print(f"  [{tc.id}] {tc.query!r} ...", end=" ", flush=True)
        eval_result, meta = run_one_case(tc, graph, eval_runner)
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
    for r in rows:
        er = r["eval_result"]
        by = {s["metric"]: s["score"] for s in er["scores"]}
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
        print(
            f"{r['test_case_id']:6s} {er['aggregate_score']:9.2f} "
            f"{by.get('answer_relevance', float('nan')):10.2f} "
            f"{by.get('groundedness', float('nan')):9.2f} "
            f"{by.get('factual_accuracy', float('nan')):8.2f} "
            f"{by.get('error_rate', float('nan')):8.2f}  {','.join(notes)}"
        )
    print("-" * 92)
    print(f"{'OVERALL':6s} {overall:9.2f}   safe_refusals={n_refusals}/{len(rows)}")
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
    main()
