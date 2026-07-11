"""
MAEDA Demo Scenarios — Phase 12.5

Five scripted interview scenarios. Run any scenario directly:
    python scripts/demo_scenarios.py --scenario 1

Each scenario prints agent reasoning, chart paths, and the final report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEMO_DATA = Path("data/demo")

SCENARIOS = {
    1: {
        "name": "The Full Stack",
        "description": (
            "Upload messy CSV → Data Cleaner auto-cleans → Analysis → "
            "RAG enriches → Insight Report. Shows entire ecosystem."
        ),
        "query": "Analyse total revenue by region and identify top-performing regions over the last 3 years.",
        "data_source": str(DEMO_DATA / "sales_data.csv"),
    },
    2: {
        "name": "Diagnostic: Q3 Revenue Drop",
        "description": (
            "Multi-step diagnostic with root cause analysis. "
            "Demonstrates the Analysis Agent chaining correlated steps."
        ),
        "query": "Why did revenue drop in Q3 2023? Break it down by region and product.",
        "data_source": str(DEMO_DATA / "sales_data.csv"),
    },
    3: {
        "name": "Customer Churn Root Cause",
        "description": (
            "Diagnostic query on the churn dataset. "
            "RAG Server enriches with domain knowledge about churn drivers."
        ),
        "query": "Why did churn spike in March 2024? What customer segments were most affected?",
        "data_source": str(DEMO_DATA / "churn_data.csv"),
    },
    4: {
        "name": "Marketing ROI Comparison",
        "description": (
            "Comparative analysis across channels. "
            "Demonstrates chart generation and report writing."
        ),
        "query": "Compare ROI across marketing channels. Which channels have the best spend efficiency?",
        "data_source": str(DEMO_DATA / "marketing_campaigns.csv"),
    },
    5: {
        "name": "Graceful Degradation",
        "description": (
            "Runs with RAG Server offline. MAEDA still completes analysis "
            "using only the data and built-in tools."
        ),
        "query": "Summarise customer purchasing patterns and recommend high-value segments to target.",
        "data_source": str(DEMO_DATA / "ecommerce_orders.db"),
    },
}


def run_scenario(n: int, verbose: bool = True) -> dict:
    scenario = SCENARIOS[n]
    print(f"\n{'=' * 60}")
    print(f"Scenario {n}: {scenario['name']}")
    print(f"{'=' * 60}")
    print(f"Description: {scenario['description']}")
    print(f"Query: {scenario['query']}")
    print(f"Data: {scenario['data_source']}")
    print()

    from src.graph.builder import build_graph
    from src.state.graph_state import initial_state

    state = initial_state(scenario["query"])
    src = scenario["data_source"]
    if src:
        ext = src.rsplit(".", 1)[-1].lower()
        type_map = {"csv": "csv", "json": "json", "xlsx": "excel",
                    "xls": "excel", "db": "sql", "sqlite": "sql"}
        src_type = type_map.get(ext, "csv")
        # SQL sources need a connection string, not a bare file path
        src_path = f"sqlite:///{src}" if src_type == "sql" else src
        state["data_sources"] = [{"path": src_path, "type": src_type}]

    t0 = time.time()
    graph = build_graph()
    result = graph.invoke(state)
    elapsed = round(time.time() - t0, 2)

    print(f"\n✅ Completed in {elapsed}s")
    print(f"Guardrail: {'PASSED' if result.get('guardrail_passed') else 'FAILED'}")

    if verbose:
        # Print agent decision summary
        print("\n--- Agent Decisions ---")
        for entry in result.get("decision_trace", []):
            print(f"  [{entry['agent_name']}] {entry['action']} — {entry['reasoning'][:80]}")

        # Print insights
        print("\n--- Key Insights ---")
        for ins in (result.get("insights") or [])[:3]:
            impact = ins.get("impact", "").upper()
            print(f"  [{impact}] {ins.get('finding', '')}")
            print(f"          → {ins.get('recommendation', '')}")

        # Print charts generated
        charts = [c for c in (result.get("charts") or []) if c.get("chart_type") != "dashboard"]
        if charts:
            print(f"\n--- Charts ({len(charts)}) ---")
            for c in charts:
                print(f"  {c.get('chart_type')} — {c.get('title')} [{c.get('image_path', '')}]")

        # Eval summary
        eval_scores = result.get("eval_scores") or {}
        agg = eval_scores.get("_aggregate")
        if agg:
            print(f"\n--- Eval Score: {agg:.0%} ---")
            for k, v in eval_scores.items():
                if k != "_aggregate" and isinstance(v, dict):
                    print(f"  {k:30s} {v['score']:.0%} [{v['label']}]")

        # Abbreviated report
        report = result.get("report", "")
        if report:
            print("\n--- Report (first 500 chars) ---")
            print(report[:500] + ("…" if len(report) > 500 else ""))

    return result


def main():
    parser = argparse.ArgumentParser(description="Run MAEDA demo scenarios")
    parser.add_argument("--scenario", type=int, choices=list(SCENARIOS.keys()), default=1,
                        help="Scenario number to run (1–5)")
    parser.add_argument("--all", action="store_true", help="Run all 5 scenarios")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.all:
        for n in SCENARIOS:
            run_scenario(n, verbose=not args.quiet)
    else:
        run_scenario(args.scenario, verbose=not args.quiet)


if __name__ == "__main__":
    main()
