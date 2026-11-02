#!/usr/bin/env python3
"""
Generate the reports labelled for the H2 attribution scorer (预注册 §7.2.1, R5.1,
附录 FL). PAID: every job is one full pipeline run.

Arms B and R only, both with the cleaner forced off. In-graph eval is disabled,
so no judge is paid here; the attribution judge runs later on the saved text.
Each report is appended to the JSONL as soon as it exists, and a rerun skips jobs
already saved, so an interruption loses at most the job in flight. Stops before a
job when the running cost would pass the budget, and stops at once if an R-arm
run is invalid (that is a setup fault, not a datum).

    python scripts/generate_h2_reports.py --plan dry --budget 9
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _run_provenance, _settings_snapshot  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.eval.runner import GoldenTestCase  # noqa: E402
from src.graph.builder import build_graph  # noqa: E402
from src.state.graph_state import initial_state  # noqa: E402

CASES = Path("tests/eval/h2_dev_attribution_cases.json")
OUT_DIR = Path("data/h2_attribution")
ARMS = {"B": "force_off", "R": "force_on"}

# Dry run: two normal events and one distractor, the probe query the formal
# plan uses least, one trial per arm. Six reports, none of them in the 60.
DRY = [("H2-2023-09-EAST-PARTNER-DISPUTE-Q2", 1), ("H2-2024-05-NORTH-GADGETPRO-EOL-Q2", 1),
       ("H2-2023-04-SOUTH-DISCOUNT-CAMPAIGN-Q2", 1)]


def jobs(plan: str, cases: list[dict]) -> list[dict]:
    if plan == "dry":
        chosen = [(cid, f"dry{n}") for cid, n in DRY]
    else:
        chosen = [(c["id"], f"t{t}") for c in cases
                  for t in range(1, (3 if c["event_type"] == "normal" else 1) + 1)]
    return [{"job_id": f"{cid}|{arm}|{trial}", "case_id": cid, "arm": arm, "trial": trial}
            for cid, trial in chosen for arm in ARMS]


def run_cost(state: dict) -> float:
    return sum((v or {}).get("cost_usd", 0.0) for v in (state.get("token_usage") or {}).values())


async def run_job(job: dict, case: GoldenTestCase, graph) -> dict:
    settings.maeda_rag_mode = ARMS[job["arm"]]
    state = initial_state(case.query)
    state["data_sources"] = [dict(case.data_source)]
    t0 = time.time()
    try:
        out = await graph.ainvoke(state)
        error = out.get("error")
    except Exception as exc:  # noqa: BLE001
        out, error = state, f"graph.ainvoke raised {type(exc).__name__}: {exc}"
    return {
        **job, "event_id": case.event_id, "event_type": case.event_type,
        "query": case.query, "report": out.get("report") or "",
        "error": error, "terminal_state": out.get("terminal_state"),
        "rag_arm_invalid_reason": out.get("rag_arm_invalid_reason"),
        "n_rag_chunks": len(out.get("rag_context") or []),
        "rag_sources": [s.get("source_file") for s in out.get("rag_sources") or []],
        "mcp_modes": [c.get("mode") for c in out.get("mcp_call_log") or []],
        "degradation_log": out.get("degradation_log") or [],
        "cost_usd": round(run_cost(out), 6), "elapsed_s": round(time.time() - t0, 1),
    }


async def main_async(args) -> int:
    settings.maeda_cleaner_mode = "force_off"
    settings.eval_enabled = False
    if not (settings.rag_db_path and settings.rag_index_dir):
        raise SystemExit("MAEDA_RAG_DB_PATH and MAEDA_RAG_INDEX_DIR must point at the TB5 corpus (附录 FL)")
    cases = {c["id"]: GoldenTestCase.from_dict(c) for c in json.loads(CASES.read_text())}
    out_path = OUT_DIR / f"reports_{args.plan}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done[row["job_id"]] = row
    header = OUT_DIR / f"reports_{args.plan}.header.json"
    if not header.exists():
        header.write_text(json.dumps({"settings_snapshot": _settings_snapshot(),
                                      "provenance": _run_provenance(None),
                                      "cases_file": str(CASES)}, indent=2, default=str))
    spent = args.already_spent + sum(r["cost_usd"] for r in done.values())
    todo = [j for j in jobs(args.plan, list(json.loads(CASES.read_text()))) if j["job_id"] not in done]
    print(f"{len(done)} saved, {len(todo)} to run, ${spent:.4f} spent so far (budget ${args.budget})")
    graph = build_graph()
    for job in todo:
        per_run = (spent / len(done)) if done else args.first_estimate
        if spent + per_run > args.budget:
            print(f"STOP: next job would pass the budget (${spent:.4f} + ~${per_run:.4f})")
            return 3
        row = await run_job(job, cases[job["case_id"]], graph)
        with out_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[job["job_id"]] = row
        spent += row["cost_usd"]
        print(f"{job['job_id']}: {len(row['report'])} chars, rag_chunks={row['n_rag_chunks']}, "
              f"modes={row['mcp_modes']}, ${row['cost_usd']:.4f}, total ${spent:.4f}")
        if job["arm"] == "R" and (row["rag_arm_invalid_reason"] or row["n_rag_chunks"] == 0):
            print(f"STOP: R arm not live ({row['rag_arm_invalid_reason'] or 'no chunks'})")
            return 4
        if row["error"] or not row["report"]:
            print(f"STOP: run failed ({row['error']})")
            return 5
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=["dry", "formal"], required=True)
    parser.add_argument("--budget", type=float, required=True, help="USD cap for MAEDA-side cost")
    parser.add_argument("--already-spent", type=float, default=0.0)
    parser.add_argument("--first-estimate", type=float, default=0.1)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
