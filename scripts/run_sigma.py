#!/usr/bin/env python3
"""
σ run for H1/H3 (附录 EJ.3/EJ.4/EJ.5, GU.2, GV): four arms (B, C, R, CR) x the 40-case
H1 sample (tests/eval/h1_sample.json) x N repeats. PAID: each run is one full
pipeline run plus in-graph scoring.

Safe to interrupt and rerun: every run is appended to <out-dir>/runs.jsonl as soon as
it exists and a rerun skips keys already saved. Stops (without discarding anything
saved) on: spend at the cap; void rate over 10%; a retrieved document outside the dev
store; any hash mismatch (sales CSV, clean table, tb5_dev.db content). Never reruns
a finished run (EJ.5).

Arms are applied per batch (repeat, arm) because settings are process-global; cases
inside a batch run concurrently. Dev split only: refuses any case not split=dev.

    PYTHONPATH=. python scripts/run_sigma.py --out-dir data/sigma_v1 --budget 50
    PYTHONPATH=. python scripts/run_sigma.py --out-dir data/sigma_v1 --budget 50 --max-batches 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _run_provenance, _settings_snapshot, run_one_case  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.eval.data_provenance import tb5_content_hash  # noqa: E402
from src.eval.runner import EvalRunner, load_golden_suite  # noqa: E402
from src.graph.builder import build_graph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "eval" / "h1_sample.json"
EXPECTED = {
    "sales_csv": "b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719",
    "clean_csv": "6c95334422ccfefd4a3a4815944418dd995c3bcdd9513b3aa4db0be76f7d771d",
    "tb5_dev_content": "bc7f85fc47cf0d57bdfd6c3bc17c009cef1339b5b4e5868ad6a090998238045e",
}
#: arm -> (cleaner_mode, rag_mode)
ARMS = {"B": ("force_off", "force_off"), "C": ("force_on", "force_off"),
        "R": ("force_off", "force_on"), "CR": ("force_on", "force_on")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def current_hashes() -> dict:
    return {
        "sales_csv": _sha(ROOT / "data/demo/sales_data.csv"),
        "clean_csv": _sha(ROOT / "data/truth/sales_data_clean.csv"),
        "tb5_dev_content": tb5_content_hash(settings.rag_db_path)["content_sha256"],
    }


def hash_problem() -> str | None:
    now = current_hashes()
    bad = [k for k, v in EXPECTED.items() if now[k] != v]
    return f"hash mismatch: {bad}" if bad else None


def known_doc_ids(db_path: str) -> set[str]:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT doc_id FROM documents")}
    finally:
        conn.close()


class CapturingGraph:
    """Wraps the compiled graph so the final state of each run (retrieval sources,
    per-agent cost) is available to this script; run_one_case does not return it."""

    def __init__(self, graph):
        self._graph = graph
        self.states: dict[str, dict] = {}

    async def ainvoke(self, state):
        out = await self._graph.ainvoke(state)
        self.states[state["run_id"]] = out
        return out


def state_cost(state: dict) -> float:
    return sum((v or {}).get("cost_usd", 0.0) for v in (state.get("token_usage") or {}).values())


def load_sample() -> list:
    ids = json.loads(SAMPLE.read_text())["case_ids"]
    suite = {c.id: c for c in load_golden_suite()}
    cases = [suite[i] for i in ids]
    bad = [c.id for c in cases if c.split != "dev"]
    if bad:
        raise SystemExit(f"refusing non-dev cases: {bad}")
    return cases


def row_void(meta: dict, eval_result: dict) -> str | None:
    if "fallback" in (meta.get("mcp_modes") or []):
        return "mcp_fallback"
    if eval_result.get("rag_arm_invalid_reason"):
        return "rag_arm_invalid"
    if eval_result.get("cleaning_arm_invalid_reason"):
        return "cleaning_arm_invalid"
    return None


async def main_async(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    settings.eval_enabled = True
    if not (settings.rag_db_path and settings.rag_index_dir):
        raise SystemExit("MAEDA_RAG_DB_PATH and MAEDA_RAG_INDEX_DIR must point at tb5_dev")
    if Path(settings.rag_db_path).name != "tb5_dev.db":
        raise SystemExit("R-arm store must be tb5_dev.db (附录 GJ/GU)")
    problem = hash_problem()
    if problem:
        print(f"STOP before start: {problem}")
        return 6
    cases = load_sample()
    if args.cases:
        cases = [c for c in cases if c.id in set(args.cases)]
    docs = known_doc_ids(settings.rag_db_path)
    done: dict[str, dict] = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    header = out_dir / "header.json"
    if not header.exists():
        header.write_text(json.dumps({
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "expected_hashes": EXPECTED,
            "hashes_at_start": current_hashes(), "arms": ARMS, "repeats": args.repeats,
            "budget_usd": args.budget, "cleaner_llm_cost_per_clean_usd_assumed": args.cleaner_cost,
            "settings_snapshot": _settings_snapshot(), "provenance": _run_provenance(None),
        }, indent=2, default=str))
    spent = sum(r["cost_usd"] + r["cleaner_cost_est"] for r in done.values())
    n_void = sum(bool(r["void"]) for r in done.values())
    print(f"{len(done)} saved, ${spent:.4f} spent (tracked + cleaner estimate), cap ${args.budget}")
    graph = CapturingGraph(build_graph())
    eval_runner = EvalRunner()
    sem = asyncio.Semaphore(args.concurrency)
    batches = 0
    stop = None
    for rep in range(1, args.repeats + 1):
        for arm in args.arms:
            todo = [c for c in cases if f"{arm}|{c.id}|{rep}" not in done]
            if not todo:
                continue
            if args.max_batches and batches >= args.max_batches:
                print("max-batches reached")
                return 0
            batches += 1
            cl, rg = ARMS[arm]
            settings.maeda_cleaner_mode, settings.maeda_rag_mode = cl, rg
            per_run = (spent / len(done)) if done else args.first_estimate
            if spent + per_run * len(todo) > args.budget:
                # stop before the batch rather than part-way through an arm
                print(f"STOP: batch {arm}/{rep} (~${per_run * len(todo):.2f}) would pass the cap (${spent:.2f} spent)")
                return 3

            def build_row(tc, er, meta):
                st = graph.states.pop(meta.get("run_id"), {})
                foreign = [d for d in (r.get("source_file") if isinstance(r, dict) else r
                                       for r in (st.get("rag_sources") or [])) if d and d not in docs]
                cleaned = cl == "force_on" and any(
                    c.get("mode") == "mcp" and "clean" in str(c.get("tool", c.get("name", "")))
                    for c in (st.get("mcp_call_log") or []))
                return {
                    "key": f"{arm}|{tc.id}|{rep}", "arm": arm, "case_id": tc.id, "repeat": rep,
                    "eval_result": er, "meta": meta, "cost_usd": round(state_cost(st), 6),
                    "cleaner_cost_est": args.cleaner_cost if cleaned else 0.0,
                    "void": row_void(meta, er), "foreign_sources": foreign,
                    "n_rag_chunks": len(st.get("rag_context") or []),
                    "report": st.get("report") or "",
                }

            async def one(tc):
                async with sem:
                    try:
                        er, meta = await asyncio.wait_for(run_one_case(tc, graph, eval_runner),
                                                          timeout=args.run_timeout)
                        return build_row(tc, er.to_dict(), meta)
                    except asyncio.TimeoutError:
                        # a hung run is recorded as void, never retried (EJ.5)
                        return {"key": f"{arm}|{tc.id}|{rep}", "arm": arm, "case_id": tc.id, "repeat": rep,
                                "eval_result": {"test_case_id": tc.id, "scores": [], "terminal_state": "timeout"},
                                "meta": {"mcp_modes": [], "terminal_state": "timeout"}, "cost_usd": 0.0,
                                "cleaner_cost_est": 0.0, "void": "timeout", "foreign_sources": [],
                                "n_rag_chunks": 0, "report": ""}

            batch_cost = 0.0
            n_results = 0
            for fut in asyncio.as_completed([one(tc) for tc in todo]):
                row = await fut
                n_results += 1
                with runs_path.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                done[row["key"]] = row
                batch_cost += row["cost_usd"] + row["cleaner_cost_est"]
                n_void += bool(row["void"])
                if row["foreign_sources"]:
                    stop = f"retrieved doc outside the dev store ({len(row['foreign_sources'])}), content not printed"
            results = range(n_results)
            spent += batch_cost
            print(f"batch {arm}/rep{rep}: {len(results)} runs, ${batch_cost:.3f}, total ${spent:.3f}, "
                  f"void {n_void}/{len(done)}")
            problem = hash_problem()
            if problem:
                stop = problem
            if len(done) >= 20 and n_void / len(done) > 0.10:
                stop = f"void rate {n_void}/{len(done)} over 10%"
            if spent >= args.budget:
                stop = f"spend ${spent:.2f} reached the cap"
            if stop:
                print(f"STOP: {stop}")
                return 4
    (out_dir / "header.json").write_text(
        json.dumps({**json.loads(header.read_text()), "hashes_at_end": current_hashes(),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2, default=str))
    print("complete")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=["B", "C", "R", "CR"], choices=list(ARMS))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--cases", nargs="+", default=None, help="subset of the sample (smoke runs only)")
    p.add_argument("--run-timeout", type=float, default=900.0, help="seconds before a hung run is recorded void")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--first-estimate", type=float, default=0.04)
    p.add_argument("--cleaner-cost", type=float, default=0.0,
                   help="assumed USD per cleaner LLM planning call (not reported by the cleaner)")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
