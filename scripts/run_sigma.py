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
inside a batch run concurrently.

--sample dev (default): the 40-case dev sample, refuses any case not split=dev, R arm on
tb5_dev.db whose content hash must equal EXPECTED.

--sample holdout (附录 HK, HL.5, HM.5): the H1 reveal on the 119 split=test cases in
tests/eval/h1_holdout_test.json, arms B C R CR plus 实验 2's arm A (auto/auto) at k = 2
(预注册 §7.3.3). Refuses before loading any case unless MAEDA_REVEAL_AUTHORIZED=1,
settings.intent_refine_trigger == "always" (§1.2 (A), §3.2) and the R-arm store is
tb5_full.db; the store's content hash is taken at start and must not change.

    PYTHONPATH=. caffeinate -i -s python scripts/run_sigma.py --out-dir data/sigma_v1 --budget 50
    PYTHONPATH=. python scripts/run_sigma.py --out-dir data/sigma_v1 --budget 50 --max-batches 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import contextvars
import json
import logging
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
HOLDOUT = ROOT / "tests" / "eval" / "h1_holdout_test.json"
EXPECTED = {
    "sales_csv": "b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719",
    "clean_csv": "6c95334422ccfefd4a3a4815944418dd995c3bcdd9513b3aa4db0be76f7d771d",
    "tb5_dev_content": "bc7f85fc47cf0d57bdfd6c3bc17c009cef1339b5b4e5868ad6a090998238045e",
}
#: arm -> (cleaner_mode, rag_mode). A is 实验 2's routing arm (预注册 §0, §7.3).
ARMS = {"B": ("force_off", "force_off"), "C": ("force_on", "force_off"),
        "R": ("force_off", "force_on"), "CR": ("force_on", "force_on"),
        "A": ("auto", "auto")}
DEFAULT_ARMS = {"dev": ["B", "C", "R", "CR"], "holdout": ["B", "C", "R", "CR", "A"]}
ARM_A_REPEATS = 2  # 预注册 §7.3.3
REQUIRED_REVEAL_TRIGGER = "always"  # 预注册 §1.2 (A), §3.2; 附录 HM.5


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def current_hashes() -> dict:
    return {
        "sales_csv": _sha(ROOT / "data/demo/sales_data.csv"),
        "clean_csv": _sha(ROOT / "data/truth/sales_data_clean.csv"),
        "tb5_dev_content": tb5_content_hash(settings.rag_db_path)["content_sha256"],
    }


def hash_problem(expected: dict | None = None) -> str | None:
    now = current_hashes()
    bad = [k for k, v in (expected or EXPECTED).items() if now[k] != v]
    return f"hash mismatch: {bad}" if bad else None


def preflight(sample: str) -> str | None:
    """Refusals checked before any case is loaded or any call is made."""
    store = Path(settings.rag_db_path or "").name
    if sample == "dev":
        return None if store == "tb5_dev.db" else "R-arm store must be tb5_dev.db (附录 GJ/GU)"
    from src.eval.reveal_guard import REVEAL_ENV, reveal_authorized
    if not reveal_authorized():
        return f"holdout reveal needs {REVEAL_ENV}=1 set by the reveal command (附录 HA.5)"
    if settings.intent_refine_trigger != REQUIRED_REVEAL_TRIGGER:
        return (f"holdout reveal needs MAEDA_INTENT_REFINE_TRIGGER={REQUIRED_REVEAL_TRIGGER}, "
                f"got {settings.intent_refine_trigger!r} (预注册 §1.2 (A), 附录 HM.5)")
    if store != "tb5_full.db":
        return "holdout reveal: R-arm store must be tb5_full.db (预注册 §10.5)"
    return None


def repeats_for(arm: str, repeats: int, sample: str) -> int:
    """Arm A runs at k = 2 on the holdout (§7.3.3); every other arm at --repeats."""
    return min(repeats, ARM_A_REPEATS) if (arm == "A" and sample == "holdout") else repeats


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


def load_sample(sample: str = "dev") -> list:
    if sample == "holdout":
        cases = load_golden_suite(str(HOLDOUT))  # reveal guard refuses here without the flag
        bad = [c.id for c in cases if c.split != "test"]
        if bad:
            raise SystemExit(f"holdout file holds non-test cases: {bad}")
        return cases
    ids = json.loads(SAMPLE.read_text())["case_ids"]
    suite = {c.id: c for c in load_golden_suite()}
    cases = [suite[i] for i in ids]
    bad = [c.id for c in cases if c.split != "dev"]
    if bad:
        raise SystemExit(f"refusing non-dev cases: {bad}")
    return cases


#: 附录 GZ: which run the current asyncio task belongs to, so a log handler can
#: attribute an LLM-fallback warning to its run even with concurrent runs.
CURRENT_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("sigma_run_key", default="")
LLM_FALLBACKS: dict[str, list[str]] = {}
_REPORT_FALLBACK_SIGNATURE = "Analysis completed for query: *"  # insight_agent._rule_based_report header


class _FallbackHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "using rule-based fallback" in msg and ("Report generation" in msg or "Insight generation" in msg):
            key = CURRENT_KEY.get()
            if key:
                LLM_FALLBACKS.setdefault(key, []).append(msg.split(":")[0])


def row_void(meta: dict, eval_result: dict, report: str = "", key: str = "",
             run_timeout: float | None = None) -> str | None:
    if "fallback" in (meta.get("mcp_modes") or []):
        return "mcp_fallback"
    # The insight/report LLM falling back to the rule-based writer is not in
    # degradation_log, so it is detected from the log record and, as a second
    # independent check, from the fallback report's fixed header (附录 GZ).
    if key in LLM_FALLBACKS or _REPORT_FALLBACK_SIGNATURE in (report or ""):
        return "llm_report_fallback"
    if run_timeout is not None and (meta.get("elapsed_s") or 0) > run_timeout:
        return "timeout"
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
        raise SystemExit("MAEDA_RAG_DB_PATH and MAEDA_RAG_INDEX_DIR must be set")
    refusal = preflight(args.sample)
    if refusal:
        raise SystemExit(f"REFUSING: {refusal}. Nothing was run and nothing was billed.")
    arms = args.arms or DEFAULT_ARMS[args.sample]
    if args.sample == "holdout":
        # tb5_full.db is built at the reveal (§10.5): its hash is fixed at start, then held.
        expected = {**{k: v for k, v in EXPECTED.items() if k != "tb5_dev_content"},
                    "tb5_dev_content": current_hashes()["tb5_dev_content"]}
    else:
        expected = EXPECTED
    problem = hash_problem(expected)
    if problem:
        print(f"STOP before start: {problem}")
        return 6
    cases = load_sample(args.sample)
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
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sample": args.sample,
            "expected_hashes": expected, "hashes_at_start": current_hashes(),
            "arms": {a: ARMS[a] for a in arms}, "repeats": args.repeats,
            "repeats_per_arm": {a: repeats_for(a, args.repeats, args.sample) for a in arms},
            "budget_usd": args.budget, "cleaner_llm_cost_per_clean_usd_assumed": args.cleaner_cost,
            "settings_snapshot": _settings_snapshot(), "provenance": _run_provenance(None),
        }, indent=2, default=str))
    spent = sum(r["cost_usd"] + r["cleaner_cost_est"] for r in done.values())
    n_void = sum(bool(r["void"]) for r in done.values())
    print(f"{len(done)} saved, ${spent:.4f} spent (tracked + cleaner estimate), cap ${args.budget}")
    graph = CapturingGraph(build_graph())
    eval_runner = EvalRunner()
    logging.getLogger().addHandler(_FallbackHandler(level=logging.WARNING))
    sem = asyncio.Semaphore(args.concurrency)
    batches = 0
    stop = None
    for rep in range(1, args.repeats + 1):
        for arm in arms:
            if rep > repeats_for(arm, args.repeats, args.sample):
                continue
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
                    "void": row_void(meta, er, st.get("report") or "", f"{arm}|{tc.id}|{rep}", args.run_timeout),
                    "foreign_sources": foreign,
                    "n_rag_chunks": len(st.get("rag_context") or []),
                    "report": st.get("report") or "",
                }

            async def one(tc):
                async with sem:
                    CURRENT_KEY.set(f"{arm}|{tc.id}|{rep}")
                    try:
                        # wall-clock timeout (time.time), not asyncio.wait_for: the loop's
                        # monotonic clock stops while the machine sleeps (附录 GZ).
                        task = asyncio.ensure_future(run_one_case(tc, graph, eval_runner))
                        t_start = time.time()
                        while not task.done():
                            await asyncio.wait({task}, timeout=5)
                            if not task.done() and time.time() - t_start > args.run_timeout:
                                task.cancel()
                                raise asyncio.TimeoutError()
                        er, meta = task.result()
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
                    stop = f"retrieved doc outside the store ({len(row['foreign_sources'])}), content not printed"
            results = range(n_results)
            spent += batch_cost
            print(f"batch {arm}/rep{rep}: {len(results)} runs, ${batch_cost:.3f}, total ${spent:.3f}, "
                  f"void {n_void}/{len(done)}")
            problem = hash_problem(expected)
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
    p.add_argument("--sample", choices=["dev", "holdout"], default="dev")
    p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS),
                   help="default: B C R CR on dev; B C R CR A on the holdout")
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
