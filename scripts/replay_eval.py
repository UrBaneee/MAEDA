"""
Score cached reports from the replay cache without regenerating anything --
the fast half of the split described in docs/eval_v2_plan.md Step 1's
replay-mode note (Step 2 iterates entirely on scoring logic in src/eval/,
which never needs a different report; see src/eval/replay_cache.py).

Any cached entry whose fingerprint doesn't match the CURRENT
generation_fingerprint() (source under src/ except src/eval/ changed, demo
data changed, or a generation-relevant setting changed) is treated as
missing here -- this script refuses to silently score a stale report
against current code. Run scripts/generate_replay_corpus.py first if a
case is reported missing.

Any eval_scores baked into a cached state (from whenever it was generated)
are discarded before scoring, so this always exercises the CURRENT
src/eval/metrics.py code -- that's the entire point.

Usage:
    poetry run python scripts/replay_eval.py                       # dev split (default)
    poetry run python scripts/replay_eval.py --split test           # holdout -- must be explicit
    poetry run python scripts/replay_eval.py --split all            # every cached case, either split
    poetry run python scripts/replay_eval.py --cases D01 DG03
    poetry run python scripts/replay_eval.py --compare logs/eval_runs/some_baseline.json
    poetry run python scripts/replay_eval.py --out logs/eval_runs/my_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.replay_cache import ReplayCache, generation_fingerprint
from src.eval.runner import EvalRunner, GoldenTestCase, load_golden_suite

from run_eval import REPORT_DIR, _print_regressions, _print_summary  # noqa: E402


async def _replay_one(tc: GoldenTestCase, cache: ReplayCache, fp: str, eval_runner: EvalRunner):
    entry = cache.get(tc.id, fp)
    if entry is None:
        return None

    state = dict(entry.state)
    state["eval_scores"] = None  # force a fresh judge call against CURRENT src/eval code

    t0 = time.time()
    eval_result = await eval_runner.score(state, test_case=tc, run_id=tc.id)
    scoring_elapsed = time.time() - t0

    meta = {
        "elapsed_s": entry.elapsed_s,  # original GENERATION time, not scoring time
        "replay_scoring_s": round(scoring_elapsed, 2),
        "guardrail_passed": state.get("guardrail_passed"),
        "current_phase": state.get("current_phase"),
        "error": state.get("error"),
        "error_type": state.get("error_type"),
        "mcp_modes": sorted({c.get("mode", "mcp") for c in (state.get("mcp_call_log") or [])}),
        "data_mismatch": "data_mismatch" in tc.tags,
        "replayed_from_cache": True,
    }
    return eval_result, meta


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=None, help="Subset of case ids; default = every cached case in --split")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                        help="Eval v2 Step 4: which split to replay. Defaults to dev -- the holdout "
                             "'test' split must be requested explicitly. 'all' ignores the split "
                             "entirely. See data/eval_split_manifest.json.")
    parser.add_argument("--compare", type=str, default=None, help="Path to a prior report JSON to regress against")
    parser.add_argument("--out", type=str, default=None, help="Output report path (default: timestamped)")
    args = parser.parse_args()

    if args.split == "test":
        print("*** Replaying against the HOLDOUT TEST SPLIT. ***")
        print("*** This result should be recorded as an official reveal, not used to iterate. ***\n")

    full_suite = {tc.id: tc for tc in load_golden_suite()}
    suite = full_suite if args.split == "all" else {
        cid: tc for cid, tc in full_suite.items() if tc.split == args.split
    }
    cache = ReplayCache()
    fp = generation_fingerprint()

    if args.cases:
        # --cases is still bounded by --split, so it can't be used as a
        # silent side-door around the holdout gate (e.g. --cases DG04 while
        # DG04 is in the test split, without --split test, refuses instead
        # of quietly peeking).
        out_of_split = [c for c in args.cases if c in full_suite and c not in suite]
        if out_of_split:
            print(f"{len(out_of_split)} requested case(s) are not in split={args.split!r}, skipping: "
                  f"{out_of_split} (pass --split test to include them)")
        wanted = [c for c in args.cases if c in suite]
    else:
        wanted = sorted(suite.keys())
    missing = cache.stale_or_missing(wanted)
    available = [cid for cid in wanted if cid not in missing]

    if missing:
        print(f"{len(missing)} case(s) missing or stale in the replay cache, skipping: {missing}")
        print("Run: poetry run python scripts/generate_replay_corpus.py --cases " + " ".join(missing))
    if not available:
        print("Nothing to replay.")
        sys.exit(1)

    print(f"Replaying {len(available)} cached case(s) [split={args.split}] against current src/eval code "
          f"(fingerprint {fp})...\n")

    eval_runner = EvalRunner()
    t0 = time.time()
    rows = []
    for cid in available:
        print(f"  [{cid}] {suite[cid].query!r} ...", end=" ", flush=True)
        result = await _replay_one(suite[cid], cache, fp, eval_runner)
        eval_result, meta = result
        _agg = eval_result.aggregate_score
        print(f"aggregate={'n/a' if _agg is None else f'{_agg:.2f}'}  "
              f"(scored in {meta['replay_scoring_s']:.1f}s, "
              f"originally generated in {meta['elapsed_s']:.1f}s)")
        rows.append({"test_case_id": cid, "eval_result": eval_result.to_dict(), "meta": meta})
    wall = time.time() - t0

    # E3 (附录 CU): aggregate_score is None for a run that did not terminate
    # in success -- see src/eval/runner.py::EvalResult. Averaging None in
    # would crash; averaging a coerced 0.0 in is the exact "不打零分" problem
    # E3 removed one level down.
    aggregate_scores = [r["eval_result"]["aggregate_score"] for r in rows
                        if r["eval_result"].get("aggregate_score") is not None]
    overall = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0
    n_no_aggregate = len(rows) - len(aggregate_scores)
    if n_no_aggregate:
        print(f"  ({n_no_aggregate} replayed case(s) did not terminate in success and are "
              f"excluded from the overall aggregate)")

    report = {
        "timestamp": time.time(), "n_cases": len(rows), "overall_aggregate": overall,
        "n_cases_without_aggregate": n_no_aggregate,
        "replayed": True, "generation_fingerprint": fp, "split": args.split, "cases": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else REPORT_DIR / f"replay_{int(report['timestamp'])}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    _print_summary(rows, overall)
    original_total = sum(r["meta"]["elapsed_s"] for r in rows)
    print(f"\nReplay wall-clock: {wall:.1f}s for {len(rows)} case(s) "
          f"(original generation would have taken ~{original_total:.0f}s -- "
          f"{original_total / max(wall, 0.01):.0f}x faster)")
    print(f"Report saved to {out_path}")

    if args.compare:
        _print_regressions(Path(args.compare), rows, overall)


if __name__ == "__main__":
    asyncio.run(main())
