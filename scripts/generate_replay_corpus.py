"""
Generate (or refresh) the replay cache -- runs each requested golden case
through the REAL pipeline exactly once and caches the full resulting
MAEDAState (see src/eval/replay_cache.py for the cache + fingerprint
design, and docs/eval_v2_plan.md for why this exists: Step 2 iterates on
scoring logic in src/eval/, which never needs a different report, and
regenerating all cases on every metric tweak was the single biggest
cost/time sink in that work).

This is the slow half, done once. scripts/replay_eval.py is the fast half,
done every time a metric changes -- it scores straight from this cache.

A case already cached under the CURRENT generation_fingerprint() is
skipped (idempotent, resumable: rerunning this script after a partial
failure only regenerates what's missing). Source changes under src/
(except src/eval/), demo-data changes, or generation-relevant settings
changes all invalidate the fingerprint and force regeneration -- see
src/eval/replay_cache.py's generation_fingerprint() docstring for exactly
what's covered.

Usage:
    poetry run python scripts/generate_replay_corpus.py                    # whole golden suite, skip what's already cached and fresh
    poetry run python scripts/generate_replay_corpus.py --cases D01 DG03   # just these
    poetry run python scripts/generate_replay_corpus.py --force            # regenerate even if cached and fresh
    poetry run python scripts/generate_replay_corpus.py --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.replay_cache import ReplayCache, generation_fingerprint
from src.eval.runner import GoldenTestCase, load_golden_suite
from src.graph.builder import build_graph
from src.state.graph_state import initial_state

# This account was upgraded from OpenAI Tier 1 (30k TPM/500 RPM) to Tier 2
# during this project (see docs/noise_floor.md's incident writeup for how
# Tier 1's ceiling was found). Tier 2 measured at 450,000 TPM / 5,000 RPM
# (read directly off response headers, not looked up from a table that
# might not match this account). Deriving a safe concurrency from that:
#
#   safe_concurrency ~= TPM * seconds_per_case / (60 * tokens_per_case)
#                     ~= 450,000 * 70 / (60 * 6,000) ~= 87
#
# using this project's own measured ~70s/case and ~6,000 tokens/case
# (logs/noise_runs/full_noise_1785391747.json). That's the token-budget
# ceiling, not a recommendation -- other bottlenecks (chart rendering,
# local SQLite, MCP fallback connection attempts) become the real limit
# well before 87 concurrent pipelines. This default is a conservative step
# up from the Tier-1-safe value of 2, not the theoretical max; raise
# --concurrency further only after confirming those other bottlenecks
# don't fall over first.
_DEFAULT_CONCURRENCY = 12


async def _generate_one(tc: GoldenTestCase, graph, cache: ReplayCache, fp: str) -> tuple[str, float, float]:
    state = initial_state(tc.query)
    if tc.data_source:
        state["data_sources"] = [dict(tc.data_source)]

    t0 = time.time()
    result_state = await graph.ainvoke(state)
    elapsed = time.time() - t0

    cost = sum(
        v.get("cost_usd", 0) for v in (result_state.get("token_usage") or {}).values()
        if isinstance(v, dict)
    )
    cache.put(tc.id, result_state, elapsed_s=elapsed, cost_usd=cost, fingerprint=fp)
    return tc.id, elapsed, cost


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=None, help="Subset of case ids; default = whole golden suite")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a fresh cache entry already exists")
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    suite = {tc.id: tc for tc in load_golden_suite()}
    wanted = list(args.cases) if args.cases else sorted(suite.keys())
    unknown = [cid for cid in wanted if cid not in suite]
    if unknown:
        print(f"Unknown case id(s), not in the golden suite: {unknown}")
        sys.exit(1)

    cache = ReplayCache()
    fp = generation_fingerprint()

    if args.force:
        to_generate = wanted
    else:
        to_generate = cache.stale_or_missing(wanted)
    reused = [cid for cid in wanted if cid not in to_generate]

    print(f"Fingerprint: {fp}")
    print(f"{len(reused)} case(s) already cached and fresh, reusing: {reused}")
    print(f"{len(to_generate)} case(s) to generate: {to_generate}")
    if not to_generate:
        print("Nothing to do.")
        return

    graph = build_graph()
    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(tc):
        async with sem:
            return await _generate_one(tc, graph, cache, fp)

    print(f"\nGenerating {len(to_generate)} case(s), concurrency={args.concurrency}...")
    t0 = time.time()
    results = await asyncio.gather(*[_bounded(suite[cid]) for cid in to_generate])
    wall = time.time() - t0

    cache.save()

    total_cost = sum(r[2] for r in results)
    print(f"\nDone in {wall:.1f}s wall-clock. Total cost: ${total_cost:.4f}")
    print(f"Cache now holds {len(cache)} case(s) at logs/replay_cache/corpus.json")
    for cid, elapsed, cost in sorted(results, key=lambda r: -r[1]):
        print(f"  [{cid}] {elapsed:.1f}s  ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
