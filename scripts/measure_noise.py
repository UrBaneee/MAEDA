"""
Eval v2, Step 1 (see docs/eval_v2_plan.md) — measure the noise floor before
touching any metric definitions, so later comparisons have something real
to be judged against instead of eyeballing a baseline timeline.

Two noise sources, measured separately:

  judge   Fix a handful of real (query, report, analysis_results,
          rag_context) tuples once — by running a few golden cases through
          the real pipeline a single time and caching the result — then
          repeatedly re-invoke JUST the judge on the same fixed input.
          Cheap (no pipeline reruns after the first cache-fill), isolates
          judge noise from generation noise. Measures both a single raw
          judge call (n_samples=1) and the full median-of-N procedure the
          harness actually reports (n_samples=settings.eval_judge_samples),
          since the gap between the two is itself informative (how much
          does median-of-N actually buy you).

          Note: logs/eval_runs/*.json only stores computed scores, not the
          report/findings/rag_context that produced them (see
          src/eval/runner.py's EvalResult and src/persistence/run_store.py's
          schema — neither persists the raw text) — so historical reports
          can't be replayed. Fixed states have to be captured fresh, once,
          then cached.

  full    Rerun the golden suite N times end-to-end, same config, nothing
          changed. Expensive (real pipeline + real judge calls every time)
          but this is the number that actually gates whether a future score
          change is real: generation noise + judging noise stacked, exactly
          as a before/after comparison would see it.

Usage:
    poetry run python scripts/measure_noise.py judge --repeats 15
    poetry run python scripts/measure_noise.py judge --cases D01 DG01 C02 --repeats 20
    poetry run python scripts/measure_noise.py full --repeats 8              # dev split (default)
    poetry run python scripts/measure_noise.py full --repeats 5 --cases D01 D02 DG01 DG03 C01 C02 E02 E04 --concurrency 4
    poetry run python scripts/measure_noise.py full --repeats 8 --split all # explicit, warns -- touches test split (附录 CC.1)

Note (附录 CC.1/CB.2.5, 阶段 3 收尾执行计划轮次 1): `full` defaults to
`--split dev`. Before this default existed, a bare `full` invocation
silently ran the whole suite -- including the 41 test-split cases -- 8
times, spending 硬约束 1's one-time test-split reveal without anyone
noticing. Reach for `--split test`/`--split all` only when that reveal is
actually intended; both print an unmissable warning before running.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# `scripts/` isn't an installed package (only `src` is, per pyproject.toml's
# packages=[{include="src"}]), so run_eval.py's own directory needs to be on
# sys.path before it can be imported by name. Running this file directly
# (`python scripts/measure_noise.py`) already puts that directory at
# sys.path[0], but doing it explicitly here keeps the import robust to how
# the script gets invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.metrics import score_relevance_and_groundedness
from src.eval.noise import BootstrapResult, paired_bootstrap_ci, summarize
from src.eval.runner import EvalRunner, GoldenTestCase, load_golden_suite
from src.graph.builder import build_graph
from src.state.graph_state import initial_state

# Reuse scripts/run_eval.py's one-case runner rather than duplicating it.
# Eval v2 Step 4 consolidated data-source wiring onto GoldenTestCase.data_source
# (tests/eval/test_suite.json) so this no longer needs its own copy.
from run_eval import run_one_case  # noqa: E402

REPORT_DIR = Path("logs/noise_runs")
JUDGE_CACHE_PATH = REPORT_DIR / "judge_fixed_states.json"

# Default fixed-state cases for judge-noise mode: one per query_type plus
# DG03, the false-premise case ("North is underperforming" when it's
# actually the top region) — the most interesting case in the suite for
# checking whether groundedness scoring is stable on a report that pushes
# back against a leading question.
DEFAULT_JUDGE_CASES = ["D01", "DG01", "DG03", "C02", "P01", "E02"]


# ─── judge mode ───────────────────────────────────────────────────────────────

async def _fill_judge_cache(case_ids: list[str]) -> dict:
    """Run each requested case once through the real pipeline, cache the
    (query, report, analysis_results, rag_context) tuple the judge actually
    sees. Cached to disk so re-running this script doesn't repeat the
    expensive part."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if JUDGE_CACHE_PATH.exists():
        cache = json.loads(JUDGE_CACHE_PATH.read_text())

    missing = [cid for cid in case_ids if cid not in cache]
    if not missing:
        return cache

    print(f"Filling judge-noise cache for {missing} (one real pipeline run each)...")
    suite = {tc.id: tc for tc in load_golden_suite()}
    graph = build_graph()
    for cid in missing:
        tc = suite[cid]
        state = initial_state(tc.query)
        if tc.data_source:
            state["data_sources"] = [dict(tc.data_source)]
        result_state = await graph.ainvoke(state)
        cache[cid] = {
            "query": result_state.get("user_query", tc.query),
            "report": result_state.get("report") or "",
            "analysis_results": result_state.get("analysis_results") or [],
            "rag_context": result_state.get("rag_context") or [],
        }
        print(f"  [{cid}] cached (report length={len(cache[cid]['report'])} chars)")

    JUDGE_CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))
    return cache


async def run_judge_mode(case_ids: list[str], repeats: int, concurrency: int) -> dict:
    cache = await _fill_judge_cache(case_ids)
    sem = asyncio.Semaphore(concurrency)

    async def _one_raw(cid: str):
        async with sem:
            fixed = cache[cid]
            rel, gnd = await score_relevance_and_groundedness(
                fixed["query"], fixed["report"], fixed["analysis_results"],
                fixed["rag_context"], n_samples=1,
            )
            return cid, "raw", rel.score, gnd.score

    async def _one_median(cid: str):
        async with sem:
            fixed = cache[cid]
            rel, gnd = await score_relevance_and_groundedness(
                fixed["query"], fixed["report"], fixed["analysis_results"],
                fixed["rag_context"],
            )
            return cid, "median", rel.score, gnd.score

    print(f"\nRunning {repeats} judge repeats x {len(case_ids)} cases "
          f"(raw single-call + median-of-N), concurrency={concurrency}...")
    tasks = []
    for cid in case_ids:
        for _ in range(repeats):
            tasks.append(_one_raw(cid))
            tasks.append(_one_median(cid))
    rows = await asyncio.gather(*tasks)

    by_key: dict[tuple[str, str], dict[str, list[float]]] = {}
    for cid, kind, rel_score, gnd_score in rows:
        key = (cid, kind)
        by_key.setdefault(key, {"answer_relevance": [], "groundedness": []})
        by_key[key]["answer_relevance"].append(rel_score)
        by_key[key]["groundedness"].append(gnd_score)

    per_case = {}
    pooled = {"raw": {"answer_relevance": [], "groundedness": []},
              "median": {"answer_relevance": [], "groundedness": []}}
    for (cid, kind), scores in by_key.items():
        per_case.setdefault(cid, {})
        per_case[cid][kind] = {
            m: (summarize(f"{cid}/{kind}/{m}", v).to_dict() if summarize(f"{cid}/{kind}/{m}", v) else None)
            for m, v in scores.items()
        }
        for m, v in scores.items():
            pooled[kind][m].extend(v)

    pooled_summary = {
        kind: {m: (summarize(f"pooled/{kind}/{m}", v).to_dict() if summarize(f"pooled/{kind}/{m}", v) else None)
               for m, v in metrics.items()}
        for kind, metrics in pooled.items()
    }

    return {"per_case": per_case, "pooled": pooled_summary, "cases": case_ids, "repeats": repeats}


def _print_judge_report(report: dict):
    # Per-case numbers are the actual noise floor: repeated judge calls on
    # the SAME fixed input. The pooled table further down mixes that with
    # how much the 6 DIFFERENT cases genuinely differ from each other (D01
    # is an easy case that should score near 1.0; DG01/DG03/C02/P01 are
    # meant to be hard) -- pooling makes std look far larger than the real
    # per-case noise, so per-case is what belongs in a noise-floor writeup.
    print("\n" + "=" * 78)
    print("JUDGE NOISE — per fixed case (the actual noise floor)")
    print("=" * 78)
    for cid in report["cases"]:
        for kind in ("raw", "median"):
            label = "single call" if kind == "raw" else "median-of-N"
            row = report["per_case"].get(cid, {}).get(kind, {})
            parts = []
            for metric, s in row.items():
                if s is None:
                    continue
                parts.append(f"{metric}=std:{s['std']:.3f}(2σ={s['two_sigma']:.3f})")
            print(f"    {cid:6s} {label:12s} {'  '.join(parts)}")
    print("\n" + "-" * 78)
    print("  Pooled across all cases (mixes real between-case differences with noise --")
    print("  NOT the noise floor, shown only as an overall-spread sanity check):")
    for kind in ("raw", "median"):
        label = "single call (n_samples=1)" if kind == "raw" else "harness procedure (median-of-N)"
        print(f"\n    {label}:")
        for metric, s in report["pooled"][kind].items():
            if s is None:
                continue
            print(f"      {metric:20s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"2σ={s['two_sigma']:.3f}  range=[{s['min']:.2f}, {s['max']:.2f}]  n={s['n']}")
    print("=" * 78)


# ─── full-pipeline mode ────────────────────────────────────────────────────────

async def run_full_mode(cases: list[GoldenTestCase], repeats: int, concurrency: int) -> dict:
    graph = build_graph()
    eval_runner = EvalRunner()
    sem = asyncio.Semaphore(concurrency)

    async def _one(trial_idx: int, tc: GoldenTestCase):
        async with sem:
            eval_result, meta = await run_one_case(tc, graph, eval_runner)
            return trial_idx, tc.id, eval_result.to_dict(), meta

    print(f"\nRunning {repeats} full-suite repeats x {len(cases)} cases, "
          f"concurrency={concurrency} (this calls the real pipeline + real judge "
          f"every time — expect real API cost)...")
    tasks = [_one(t, tc) for t in range(repeats) for tc in cases]
    rows = await asyncio.gather(*tasks)

    trials: list[dict] = [dict() for _ in range(repeats)]
    for trial_idx, cid, eval_result_dict, meta in rows:
        trials[trial_idx][cid] = {"eval_result": eval_result_dict, "meta": meta}

    # Per-trial overall aggregate (mean of that trial's per-case aggregates,
    # matching scripts/run_eval.py's own definition of "overall_aggregate")
    # and per-trial per-metric mean (mean of that metric across the cases in
    # that trial) — the two units of "one measurement" that get compared
    # across the `repeats` trials to characterize full-pipeline noise.
    trial_aggregates = []
    trial_metric_means: dict[str, list[float]] = {}
    total_cost = 0.0
    for trial in trials:
        # E3 (附录 CU): a run that did not terminate in success has
        # aggregate_score None -- it has no comparable aggregate, so it is
        # excluded from the trial mean rather than contributing a zero that
        # would be read as full-pipeline noise (src/eval/runner.py::EvalResult).
        case_aggs = [c["eval_result"]["aggregate_score"] for c in trial.values()
                     if c["eval_result"].get("aggregate_score") is not None]
        trial_aggregates.append(sum(case_aggs) / len(case_aggs) if case_aggs else 0.0)
        per_metric: dict[str, list[float]] = {}
        for c in trial.values():
            for s in c["eval_result"]["scores"]:
                per_metric.setdefault(s["metric"], []).append(s["score"])
                if s["metric"] == "token_cost" and s.get("raw_value") is not None:
                    total_cost += s["raw_value"]
        for metric, vals in per_metric.items():
            trial_metric_means.setdefault(metric, []).append(sum(vals) / len(vals))

    overall_summary = summarize("overall_aggregate", trial_aggregates)
    metric_summaries = {
        m: summarize(m, vals).to_dict() if summarize(m, vals) else None
        for m, vals in trial_metric_means.items()
    }

    # Sanity check on the bootstrap procedure itself: compare trial 0 vs
    # trial 1 (nothing actually changed between them) and confirm the CI
    # straddles zero, i.e. the method correctly reports "not significant"
    # when there's genuinely no change to detect.
    #
    # Bootstrap resampling needs a real sample to say anything — with too
    # few paired cases, every resample is drawn from the same handful of
    # values, so the CI comes out artificially narrow and "significant" by
    # construction, not because anything real was detected. Below this
    # threshold, report the raw numbers but don't claim the check passed or
    # failed; the check is only meaningful with enough cases to resample.
    _MIN_CASES_FOR_BOOTSTRAP = 8
    bootstrap_sanity = None
    if repeats >= 2:
        # Paired: a case only contributes a delta when BOTH trials produced a
        # comparable aggregate (E3, 附录 CU -- None on either side means that
        # trial did not terminate in success).
        common_ids = [
            cid for cid in sorted(set(trials[0]) & set(trials[1]))
            if trials[0][cid]["eval_result"].get("aggregate_score") is not None
            and trials[1][cid]["eval_result"].get("aggregate_score") is not None
        ]
        deltas = [
            trials[1][cid]["eval_result"]["aggregate_score"]
            - trials[0][cid]["eval_result"]["aggregate_score"]
            for cid in common_ids
        ]
        if len(deltas) >= 2:
            br: BootstrapResult = paired_bootstrap_ci(deltas)
            reliable = len(deltas) >= _MIN_CASES_FOR_BOOTSTRAP
            bootstrap_sanity = {
                "compared": "trial_0_vs_trial_1", "n_cases": br.n,
                "mean_delta": round(br.mean_delta, 4),
                "ci_low": round(br.ci_low, 4), "ci_high": round(br.ci_high, 4),
                "significant": br.significant,
                "reliable": reliable,
                "expected": "significant should be False -- nothing changed between these two trials"
                            if reliable else
                            f"n_cases={len(deltas)} < {_MIN_CASES_FOR_BOOTSTRAP} -- too few paired cases for "
                            f"this check to mean anything; ignore 'significant' until run on more cases",
            }

    return {
        "repeats": repeats,
        "n_cases": len(cases),
        "total_cost_usd": round(total_cost, 4),
        "overall_aggregate_summary": overall_summary.to_dict() if overall_summary else None,
        "per_metric_summary": metric_summaries,
        "bootstrap_sanity_check": bootstrap_sanity,
        "trial_aggregates": [round(a, 4) for a in trial_aggregates],
        "trials_raw": trials,
    }


def _print_full_report(report: dict):
    print("\n" + "=" * 78)
    print(f"FULL-PIPELINE NOISE — {report['repeats']} repeats x {report['n_cases']} cases")
    print(f"Total API cost across this measurement: ${report['total_cost_usd']:.4f}")
    print("=" * 78)
    ov = report["overall_aggregate_summary"]
    if ov:
        print(f"\n  overall_aggregate: mean={ov['mean']:.3f}  std={ov['std']:.3f}  "
              f"2σ={ov['two_sigma']:.3f}  range=[{ov['min']:.3f}, {ov['max']:.3f}]")
        print(f"  trial values: {report['trial_aggregates']}")
    print("\n  per-metric (mean-across-cases-per-trial, then variation across trials):")
    for metric, s in report["per_metric_summary"].items():
        if s is None:
            continue
        print(f"    {metric:22s} mean={s['mean']:.3f}  std={s['std']:.3f}  2σ={s['two_sigma']:.3f}")
    bs = report.get("bootstrap_sanity_check")
    if bs:
        if not bs["reliable"]:
            verdict = f"NOT MEANINGFUL — only {bs['n_cases']} paired cases; rerun with >= 8"
        elif not bs["significant"]:
            verdict = "OK (not significant, as expected)"
        else:
            verdict = "UNEXPECTED -- flagged significant with no real change; investigate"
        print(f"\n  Bootstrap sanity check (trial 0 vs trial 1, nothing actually changed):")
        print(f"    mean_delta={bs['mean_delta']:.4f}  95% CI=[{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]  {verdict}")
    print("=" * 78)


# ─── full mode: split-aware case selection (附录 CC.1/CB.2.5) ─────────────────

def select_full_mode_cases(
    suite: list, cases, split: str,
) -> list:
    """
    `full` mode's case selection, factored out of main() so the default --
    and the exact danger 附录 CC.1 found in not having one -- can be locked
    by a test that never touches the real pipeline.

    Before this function existed, `full`'s only selection knob was
    `--cases` (default None = the WHOLE golden suite, dev and test split
    mixed, no way to filter by split at all). A bare
    `measure_noise.py full` therefore silently ran all 41 test-split cases
    8 times alongside the 59 dev-split ones -- spending 硬约束 1's one-time
    test-split reveal in a routine noise-floor-remeasurement invocation,
    with nothing warning that this had happened.

    - `--cases` given: unchanged pre-existing behavior, an explicit
      id-by-id allowlist. `--split` is ignored in this branch -- an
      explicit case list is already a deliberate, informed choice, unlike
      the *default* path CC.1 found dangerous.
    - `--cases` absent: filtered to `tc.split == split`. `"dev"` is the
      new default (see --split's argparse help below) -- this is the
      actual fix: the single most common invocation (no flags at all) now
      can't touch the test split. `"all"` is the explicit escape hatch
      back to the old (dangerous-by-default) behavior; it is never the
      default and must be typed out on purpose.
    """
    if cases:
        wanted = set(cases)
        return [tc for tc in suite if tc.id in wanted]
    if split == "all":
        return list(suite)
    return [tc for tc in suite if tc.split == split]


def warn_if_test_split_involved(selected: list) -> None:
    """
    附录 CC.1: an unmissable, readable warning whenever a `full` run is
    about to touch ANY test-split case, for whatever reason (`--split
    test`, `--split all`, or an explicit `--cases` list that happens to
    include test ids -- an explicit list is already a deliberate choice,
    but a copy-pasted or typo'd id is still worth flagging cheaply).
    Printed to stdout, not just logged -- 硬约束 1: the test split gets
    exactly ONE official reveal, and this is the last point before a real
    `full` run actually executes where a human is still watching the
    terminal it was launched from.
    """
    test_ids = sorted(tc.id for tc in selected if tc.split == "test")
    if not test_ids:
        return
    print("\n" + "!" * 78)
    print("!! WARNING: this `full` run includes TEST-SPLIT cases.")
    print("!! 硬约束 1: test split gets exactly ONE official reveal --")
    print("!! if this is that reveal, make sure it's the real predetermined")
    print("!! run, not routine noise-floor/infra maintenance.")
    print(f"!! {len(test_ids)} test case(s): {test_ids}")
    print("!! Routine noise-floor remeasurement should use --split dev")
    print("!! (the default) or an explicit dev-only --cases list instead.")
    print("!" * 78 + "\n")


# ─── CLI ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    # Concurrency defaults are conservative on purpose: this project's
    # configured OpenAI org caps out at 30,000 TPM / 500 RPM (discovered by
    # running this script at the old defaults of 6 / 4 and getting most
    # samples silently rate-limited -- see docs/noise_floor.md). The
    # rate-limit retry in src/eval/metrics.py's _judge_once now absorbs
    # occasional 429s instead of corrupting the measurement, but running
    # deliberately far below the ceiling means fewer retries, less wall
    # clock time burned on backoff, and no risk of a "thundering herd" of
    # simultaneous retries re-triggering the same limit.
    p_judge = sub.add_parser("judge", help="Measure judge-only noise on fixed, cached states")
    p_judge.add_argument("--cases", nargs="+", default=DEFAULT_JUDGE_CASES)
    p_judge.add_argument("--repeats", type=int, default=15)
    p_judge.add_argument("--concurrency", type=int, default=3)

    p_full = sub.add_parser("full", help="Measure full-pipeline noise by rerunning the suite N times")
    p_full.add_argument("--cases", nargs="+", default=None,
                        help="Subset of case ids; default = filtered by --split instead of "
                             "the whole suite. See --split.")
    p_full.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                        help="附录 CC.1/CB.2.5: which split(s) to draw cases from when --cases "
                             "is not given. Defaults to 'dev' -- before this flag existed, a "
                             "bare `measure_noise.py full` silently ran the WHOLE 100-case "
                             "suite (41 of them test-split) 8 times, spending 硬约束 1's "
                             "one-time test-split reveal without anyone noticing (附录 CC.1). "
                             "'test'/'all' must be requested explicitly and print an unmissable "
                             "warning banner before running. Ignored when --cases is given.")
    p_full.add_argument("--repeats", type=int, default=8)
    p_full.add_argument("--concurrency", type=int, default=2)

    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    if args.mode == "judge":
        report = await run_judge_mode(args.cases, args.repeats, args.concurrency)
        _print_judge_report(report)
        out = REPORT_DIR / f"judge_noise_{ts}.json"
    else:
        suite = load_golden_suite()
        selected = select_full_mode_cases(suite, args.cases, args.split)
        warn_if_test_split_involved(selected)
        report = await run_full_mode(selected, args.repeats, args.concurrency)
        _print_full_report(report)
        out = REPORT_DIR / f"full_noise_{ts}.json"

    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
