"""
Score already-cached golden-suite reports with whatever judge
settings.resolved_eval_provider currently resolves to (eval v2 roadmap
#29) -- built for the judge-model comparison table scaffolded in
docs/judge_calibration.md: re-score the SAME reports a prior judge model
was already scored against, using the SAME human labels already collected,
so a QWK comparison isolates "which judge model" as the only variable.

Deliberately bypasses ReplayCache.get()'s current-fingerprint check --
export_for_annotation.py's strict freshness requirement exists so a blind
labeling tool never shows a stale report, but here the goal is the
opposite: score the exact reports a specific prior answer key was built
from, not whatever the pipeline would generate right now. Reads each
entry's own stored fingerprint directly.

Usage:
    poetry run python scripts/score_with_judge.py \
        --cases-from data/annotations/human_labels_pooled100.json \
        --out data/annotations/answer_key_claude_sonnet_5_100.json \
        --label claude-sonnet-5
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.eval.metrics import score_answer_relevance, score_groundedness
from src.eval.replay_cache import ReplayCache

ANNOTATIONS_DIR = Path("data/annotations")

# Each case makes 2 metric calls (relevance + groundedness), and each metric
# call internally fires settings.eval_judge_samples (default 3) independent
# LLM calls to median-aggregate away single-call noise -- so one case is
# really ~6 real API calls, not 1. The first live run of this script (2026)
# gathered all 100 cases with NO concurrency cap at all, unlike
# generate_replay_corpus.py's deliberate Semaphore(12) (added earlier for
# the exact same reason, against OpenAI -- this script just didn't carry
# that pattern over when it was written): ~600 calls fired essentially at
# once against Anthropic, triggering heavy rate-limit retries and burning
# through the account's credit balance in an uncontrolled burst before
# finishing. Mirrors generate_replay_corpus.py's default exactly.
_DEFAULT_CONCURRENCY = 12
_ALLOW_PAID_ENV = "MAEDA_ALLOW_PAID_TESTS"


def _dmx_peak_window(now: datetime | None = None) -> str | None:
    """Return the active DMX peak window, or None when off-peak.

    User-frozen pricing rule: Monday-Friday, 01:00-04:00 and 06:00-10:00
    UTC are peak. Boundaries are half-open. This intentionally has no
    override: calibration runs are required to be off-peak.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if current.weekday() >= 5:
        return None
    current_time = current.time().replace(tzinfo=None)
    for start, end in ((dt_time(1), dt_time(4)), (dt_time(6), dt_time(10))):
        if start <= current_time < end:
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')} UTC"
    return None


def _require_dmx_off_peak(uses_dmx: bool = True) -> None:
    if not uses_dmx:
        return
    active = _dmx_peak_window()
    if active:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        raise SystemExit(
            f"Refusing paid DMXAPI calls during peak window {active}; now={now}. "
            "Run during off-peak hours."
        )


async def _score_case(cid: str, state: dict, metrics: set[str]) -> dict:
    query = state.get("user_query", "")
    report = state.get("report") or ""
    analysis_results = state.get("analysis_results") or []
    rag_context = state.get("rag_context") or []
    data_quality_report = state.get("data_quality_report")

    tasks = {}
    if "answer_relevance" in metrics:
        tasks["answer_relevance"] = score_answer_relevance(
            query, report, analysis_results, rag_context,
            data_quality_report=data_quality_report,
        )
    if "groundedness" in metrics:
        tasks["groundedness"] = score_groundedness(
            query, report, analysis_results, rag_context,
            data_quality_report=data_quality_report,
        )
    names = list(tasks)
    scores = await asyncio.gather(*(tasks[name] for name in names))
    by_name = dict(zip(names, scores))
    result = {"case_id": cid}
    if "answer_relevance" in by_name:
        rel = by_name["answer_relevance"]
        result["answer_relevance"] = {
            "score": rel.score, "label": rel.label, "reasoning": rel.reasoning,
            "valid": rel.valid, "raw_value": rel.raw_value,
        }
    if "groundedness" in by_name:
        gnd = by_name["groundedness"]
        result["groundedness"] = {
            "score": gnd.score, "label": gnd.label, "reasoning": gnd.reasoning,
            "valid": gnd.valid, "raw_value": gnd.raw_value,
        }
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases-from", type=str, required=True,
                        help="A human_labels_*.json (or bare {case_id:...} file) -- scores exactly its case ids")
    parser.add_argument("--out", type=str, required=True, help="Output answer-key path")
    parser.add_argument("--label", type=str, default=None, help="Judge label for the printed summary only")
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY,
                         help="Max cases scored at once (each case is ~6 real LLM calls -- see module docstring)")
    parser.add_argument("--execute", action="store_true", help="Actually send paid judge calls")
    parser.add_argument(
        "--metrics", choices=["both", "answer_relevance", "groundedness"], default="both",
        help="Run both judge metrics or only one",
    )
    args = parser.parse_args()

    metrics = (
        {"answer_relevance", "groundedness"}
        if args.metrics == "both"
        else {args.metrics}
    )
    uses_dmx = (
        ("answer_relevance" in metrics and settings.resolved_eval_provider == "dmxapi")
        or (
            "groundedness" in metrics
            and settings.resolved_groundedness_provider == "dmxapi"
        )
    )

    if not args.execute:
        raise SystemExit("Refusing paid calls: pass --execute after reviewing the command")
    if os.environ.get(_ALLOW_PAID_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit(f"Refusing paid calls: set {_ALLOW_PAID_ENV}=1 for this process")
    if settings.resolved_eval_provider == "dmxapi" and not settings.dmx_api_key:
        raise SystemExit("Refusing paid calls: DMX_API_KEY is missing")
    if (
        "groundedness" in metrics
        and settings.resolved_groundedness_provider == "openai"
        and not settings.openai_api_key
    ):
        raise SystemExit("Refusing paid calls: OPENAI_API_KEY is missing")
    _require_dmx_off_peak(uses_dmx)

    # Validate the paid run's destination before the first request. A shell
    # line break once truncated ``--out data/annotations/file.json`` to the
    # directory ``data/annotations/``; all calls completed and only then did
    # write_text fail, losing every paid result held in memory.
    out_path = Path(args.out)
    if out_path.exists() and out_path.is_dir():
        raise SystemExit(f"Refusing paid calls: --out is a directory: {out_path}")
    if out_path.suffix.lower() != ".json":
        raise SystemExit(f"Refusing paid calls: --out must be a .json file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases_path = Path(args.cases_from)
    raw_bytes = cases_path.read_bytes()
    raw = json.loads(raw_bytes)
    case_ids = sorted((raw.get("labels", raw)).keys())

    cache = ReplayCache()
    missing = [cid for cid in case_ids if cid not in cache._entries]
    if missing:
        print(f"{len(missing)} case(s) not in the replay cache at all, skipping: {missing}")
        case_ids = [c for c in case_ids if c not in missing]

    judge = args.label or f"{settings.resolved_eval_provider}:{settings.resolved_eval_model}"
    print(f"Scoring {len(case_ids)} cached case(s) with judge={judge!r}, "
          f"concurrency={args.concurrency} "
          f"(~{len(case_ids) * settings.eval_judge_samples * len(metrics)} real LLM calls total) "
          f"(each case's OWN stored report, not regenerated)...\n")

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    done = 0
    completed: dict[str, dict] = {}

    metadata = {
        "provider": settings.resolved_eval_provider,
        "model": settings.resolved_eval_model,
        "judge_samples": settings.eval_judge_samples,
        "base_url": settings.dmx_api_base_url if settings.resolved_eval_provider == "dmxapi" else None,
        "cases_from": str(cases_path),
        "cases_from_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "case_ids": case_ids,
        "concurrency": args.concurrency,
        "metrics": sorted(metrics),
        "groundedness_provider": settings.resolved_groundedness_provider,
        "groundedness_model": settings.resolved_groundedness_model,
        "groundedness_reasoning_effort": settings.eval_groundedness_reasoning_effort,
        "groundedness_max_completion_tokens": settings.eval_groundedness_max_completion_tokens,
        "status": "incomplete",
    }

    def _checkpoint() -> None:
        payload = {"metadata": metadata, "scores": completed}
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp_path, out_path)

    async def _bounded(cid: str) -> dict:
        nonlocal done
        async with sem:
            _require_dmx_off_peak(uses_dmx)
            result = await _score_case(cid, cache._entries[cid].state, metrics)
        async with write_lock:
            completed[cid] = {k: v for k, v in result.items() if k != "case_id"}
            done += 1
            failed = [metric for metric in metrics if not result[metric]["valid"]]
            if failed:
                metadata["status"] = "failed_validation"
                metadata["invalid_metrics"] = [[cid, metric] for metric in failed]
            _checkpoint()
            print(f"  [{done}/{len(case_ids)}] {cid} done (checkpoint saved)")
            if failed:
                raise RuntimeError(
                    f"Fail-fast: {cid} has invalid metric(s) {failed}; remaining tasks cancelled"
                )
        return result

    t0 = time.time()
    results = await asyncio.gather(*[_bounded(cid) for cid in case_ids])
    elapsed = time.time() - t0

    invalid = [(r["case_id"], m) for r in results for m in metrics
               if not r[m]["valid"]]
    metadata["status"] = "failed_validation" if invalid else "complete"
    metadata["elapsed_seconds"] = round(elapsed, 3)
    metadata["invalid_metrics"] = invalid
    _checkpoint()
    print(f"\nScored {len(results)} case(s) in {elapsed:.1f}s. {len(invalid)} invalid metric call(s): {invalid}")
    print(f"Answer key saved to {out_path}")
    if invalid:
        raise SystemExit(
            f"Calibration invalid: {len(invalid)} metric(s) failed; "
            "the answer key must not be used for agreement."
        )


if __name__ == "__main__":
    asyncio.run(main())
