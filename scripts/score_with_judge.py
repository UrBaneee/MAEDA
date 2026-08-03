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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.eval.metrics import score_answer_relevance, score_groundedness
from src.eval.replay_cache import ReplayCache

ANNOTATIONS_DIR = Path("data/annotations")


async def _score_case(cid: str, state: dict) -> dict:
    query = state.get("user_query", "")
    report = state.get("report") or ""
    analysis_results = state.get("analysis_results") or []
    rag_context = state.get("rag_context") or []
    data_quality_report = state.get("data_quality_report")

    rel, gnd = await asyncio.gather(
        score_answer_relevance(query, report, analysis_results, rag_context,
                               data_quality_report=data_quality_report),
        score_groundedness(query, report, analysis_results, rag_context,
                           data_quality_report=data_quality_report),
    )
    return {
        "case_id": cid,
        "answer_relevance": {"score": rel.score, "label": rel.label, "reasoning": rel.reasoning, "valid": rel.valid},
        "groundedness": {"score": gnd.score, "label": gnd.label, "reasoning": gnd.reasoning, "valid": gnd.valid},
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases-from", type=str, required=True,
                        help="A human_labels_*.json (or bare {case_id:...} file) -- scores exactly its case ids")
    parser.add_argument("--out", type=str, required=True, help="Output answer-key path")
    parser.add_argument("--label", type=str, default=None, help="Judge label for the printed summary only")
    args = parser.parse_args()

    raw = json.loads(Path(args.cases_from).read_text())
    case_ids = sorted((raw.get("labels", raw)).keys())

    cache = ReplayCache()
    missing = [cid for cid in case_ids if cid not in cache._entries]
    if missing:
        print(f"{len(missing)} case(s) not in the replay cache at all, skipping: {missing}")
        case_ids = [c for c in case_ids if c not in missing]

    judge = args.label or f"{settings.resolved_eval_provider}:{settings.resolved_eval_model}"
    print(f"Scoring {len(case_ids)} cached case(s) with judge={judge!r} "
          f"(each case's OWN stored report, not regenerated)...\n")

    t0 = time.time()
    results = await asyncio.gather(*[
        _score_case(cid, cache._entries[cid].state) for cid in case_ids
    ])
    elapsed = time.time() - t0

    answer_key = {r["case_id"]: {k: v for k, v in r.items() if k != "case_id"} for r in results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer_key, indent=2, default=str))

    invalid = [(r["case_id"], m) for r in results for m in ("answer_relevance", "groundedness")
               if not r[m]["valid"]]
    print(f"Scored {len(results)} case(s) in {elapsed:.1f}s. {len(invalid)} invalid metric call(s): {invalid}")
    print(f"Answer key saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
