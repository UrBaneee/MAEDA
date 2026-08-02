"""
Export blind labeling material for eval v2 Step 3 (docs/eval_v2_plan.md) --
human-vs-judge agreement on answer_relevance/groundedness.

Uses all 20 cases in the current replay cache (logs/replay_cache/corpus.json)
rather than a stratified subset: the golden suite only has 20 distinct
queries right now (Step 4 will expand it to 100-200), so there's no larger
pool to sample down from yet. Reports the natural query_type / score-band
distribution across those 20 for transparency instead.

Writes two separate files, deliberately never merged:
  - data/annotations/answer_key_<ts>.json   the judge's own scores +
    reasoning for each case. NOT shown during labeling -- seeing the
    judge's score first anchors the human rating and inflates measured
    agreement.
  - <html_out>                              a self-contained labeling page:
    query + report + findings + RAG context per case, two 5-band rating
    controls, autosave to localStorage, and a "Download labels" button that
    produces a data/annotations/human_labels.json-shaped file. No judge
    scores anywhere in this file's data.

Usage:
    poetry run python scripts/export_for_annotation.py
    poetry run python scripts/export_for_annotation.py --html-out /tmp/label.html
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.metrics import (
    render_data_quality,
    render_findings,
    render_rag_context,
    score_answer_relevance,
    score_groundedness,
)
from src.eval.replay_cache import ReplayCache, generation_fingerprint
from src.eval.runner import load_golden_suite

ANNOTATIONS_DIR = Path("data/annotations")
_TEMPLATE_PATH = Path(__file__).resolve().parent / "_annotation_template.html"


async def _score_case(cid: str, state: dict, suite: dict) -> dict:
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

    tc = suite.get(cid)
    return {
        # render_findings/render_rag_context/render_data_quality (src/eval/metrics.py)
        # are the SAME functions score_answer_relevance/score_groundedness use to
        # build the judge's own prompt -- the judge and the human annotator must
        # see byte-for-byte identical material, or measured agreement compares two
        # different inputs instead of the same one (found via a real annotation
        # session, see docs/judge_calibration.md, case C01).
        "blind": {
            "case_id": cid,
            "query_type": tc.query_type if tc else "unknown",
            "query": query,
            "report": report,
            "findings": render_findings(analysis_results),
            "data_quality": render_data_quality(data_quality_report),
            "rag_context": render_rag_context(rag_context),
        },
        "answer_key": {
            "case_id": cid,
            "answer_relevance": {"score": rel.score, "label": rel.label, "reasoning": rel.reasoning, "valid": rel.valid},
            "groundedness": {"score": gnd.score, "label": gnd.label, "reasoning": gnd.reasoning, "valid": gnd.valid},
        },
    }


def _band(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "mid-high"
    if score >= 0.4:
        return "mid"
    if score >= 0.1:
        return "mid-low"
    return "low"


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html-out", type=str, default=None,
                        help="Path to write the self-contained labeling HTML (default: data/annotations/labeling_tool_<ts>.html)")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Subset of case ids to export (default: whole golden suite). "
                             "Used to export a fresh batch of cases separately from ones "
                             "already labeled, e.g. a Step 4 suite-expansion batch.")
    args = parser.parse_args()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    suite = {tc.id: tc for tc in load_golden_suite()}
    cache = ReplayCache()
    fp = generation_fingerprint()
    if args.cases:
        unknown = set(args.cases) - set(suite.keys())
        if unknown:
            print(f"Unknown case id(s), not in the golden suite: {sorted(unknown)}")
            sys.exit(1)
        case_ids = sorted(args.cases)
    else:
        case_ids = sorted(suite.keys())
    missing = cache.stale_or_missing(case_ids)
    if missing:
        print(f"{len(missing)} case(s) missing or stale in the replay cache: {missing}")
        print("Run: poetry run python scripts/generate_replay_corpus.py")
        sys.exit(1)

    print(f"Scoring {len(case_ids)} cached case(s) with the current judge to build the answer key...")
    results = await asyncio.gather(*[
        _score_case(cid, cache.get(cid, fp).state, suite) for cid in case_ids
    ])

    ts = int(time.time())
    answer_key = {r["blind"]["case_id"]: r["answer_key"] for r in results}
    answer_key_path = ANNOTATIONS_DIR / f"answer_key_{ts}.json"
    answer_key_path.write_text(json.dumps(answer_key, indent=2, default=str))

    # Transparency on what's actually being labeled -- not a sampling
    # decision (all 20 cases are used), just a report on the natural spread.
    print(f"\n{'case':6}{'query_type':14}{'rel_band':10}{'gnd_band'}")
    by_type = {}
    by_band = {}
    for r in results:
        cid = r["blind"]["case_id"]
        qt = r["blind"]["query_type"]
        rel_b = _band(r["answer_key"]["answer_relevance"]["score"])
        gnd_b = _band(r["answer_key"]["groundedness"]["score"])
        by_type[qt] = by_type.get(qt, 0) + 1
        by_band[gnd_b] = by_band.get(gnd_b, 0) + 1
        print(f"{cid:6}{qt:14}{rel_b:10}{gnd_b}")
    print(f"\nquery_type distribution: {by_type}")
    print(f"groundedness score-band distribution: {by_band}")

    blind_samples = [r["blind"] for r in results]
    template = _TEMPLATE_PATH.read_text()
    html = template.replace(
        "/*__SAMPLES_JSON__*/",
        json.dumps(blind_samples, ensure_ascii=False),
    )

    html_out = Path(args.html_out) if args.html_out else ANNOTATIONS_DIR / f"labeling_tool_{ts}.html"
    html_out.write_text(html)

    print(f"\nAnswer key (hidden during labeling): {answer_key_path}")
    print(f"Labeling tool (open this / publish as an Artifact): {html_out}")
    print("\nAfter labeling, run:")
    print(f"  poetry run python scripts/compute_agreement.py --labels <downloaded human_labels.json> --answer-key {answer_key_path}")


if __name__ == "__main__":
    asyncio.run(main())
