#!/usr/bin/env python3
"""Export a zero-API, anonymized relabeling page from a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.eval.metrics import render_data_quality, render_findings, render_rag_context
from src.eval.replay_cache import ReplayCache
from src.eval.runner import load_golden_suite


TEMPLATE = Path(__file__).resolve().parent / "_annotation_template.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--html-out", type=Path, required=True)
    parser.add_argument("--mapping-out", type=Path, required=True)
    parser.add_argument("--seed", default="dmxapi-calibration20-blind-relabel-v1")
    parser.add_argument("--blind-prefix", default="R")
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    original_ids = list(manifest["labels"])
    ordered = sorted(
        original_ids,
        key=lambda cid: hashlib.sha256(f"{args.seed}:{cid}".encode()).hexdigest(),
    )

    cache = ReplayCache()
    suite = {case.id: case for case in load_golden_suite()}
    missing = [cid for cid in ordered if cid not in cache._entries]
    if missing:
        raise SystemExit(f"Missing replay entries: {missing}")

    samples = []
    mapping = {}
    for index, cid in enumerate(ordered, 1):
        blind_id = f"{args.blind_prefix}{index:02d}"
        mapping[blind_id] = cid
        state = cache._entries[cid].state
        test_case = suite.get(cid)
        samples.append({
            "case_id": blind_id,
            "query_type": test_case.query_type if test_case else "unknown",
            "query": state.get("user_query", ""),
            "report": state.get("report") or "",
            "findings": render_findings(state.get("analysis_results") or []),
            "data_quality": render_data_quality(state.get("data_quality_report")),
            "rag_context": render_rag_context(state.get("rag_context") or []),
        })

    template = TEMPLATE.read_text()
    html = template.replace("/*__SAMPLES_JSON__*/", json.dumps(samples, ensure_ascii=False))
    html = html.replace("Judge calibration labeling", "Blind judge-reference relabeling")
    html = html.replace(
        "Read the query, report, and findings for each case, then rate",
        "Judge each dimension independently. Relevance ignores factual correctness: a direct but "
        "wrong answer can still be relevant. Groundedness ignores whether the report answered the "
        "question: check every factual/comparative claim against only the visible evidence. Then rate",
    )
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    html = html.replace(
        'const STORAGE_KEY = "maeda_judge_calibration_labels_pass2_v1";',
        f'const STORAGE_KEY = "maeda_blind_relabel_{manifest_hash[:16]}";',
    )

    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html)
    args.mapping_out.write_text(json.dumps({
        "schema_version": "1",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "seed": args.seed,
        "mapping": mapping,
    }, indent=2) + "\n")
    print(f"PASS blind_cases={len(samples)}")
    print(f"html={args.html_out}")
    print(f"mapping={args.mapping_out}")


if __name__ == "__main__":
    main()
