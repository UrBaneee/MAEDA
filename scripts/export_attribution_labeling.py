#!/usr/bin/env python3
"""
Build the blind labelling page for H2 attribution (预注册 R5.1, 附录 FL) from a
reports JSONL. The arm, case id and refusal status are left out of the page;
the mapping from blind id to job is written to a separate file that is never
published. Makes no API call.

    python scripts/export_attribution_labeling.py --set dry
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("data/h2_attribution")
TEMPLATE = Path(__file__).resolve().parent / "_attribution_template.html"
CASES = Path("tests/eval/h2_dev_attribution_cases.json")
SETS = {"dry": ("reports_dry.jsonl", "Dry run"), "pass1": ("reports_formal.jsonl", "Pass 1"),
        "pass2": ("reports_formal.jsonl", "Pass 2")}
# Blind-id prefix per set. pass1 and pass2 share a first letter, so pass2 gets
# its own prefix: the same id must never name two different reports across passes.
PREFIX = {"dry": "D", "pass1": "P", "pass2": "Q"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=list(SETS), required=True)
    args = parser.parse_args()
    source, title = SETS[args.set]
    cases = {c["id"]: c for c in json.loads(CASES.read_text())}
    rows = [json.loads(line) for line in (ROOT / source).read_text().splitlines()]
    rows = [r for r in rows if r["report"]]
    seed = f"h2-attribution-{args.set}-v1"
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['job_id']}".encode()).hexdigest())
    samples, mapping = [], {}
    for i, row in enumerate(rows, 1):
        blind = f"{PREFIX[args.set]}{i:02d}"
        case = cases[row["case_id"]]
        mapping[blind] = row["job_id"]
        samples.append({"id": blind, "query": case["query"], "anomaly": case["event_anomaly"],
                        "cause": case["event_cause"], "report": row["report"]})
    page = (TEMPLATE.read_text()
            .replace("__SET_TITLE__", title)
            .replace("__COLLECTION__", f"labels_{args.set}")
            .replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")))
    (ROOT / f"labeling_{args.set}.html").write_text(page)
    (ROOT / f"labeling_{args.set}.mapping.json").write_text(
        json.dumps({"seed": seed, "source": source, "mapping": mapping}, indent=2))
    print(f"{len(samples)} reports -> {ROOT / f'labeling_{args.set}.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
