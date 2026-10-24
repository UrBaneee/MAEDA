#!/usr/bin/env python3
"""Merge a checkpointed judge run with a successful resume artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    resume = json.loads(args.resume.read_text())
    expected = set(base["metadata"]["case_ids"])
    scores = dict(base["scores"])
    scores.update(resume["scores"])
    missing = sorted(expected - set(scores))
    extra = sorted(set(scores) - expected)
    invalid = sorted(
        (cid, metric)
        for cid, case in scores.items()
        for metric, value in case.items()
        if isinstance(value, dict) and "valid" in value and not value["valid"]
    )
    if missing or extra or invalid:
        raise SystemExit(f"Merge invalid: missing={missing}, extra={extra}, invalid={invalid}")

    metadata = dict(base["metadata"])
    metadata.update({
        "status": "complete",
        "invalid_metrics": [],
        "resumed_from": str(args.resume),
        "resume_case_ids": sorted(resume["scores"]),
    })
    payload = {"metadata": metadata, "scores": scores}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, args.out)
    print(f"PASS merged={len(scores)} out={args.out}")


if __name__ == "__main__":
    main()
