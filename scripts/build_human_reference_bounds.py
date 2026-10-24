#!/usr/bin/env python3
"""Build strict/lenient human-reference bounds with adjudication overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("answer_relevance", "groundedness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--strict-out", type=Path, required=True)
    parser.add_argument("--lenient-out", type=Path, required=True)
    args = parser.parse_args()

    pass1 = json.loads(args.pass1.read_text())["labels"]
    pass2 = json.loads(args.pass2.read_text())["labels"]
    adjudication = json.loads(args.adjudication.read_text())["labels"]
    if set(pass1) != set(pass2):
        raise SystemExit("Pass 1 and pass 2 case IDs differ")
    if not set(adjudication).issubset(pass1):
        raise SystemExit("Adjudication contains unknown case IDs")

    outputs = {"strict_lower": {}, "lenient_upper": {}}
    for cid in pass1:
        outputs["strict_lower"][cid] = {}
        outputs["lenient_upper"][cid] = {}
        for metric in METRICS:
            if cid in adjudication:
                lower = upper = adjudication[cid][metric]
                source = "third_pass_adjudication"
            else:
                lower = min(pass1[cid][metric], pass2[cid][metric])
                upper = max(pass1[cid][metric], pass2[cid][metric])
                source = "human_pass1_pass2_bounds"
            outputs["strict_lower"][cid][metric] = lower
            outputs["lenient_upper"][cid][metric] = upper
            outputs["strict_lower"][cid].setdefault("notes", source)
            outputs["lenient_upper"][cid].setdefault("notes", source)

    for mode, path in (("strict_lower", args.strict_out), ("lenient_upper", args.lenient_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "reference_mode": mode,
            "n_cases": len(pass1),
            "rule": (
                "Third-pass adjudication overrides both bounds for five material disagreements; "
                "all other values use min/max of blind human passes 1 and 2."
            ),
            "labels": outputs[mode],
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"PASS mode={mode} cases={len(pass1)} out={path}")


if __name__ == "__main__":
    main()
