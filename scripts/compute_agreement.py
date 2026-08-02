"""
Compute human-vs-judge agreement (QWK / Spearman / MAE / confusion matrix)
for eval v2 Step 3 (docs/eval_v2_plan.md).

Takes a human_labels.json (downloaded from the labeling tool produced by
scripts/export_for_annotation.py) and the matching answer_key_<ts>.json
(the judge's own scores, kept hidden during labeling), pairs them by
case_id, and reports agreement for answer_relevance and groundedness
independently.

Usage:
    poetry run python scripts/compute_agreement.py \\
        --labels data/annotations/human_labels_1234567890.json \\
        --answer-key data/annotations/answer_key_1785447682.json

    # Compare two labeling sessions against each other instead of against
    # the judge -- the self-consistency check (re-label 24h+ later, see
    # docs/eval_v2_plan.md Step 3's "three non-negotiable process details").
    poetry run python scripts/compute_agreement.py \\
        --labels data/annotations/human_labels_session1.json \\
        --labels2 data/annotations/human_labels_session2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.agreement import compute_agreement, print_confusion_matrix

METRICS = ["answer_relevance", "groundedness"]


def _load_labels(path: Path) -> dict:
    data = json.loads(path.read_text())
    labels = data.get("labels", data)  # tolerate a bare {case_id: {...}} file too
    return labels


def _paired_scores(a: dict, b: dict, metric: str, a_name: str, b_name: str) -> tuple[list[float], list[float], list[str]]:
    common = sorted(set(a.keys()) & set(b.keys()))
    missing_a = sorted(set(b.keys()) - set(a.keys()))
    missing_b = sorted(set(a.keys()) - set(b.keys()))
    if missing_a:
        print(f"  (skipping {missing_a}, not present in {a_name})")
    if missing_b:
        print(f"  (skipping {missing_b}, not present in {b_name})")

    xs, ys, used = [], [], []
    for cid in common:
        av = a[cid].get(metric)
        bv = b[cid].get(metric)
        if av is None or bv is None:
            print(f"  (skipping {cid}: {metric} not rated in one of the two files)")
            continue
        xs.append(av)
        ys.append(bv)
        used.append(cid)
    return xs, ys, used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=str, required=True, help="Path to a human_labels.json file")
    parser.add_argument("--answer-key", type=str, default=None,
                        help="Path to the judge's answer_key_<ts>.json (compare human vs. judge)")
    parser.add_argument("--labels2", type=str, default=None,
                        help="Path to a second human_labels.json (compare human vs. human -- self-consistency)")
    parser.add_argument("--out", type=str, default=None, help="Path to save the JSON report (default: alongside --labels)")
    args = parser.parse_args()

    if bool(args.answer_key) == bool(args.labels2):
        print("Pass exactly one of --answer-key (human vs. judge) or --labels2 (human vs. human).")
        sys.exit(1)

    human = _load_labels(Path(args.labels))

    if args.answer_key:
        raw_key = json.loads(Path(args.answer_key).read_text())
        other = {cid: {m: v.get("score") for m, v in entry.items() if m in METRICS}
                 for cid, entry in raw_key.items()}
        other_name, comparison = "judge", "human vs. judge"
    else:
        other = _load_labels(Path(args.labels2))
        other_name, comparison = "second labeling session", "human vs. human (self-consistency)"

    print(f"Comparing {args.labels} against {other_name} ({comparison})\n")

    reports = {}
    for metric in METRICS:
        print(f"--- {metric} ---")
        human_scores, other_scores, used_ids = _paired_scores(human, other, metric, "labels", other_name)
        if len(used_ids) < 2:
            print(f"  Not enough paired samples for {metric} (n={len(used_ids)}) -- skipping.\n")
            continue
        report = compute_agreement(human_scores, other_scores, metric=metric)
        reports[metric] = report
        print(f"  {report.summary_line()}")
        print_confusion_matrix(report)
        print()

    if not reports:
        print("No metric had enough paired samples to compute agreement.")
        sys.exit(1)

    out = {
        "labels_file": args.labels,
        "compared_against": args.answer_key or args.labels2,
        "comparison_type": comparison,
        "metrics": {m: r.to_dict() for m, r in reports.items()},
    }
    out_path = Path(args.out) if args.out else Path(args.labels).with_suffix("").with_name(
        Path(args.labels).stem + "_agreement.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
