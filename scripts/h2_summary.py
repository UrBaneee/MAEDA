#!/usr/bin/env python3
"""
H2 summary over saved runs (预注册 §7.2.1, 附录 FM). No API call.

    python scripts/h2_summary.py --set formal --labels judge
    python scripts/h2_summary.py --set formal --labels human \
        --human-dir data/h2_attribution/labels_pass1_snapshot \
        --mapping data/h2_attribution/labeling_pass1.mapping.json

Human labels are keyed by blind id; the mapping file turns them back into job
ids. Prints JSON; --out also writes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.h2_summary import summarise_h2  # noqa: E402

DEFAULT_DIR = Path("data/h2_attribution")


def _human_labels(label_dir: Path, mapping_path: Path) -> dict[str, str]:
    mapping = json.loads(mapping_path.read_text())["mapping"]
    labels = {}
    for f in label_dir.glob("*.json"):
        doc = json.loads(f.read_text())
        doc = doc.get("data", doc)
        if doc.get("label"):
            labels[mapping[f.stem]] = doc["label"]
    return labels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", choices=["dry", "formal", "reveal"], required=True)
    p.add_argument("--labels", choices=["judge", "human"], required=True)
    p.add_argument("--human-dir", type=Path)
    p.add_argument("--mapping", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help="directory holding reports_<set>.jsonl and judge_<set>.jsonl (附录 HJ)")
    a = p.parse_args()
    runs = [json.loads(line) for line in (a.dir / f"reports_{a.set}.jsonl").open()]
    if a.labels == "judge":
        rows = map(json.loads, (a.dir / f"judge_{a.set}.jsonl").open())
        labels = {r["job_id"]: r["label"] for r in rows if r.get("valid") and r.get("label")}
    else:
        if not (a.human_dir and a.mapping):
            p.error("--labels human needs --human-dir and --mapping")
        labels = _human_labels(a.human_dir, a.mapping)
    out = json.dumps(summarise_h2(runs, labels).to_dict(), ensure_ascii=False, indent=2)
    print(out)
    if a.out:
        a.out.write_text(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
