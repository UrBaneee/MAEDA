#!/usr/bin/env python3
"""Decode anonymized relabel IDs using the separately stored mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.labels.read_text())
    mapping_payload = json.loads(args.mapping.read_text())
    mapping = mapping_payload["mapping"]
    labels = payload["labels"]
    if set(labels) != set(mapping):
        raise SystemExit(
            f"Blind ID mismatch: labels-only={sorted(set(labels) - set(mapping))}, "
            f"mapping-only={sorted(set(mapping) - set(labels))}"
        )
    decoded = {mapping[blind_id]: value for blind_id, value in labels.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "exported_at": payload.get("exported_at"),
        "n_cases": len(decoded),
        "mapping_manifest_sha256": mapping_payload["manifest_sha256"],
        "labels": decoded,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS decoded={len(decoded)} out={args.out}")


if __name__ == "__main__":
    main()
