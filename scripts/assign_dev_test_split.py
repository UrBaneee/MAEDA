"""
Assign each golden case to a dev/test split, stratified on
(query_type, is_hard_case) -- docs/eval_v2_plan.md Step 4.

Idempotent and growth-safe: the authoritative record is
data/eval_split_manifest.json (case_id -> "dev"|"test"), not a fresh
random draw over the whole suite. Every run:
  1. Keeps every case_id already in the manifest exactly as it is --
     re-running this after the suite grows must never flip an existing
     case's split (that would compromise a holdout that's already been
     scored against, per the plan's own caveat).
  2. For any case_id NOT yet in the manifest (new cases added since the
     last run), tops up its (query_type, is_hard_case) bucket toward the
     60:40 target using a fixed-seed shuffle applied only to the new
     entries in that bucket.
  3. Writes the manifest, then syncs the "split" field in
     tests/eval/test_suite.json from the manifest (the manifest is the
     source of truth; the JSON's field is a synced, human-readable copy).

Usage:
    poetry run python scripts/assign_dev_test_split.py            # assign + apply
    poetry run python scripts/assign_dev_test_split.py --dry-run  # show plan only
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20260731  # fixed -- never change once real splits have been used
DEV_FRACTION = 0.6

SUITE_PATH = Path("tests/eval/test_suite.json")
MANIFEST_PATH = Path("data/eval_split_manifest.json")

HARD_TAGS = {"data_mismatch", "false_premise", "ambiguous", "cross_table_join"}


def _bucket_key(case: dict) -> tuple[str, bool]:
    is_hard = bool(set(case.get("tags", [])) & HARD_TAGS)
    return (case["query_type"], is_hard)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing anything")
    args = parser.parse_args()

    suite = json.loads(SUITE_PATH.read_text())
    manifest: dict[str, str] = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    by_bucket: dict[tuple[str, bool], list[dict]] = {}
    for case in suite:
        by_bucket.setdefault(_bucket_key(case), []).append(case)

    rng = random.Random(SEED)
    new_assignments: dict[str, str] = {}

    for bucket, cases in sorted(by_bucket.items()):
        already_dev = sum(1 for c in cases if manifest.get(c["id"]) == "dev")
        already_test = sum(1 for c in cases if manifest.get(c["id"]) == "test")
        unassigned = [c for c in cases if c["id"] not in manifest]
        if not unassigned:
            continue

        target_dev_total = round(DEV_FRACTION * len(cases))
        dev_needed = max(0, min(len(unassigned), target_dev_total - already_dev))

        shuffled = list(unassigned)
        rng.shuffle(shuffled)
        for i, case in enumerate(shuffled):
            new_assignments[case["id"]] = "dev" if i < dev_needed else "test"

        print(f"{bucket}: {len(cases)} total ({already_dev} dev / {already_test} test already assigned), "
              f"{len(unassigned)} new -> {dev_needed} dev / {len(unassigned) - dev_needed} test")

    if not new_assignments:
        print("\nNothing new to assign -- manifest already covers every case in the suite.")
    else:
        print(f"\n{len(new_assignments)} new assignment(s):")
        for cid, split in sorted(new_assignments.items()):
            print(f"  {cid}: {split}")

    full_manifest = {**manifest, **new_assignments}

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    MANIFEST_PATH.write_text(json.dumps(full_manifest, indent=2, sort_keys=True) + "\n")

    for case in suite:
        case["split"] = full_manifest[case["id"]]
    SUITE_PATH.write_text(json.dumps(suite, indent=2) + "\n")

    dev_n = sum(1 for v in full_manifest.values() if v == "dev")
    test_n = sum(1 for v in full_manifest.values() if v == "test")
    print(f"\nWrote {MANIFEST_PATH} and synced {SUITE_PATH}. "
          f"Suite split: {dev_n} dev / {test_n} test ({dev_n + test_n} total).")


if __name__ == "__main__":
    main()
