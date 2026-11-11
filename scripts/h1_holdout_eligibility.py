#!/usr/bin/env python3
"""
预注册 §7.4 eligibility fields for the H1 holdout (附录 HM.3 item 2, HO). Computed from frozen
inputs only, without running the system on any case: each case's own ground-truth builder
(scripts/generate_h1_holdout.py, looked up through tests/eval/h1_holdout_manifest.json) is
evaluated on the dirty pipeline input data/demo/sales_data.csv, raw and repaired.

  cleaning_relevant  the case's data slice holds at least one row of an injected defect class
                     named in §7.4: missing revenue, negative revenue, or a revenue / units
                     value outside the declared IQR fences. A slice "holds" a defect row when
                     either dropping all defect rows, or adding 1e6 to their revenue and units,
                     changes any truth field of the case.
  cleaning_material  the §7.4 repair (drop null-revenue rows, drop negative-revenue rows, clip
                     revenue and units to the declared fences) moves at least one truth field
                     by more than the 1% scoring tolerance (src.eval.metrics._numbers_match);
                     non-numeric fields (rep ids) by any change.
  corpus_backed      false for every case: the holdout is sales-only and no case is linked to
                     an event (declared unlinked at authoring, 附录 HB).

Declared IQR fences: 1.5 x IQR over the full dirty table, per column, the same convention as
compute_ground_truth._iqr_outlier_count; the values are written into the output.

Disclosure kept in the output: §7.4's defect list does not name the injected duplicate rows, so
the repair above does not deduplicate. A sensitivity field `cleaning_material_with_dedup` adds
exact-duplicate removal to the repair; it is descriptive and labels nothing.

Writes tests/eval/h1_holdout_eligibility.json. The holdout file itself is not changed.

    PYTHONPATH=. python scripts/h1_holdout_eligibility.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_h1_holdout as gen  # noqa: E402
from src.eval.metrics import _numbers_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIRTY = ROOT / "data/demo/sales_data.csv"
DIRTY_SHA256 = "b863b60173f3c0bf30c4582738b85ba3304a7a212794d66ab14aa35dce422719"
MANIFEST = ROOT / "tests/eval/h1_holdout_manifest.json"
OUT = ROOT / "tests/eval/h1_holdout_eligibility.json"
BUMP = 1e6


def load_dirty() -> pd.DataFrame:
    actual = hashlib.sha256(DIRTY.read_bytes()).hexdigest()
    if actual != DIRTY_SHA256:
        raise SystemExit(f"{DIRTY} sha256 {actual} != {DIRTY_SHA256}")
    df = pd.read_csv(DIRTY)
    df["date"] = pd.to_datetime(df["date"])
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    return df


def fences(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for col in ("revenue", "units"):
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        out[col] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))
    return out


def defect_mask(df: pd.DataFrame, f: dict) -> pd.Series:
    rev, units = df["revenue"], df["units"]
    out_rev = rev.notna() & ((rev < f["revenue"][0]) | (rev > f["revenue"][1]))
    out_units = (units < f["units"][0]) | (units > f["units"][1])
    return rev.isna() | (rev < 0) | out_rev | out_units


def repaired(df: pd.DataFrame, f: dict, dedup: bool = False) -> pd.DataFrame:
    t = df[df["revenue"].notna() & (df["revenue"] >= 0)].copy()
    t["revenue"] = t["revenue"].clip(*f["revenue"])
    t["units"] = t["units"].clip(*f["units"])
    if dedup:
        t = t.drop_duplicates()
    return t


def _same(a, b) -> bool:
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        if pd.isna(a) or pd.isna(b):
            return pd.isna(a) and pd.isna(b)
        return _numbers_match(float(b), float(a))
    return a == b


def truths_differ(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    return set(a) != set(b) or any(not _same(a[k], b[k]) for k in a)


def builders() -> dict:
    return {fn.__name__: fn()[1] for fns in gen.TEMPLATES.values() for fn in fns}


def parse(ref: str) -> tuple[str, list]:
    i = min(ref.index(c) for c in '["' if c in ref)  # params are a JSON list or a JSON scalar
    return ref[:i], json.loads(ref[i:])


def truth(build, df, params) -> dict | None:
    try:
        return build(df, params)[1]
    except Exception:  # noqa: BLE001  a builder can fail on a perturbed table (e.g. a new tie)
        return None


def main() -> int:
    df = load_dirty()
    f = fences(df)
    mask = defect_mask(df, f)
    bumped = df.copy()
    bumped.loc[mask, "revenue"] = bumped.loc[mask, "revenue"].fillna(0) + BUMP
    bumped.loc[mask, "units"] = bumped.loc[mask, "units"] + BUMP
    variants = {"dropped": df[~mask], "bumped": bumped,
                "repaired": repaired(df, f), "repaired_dedup": repaired(df, f, dedup=True)}
    build = builders()
    prov = json.loads(MANIFEST.read_text())["case_provenance"]
    rows = {}
    for cid in sorted(prov):
        name, params = parse(prov[cid]["params"])
        raw = truth(build[name], df, params)
        if raw is None:
            raise SystemExit(f"{cid}: builder failed on the raw table")
        t = {k: truth(build[name], v, params) for k, v in variants.items()}
        rows[cid] = {
            "cleaning_relevant": truths_differ(raw, t["dropped"]) or truths_differ(raw, t["bumped"]),
            "cleaning_material": truths_differ(raw, t["repaired"]),
            "cleaning_material_with_dedup": truths_differ(raw, t["repaired_dedup"]),
            "event_id": None,
            "corpus_backed": False,
        }
    count = {k: sum(r[k] for r in rows.values()) for k in
             ("cleaning_relevant", "cleaning_material", "cleaning_material_with_dedup", "corpus_backed")}
    out = {
        "source": "预注册 §7.4; 附录 HO", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dirty_table_sha256": DIRTY_SHA256,
        "holdout_sha256": hashlib.sha256((ROOT / "tests/eval/h1_holdout_test.json").read_bytes()).hexdigest(),
        "declared_iqr_fences": {k: [round(a, 4), round(b, 4)] for k, (a, b) in f.items()},
        "defect_rows": int(mask.sum()), "n_cases": len(rows), "counts": count, "cases": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("declared_iqr_fences", "defect_rows", "n_cases", "counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
