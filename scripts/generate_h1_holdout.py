#!/usr/bin/env python3
"""
Generate the H1 holdout (附录 HA.5, HB; 预注册 §10.14): 119 sales questions, split "test",
truths from the CLEAN table (data/truth/sales_data_clean.csv, sha256 6c953344...), pipeline input
the dirty data/demo/sales_data.csv. Generated from templates with a fixed seed; nothing is
hand-picked or edited afterwards. Cases are dropped only for a truth-computation failure, an
ambiguous truth, or a duplicate of a dev case; every drop is listed with its reason in the
manifest and the next candidate in the seeded order takes its place.

The type mix is tests/eval/h1_sample.json's 40 cases scaled to 119 by largest remainder.
Writes tests/eval/h1_holdout_test.json and tests/eval/h1_holdout_manifest.json. Makes no API call
and runs nothing on the generated cases (reveal guard, src/eval/reveal_guard.py).

    PYTHONPATH=. python scripts/generate_h1_holdout.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_ground_truth as cg  # noqa: E402

SEED = 20261106
N_TOTAL = 119
OUT = Path("tests/eval/h1_holdout_test.json")
MANIFEST = Path("tests/eval/h1_holdout_manifest.json")
REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = ["Widget A", "Widget B", "Gadget Pro", "Gadget Lite", "Service Pack"]
CHANNELS = ["Direct", "Partner", "Online", "Retail"]
YEARS = [2022, 2023, 2024]
DIMS = {"region": REGIONS, "product": PRODUCTS, "channel": CHANNELS}
ENTITIES = [(d, v) for d, vs in DIMS.items() for v in vs]

#: type -> 40-case count in h1_sample.json (GX.3 classification)
TYPE_COUNTS_40 = {"收入合计": 23, "收入均值": 4, "销量合计": 6, "销量均值": 1, "计数": 2,
                  "排名(销售员销量)": 1, "排名(销售员收入)": 1, "占比": 1, "增长率": 1}


def quotas(total: int = N_TOTAL) -> dict[str, int]:
    raw = {k: v * total / 40 for k, v in TYPE_COUNTS_40.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[: total - sum(out.values())]:
        out[k] += 1
    return out


def slug(*parts) -> str:
    return "_".join(str(p).lower().replace(" ", "_") for p in parts)


def sel(df: pd.DataFrame, dim: str, val: str) -> pd.DataFrame:
    return df[df[dim] == val]


def yr(df, y):
    return df[df["year"] == y]


R2 = lambda x: round(float(x), 2)  # noqa: E731


def pairs(dim):
    return list(itertools.combinations(DIMS[dim], 2))


# ── templates: each yields a pool of (params) and a build(df, params) -> (query, truth, meta) ──────

def T_rev_dim_year():
    pool = [(d, y) for d in DIMS for y in YEARS]
    def build(df, p):
        d, y = p; g = yr(df, y).groupby(d)["revenue"].sum()
        return (f"What was total revenue for each {d} in {y}?",
                {slug(v, y, "revenue"): R2(g[v]) for v in DIMS[d]}, ["revenue"], [d, "year"], "bar")
    return pool, build


def T_rev_entity_year():
    pool = [(e, y) for e in ENTITIES for y in YEARS]
    def build(df, p):
        (d, v), y = p
        return (f"What was the total revenue of {v} ({d}) in {y}?",
                {slug(v, y, "revenue"): R2(yr(sel(df, d, v), y)["revenue"].sum())}, ["revenue"], [d, "year"], "")
    return pool, build


def T_rev_entity_quarter():
    pool = [(e, y, q) for e in ENTITIES for y in YEARS for q in (1, 2, 3, 4)]
    def build(df, p):
        (d, v), y, q = p
        t = sel(df, d, v); t = t[(t["year"] == y) & (t["quarter"] == q)]
        return (f"How much revenue did {v} ({d}) bring in during Q{q} {y}?",
                {slug(v, y, f"q{q}", "revenue"): R2(t["revenue"].sum())}, ["revenue"], [d, "quarter"], "")
    return pool, build


def T_rev_pair_year():
    pool = [(d, a, b, y) for d in DIMS for a, b in pairs(d) for y in YEARS]
    def build(df, p):
        d, a, b, y = p; t = yr(df, y)
        return (f"Compare total revenue of {a} and {b} ({d}) in {y}",
                {slug(a, y, "revenue"): R2(sel(t, d, a)["revenue"].sum()),
                 slug(b, y, "revenue"): R2(sel(t, d, b)["revenue"].sum())}, ["revenue"], [d], "bar")
    return pool, build


def T_rev_cross():
    combos = [("product", "channel"), ("region", "product"), ("region", "channel")]
    pool = [(da, a, db, b) for da, db in combos for a in DIMS[da] for b in DIMS[db]]
    def build(df, p):
        da, a, db, b = p
        return (f"What is the total revenue for {a} ({da}) within {b} ({db})?",
                {slug(a, b, "revenue"): R2(sel(sel(df, da, a), db, b)["revenue"].sum())}, ["revenue"], [da, db], "")
    return pool, build


def T_rev_best():
    pool = [(d, y, k) for d in DIMS for y in YEARS for k in ("highest", "lowest")]
    def build(df, p):
        d, y, k = p; g = yr(df, y).groupby(d)["revenue"].sum()
        v = g.max() if k == "highest" else g.min()
        return (f"Which {d} had the {k} total revenue in {y}, and how much was it?",
                {slug(k, d, y, "revenue"): R2(v)}, ["revenue"], [d, "year"], "bar")
    return pool, build


def T_rev_halves():
    pool = [(r, y) for r in REGIONS for y in YEARS]
    def build(df, p):
        r, y = p; t = yr(sel(df, "region", r), y)
        return (f"Compare revenue in the first half and the second half of {y} for the {r} region",
                {slug(r, y, "h1", "revenue"): R2(t[t["quarter"] <= 2]["revenue"].sum()),
                 slug(r, y, "h2", "revenue"): R2(t[t["quarter"] >= 3]["revenue"].sum())},
                ["revenue"], ["region", "quarter"], "bar")
    return pool, build


def T_avg_entity_year():
    pool = [(e, y) for e in ENTITIES for y in YEARS]
    def build(df, p):
        (d, v), y = p
        return (f"What was the average revenue per transaction for {v} ({d}) in {y}?",
                {slug(v, y, "avg_revenue"): R2(yr(sel(df, d, v), y)["revenue"].mean())}, ["revenue"], [d, "year"], "")
    return pool, build


def T_avg_dim_year():
    pool = [(d, y) for d in DIMS for y in YEARS]
    def build(df, p):
        d, y = p; g = yr(df, y).groupby(d)["revenue"].mean()
        return (f"Show the average revenue per transaction by {d} in {y}",
                {slug(v, y, "avg_revenue"): R2(g[v]) for v in DIMS[d]}, ["revenue"], [d, "year"], "bar")
    return pool, build


def T_avg_pair():
    pool = [(d, a, b) for d in DIMS for a, b in pairs(d)]
    def build(df, p):
        d, a, b = p
        return (f"Compare the average revenue per transaction of {a} and {b} ({d})",
                {slug(a, "avg_revenue"): R2(sel(df, d, a)["revenue"].mean()),
                 slug(b, "avg_revenue"): R2(sel(df, d, b)["revenue"].mean())}, ["revenue"], [d], "bar")
    return pool, build


def T_units_dim_year():
    pool = [(d, y) for d in DIMS for y in YEARS]
    def build(df, p):
        d, y = p; g = yr(df, y).groupby(d)["units"].sum()
        return (f"How many units were sold for each {d} in {y}?",
                {slug(v, y, "units"): int(g[v]) for v in DIMS[d]}, ["units"], [d, "year"], "bar")
    return pool, build


def T_units_entity_year():
    pool = [(e, y) for e in ENTITIES for y in YEARS]
    def build(df, p):
        (d, v), y = p
        return (f"What was the total number of units sold for {v} ({d}) in {y}?",
                {slug(v, y, "units"): int(yr(sel(df, d, v), y)["units"].sum())}, ["units"], [d, "year"], "")
    return pool, build


def T_units_pair_year():
    pool = [(d, a, b, y) for d in DIMS for a, b in pairs(d) for y in YEARS]
    def build(df, p):
        d, a, b, y = p; t = yr(df, y)
        return (f"Compare units sold for {a} and {b} ({d}) in {y}",
                {slug(a, y, "units"): int(sel(t, d, a)["units"].sum()),
                 slug(b, y, "units"): int(sel(t, d, b)["units"].sum())}, ["units"], [d], "bar")
    return pool, build


def T_units_cross():
    combos = [("product", "channel"), ("region", "product"), ("region", "channel")]
    pool = [(da, a, db, b) for da, db in combos for a in DIMS[da] for b in DIMS[db]]
    def build(df, p):
        da, a, db, b = p
        return (f"How many units of {a} ({da}) were sold within {b} ({db})?",
                {slug(a, b, "units"): int(sel(sel(df, da, a), db, b)["units"].sum())}, ["units"], [da, db], "")
    return pool, build


def T_avgunits_entity():
    pool = [e for e in ENTITIES]
    def build(df, p):
        d, v = p
        return (f"What is the average number of units per transaction for {v} ({d})?",
                {slug(v, "avg_units"): R2(sel(df, d, v)["units"].mean())}, ["units"], [d], "")
    return pool, build


def T_avgunits_dim():
    pool = [d for d in DIMS]
    def build(df, d):
        g = df.groupby(d)["units"].mean()
        return (f"Show the average number of units per transaction by {d}",
                {slug(v, "avg_units"): R2(g[v]) for v in DIMS[d]}, ["units"], [d], "bar")
    return pool, build


def T_count_entity_year():
    pool = [(e, y) for e in ENTITIES for y in YEARS]
    def build(df, p):
        (d, v), y = p
        return (f"How many transactions were recorded for {v} ({d}) in {y}?",
                {slug(v, y, "transactions"): int(len(yr(sel(df, d, v), y)))}, ["transactions"], [d, "year"], "")
    return pool, build


def T_count_pair_year():
    pool = [(d, a, b, y) for d in DIMS for a, b in pairs(d) for y in YEARS]
    def build(df, p):
        d, a, b, y = p; t = yr(df, y)
        return (f"Compare the number of transactions for {a} and {b} ({d}) in {y}",
                {slug(a, y, "transactions"): int(len(sel(t, d, a))),
                 slug(b, y, "transactions"): int(len(sel(t, d, b)))}, ["transactions"], [d], "bar")
    return pool, build


def _rank_pool():
    return [("year", y) for y in YEARS] + [("region", r) for r in REGIONS]


def _scope(df, p):
    k, v = p
    return (yr(df, v) if k == "year" else sel(df, "region", v)), (f"in {v}" if k == "year" else f"in the {v} region")


def T_rank_units():
    def build(df, p):
        t, where = _scope(df, p)
        g = t.groupby("rep_id")["units"].sum().sort_values(ascending=False)
        if g.iloc[2] == g.iloc[3]:
            raise ValueError("tie at the third place")
        truth = {}
        for i, (rep, v) in enumerate(g.head(3).items(), 1):
            truth[f"top{i}_rep_id"] = int(rep); truth[f"top{i}_rep_units"] = int(v)
        return (f"Who are the top 3 sales reps by units sold {where}?", truth, ["units"], ["rep_id"], "bar")
    return _rank_pool(), build


def T_rank_rev():
    def build(df, p):
        t, where = _scope(df, p)
        g = t.groupby("rep_id")["revenue"].sum().sort_values(ascending=False)
        if g.iloc[0] == g.iloc[1]:
            raise ValueError("tie for first")
        return (f"Which sales rep generated the most revenue {where}? Give the rep id and the revenue",
                {"top_rep_id": int(g.index[0]), "top_rep_revenue": R2(g.iloc[0])}, ["revenue"], ["rep_id"], "bar")
    return _rank_pool(), build


def T_share():
    pool = [(e, y) for e in ENTITIES for y in YEARS]
    def build(df, p):
        (d, v), y = p; t = yr(df, y); part = sel(t, d, v)["revenue"].sum()
        return (f"What share of total revenue in {y} came from {v} ({d})?",
                {slug(v, y, "revenue"): R2(part), slug(v, y, "share_pct"): round(float(part / t["revenue"].sum() * 100), 1)},
                ["revenue"], [d, "year"], "bar")
    return pool, build


def T_growth():
    pool = [(e, a, b) for e in ENTITIES for a, b in ((2022, 2023), (2023, 2024), (2022, 2024))]
    def build(df, p):
        (d, v), a, b = p; t = sel(df, d, v)
        ra, rb = yr(t, a)["revenue"].sum(), yr(t, b)["revenue"].sum()
        return (f"By how many percent did revenue for {v} ({d}) grow from {a} to {b}?",
                {slug(v, a, "revenue"): R2(ra), slug(v, b, "revenue"): R2(rb),
                 slug(v, a, b, "growth_pct"): round(float((rb - ra) / ra * 100), 1)}, ["revenue"], [d, "year"], "line")
    return pool, build


TEMPLATES = {
    "收入合计": [T_rev_dim_year, T_rev_entity_year, T_rev_entity_quarter, T_rev_pair_year, T_rev_cross, T_rev_best, T_rev_halves],
    "收入均值": [T_avg_entity_year, T_avg_dim_year, T_avg_pair],
    "销量合计": [T_units_dim_year, T_units_entity_year, T_units_pair_year, T_units_cross],
    "销量均值": [T_avgunits_entity, T_avgunits_dim],
    "计数": [T_count_entity_year, T_count_pair_year],
    "排名(销售员销量)": [T_rank_units],
    "排名(销售员收入)": [T_rank_rev],
    "占比": [T_share],
    "增长率": [T_growth],
}


def dev_signatures() -> tuple[set, set]:
    """Truth-value sets and normalized queries of every dev case, to refuse duplicates."""
    dev = json.loads(Path("tests/eval/test_suite.json").read_text())
    vals = {tuple(sorted(round(float(v), 2) for k, v in c["ground_truth"].items() if isinstance(v, (int, float)))) for c in dev}
    qs = {" ".join(c["query"].lower().split()) for c in dev}
    return {v for v in vals if v}, qs


def generate() -> tuple[list[dict], dict]:
    df = cg._Data().sales
    rng = random.Random(SEED)
    dev_vals, dev_qs = dev_signatures()
    seen_vals: set = set()
    seen_q: set = set()
    cases, exclusions, per_type = [], [], {}
    for typ, quota in quotas().items():
        tmpls = [t() for t in TEMPLATES[typ]]
        names = [t.__name__ for t in TEMPLATES[typ]]
        pools = [list(p) for p, _ in tmpls]
        for pl in pools:
            rng.shuffle(pl)
        got = 0
        while got < quota:
            live = [i for i, pl in enumerate(pools) if pl]
            if not live:
                exclusions.append({"type": typ, "reason": "candidate pool exhausted before the quota was filled"})
                break
            i = rng.choice(live)
            params = pools[i].pop()
            ref = f"{names[i]}{json.dumps(params, ensure_ascii=False, default=str)}"
            try:
                query, truth, metrics, dims, chart = tmpls[i][1](df, params)
                vals = list(truth.values())
                if not vals or any(pd.isna(v) for v in vals):
                    raise ValueError("empty or NaN truth")
                if len(set(vals)) != len(vals):
                    raise ValueError("two truth fields share a value (scoring ambiguity)")
                if any(v == 0 for v in vals):
                    raise ValueError("zero-valued truth field")
            except Exception as exc:  # noqa: BLE001
                exclusions.append({"type": typ, "candidate": ref, "reason": f"truth computation: {exc}"})
                continue
            key = tuple(sorted(round(float(v), 2) for v in vals))
            nq = " ".join(query.lower().split())
            if key in dev_vals or nq in dev_qs:
                exclusions.append({"type": typ, "candidate": ref, "reason": "duplicates a dev case (same truth values or query)"})
                continue
            if key in seen_vals or nq in seen_q:
                exclusions.append({"type": typ, "candidate": ref, "reason": "duplicates another holdout candidate"})
                continue
            seen_vals.add(key); seen_q.add(nq)
            cases.append({"_type": typ, "_template": names[i], "_params": ref, "query": query,
                          "query_type": "comparative" if ("Compare" in query or "grow" in query) else "descriptive",
                          "expected_metrics": metrics, "expected_dimensions": dims, "ground_truth": truth,
                          "tags": ["h1_holdout", "comparative" if "Compare" in query else "descriptive"],
                          "data_source": {"type": "csv", "path": "data/demo/sales_data.csv"},
                          "expected_tools": ["pandas_transform"] + (["comparison"] if "Compare" in query else []),
                          "expected_chart_types": [chart] if chart else [], "split": "test"})
            got += 1
        per_type[typ] = got
    rng.shuffle(cases)  # ids carry no type information
    for n, c in enumerate(cases, 1):
        c["id"] = f"H1T-{n:03d}"
    meta = {"seed": SEED, "target": N_TOTAL, "quotas": quotas(), "generated": per_type,
            "final_count": len(cases), "exclusions": exclusions}
    return cases, meta


def main() -> int:
    cases, meta = generate()
    public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases]
    OUT.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
    meta["cases_sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta["clean_table_sha256"] = cg.CLEAN_SALES_SHA256
    meta["template_use"] = {}
    for c in cases:
        meta["template_use"][c["_template"]] = meta["template_use"].get(c["_template"], 0) + 1
    meta["case_provenance"] = {c["id"]: {"type": c["_type"], "template": c["_template"], "params": c["_params"]} for c in cases}
    MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases ({OUT}); {len(meta['exclusions'])} exclusion(s); sha {meta['cases_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
