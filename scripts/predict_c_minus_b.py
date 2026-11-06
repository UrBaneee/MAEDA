"""Zero-cost per-case prediction of C minus B on factual_accuracy (附录 GW.5, GX). Applies the llm planner's observed cleaning plan (dedup; revenue median-filled in 27 of 32 observed plans; no outlier clipping in 32 of 32) to the dirty CSV, computes each case's truth-function answer on the dirty and cleaned tables, and scores both with score_factual_accuracy. The emulation is checked against the real cleaner outputs in artifacts/cleaner_runs. No API call.

    PYTHONPATH=. python scripts/predict_c_minus_b.py
"""
import sys, json, glob, os, hashlib
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import pandas as pd, numpy as np
import compute_ground_truth as cg
from h1_sales_cases import SPEC
from src.eval.metrics import score_factual_accuracy

dirty = pd.read_csv("data/demo/sales_data.csv")

def plan_variant(df, fill_revenue):
    d = df.copy()
    if fill_revenue:
        d["revenue"] = d["revenue"].fillna(d["revenue"].median())
    d = d.drop_duplicates()
    return d.reset_index(drop=True)

# verify emulation against real artifacts
base = "artifacts/cleaner_runs"
final = "stage2/data_versions/sales_data__handle_missing__normalize_dates__deduplicate__handle_outliers.csv"
res = {}
for f in glob.glob(base + "/*_clean1/plan_stage2.json"):
    p = json.load(open(f))
    steps = {s["mcp_tool"]: s for s in p["steps"]}
    if "handle_missing" not in steps or p.get("planner_mode_used") != "llm":
        continue
    fill = "revenue" not in steps["handle_missing"]["params"]["skip_columns"]
    csvp = os.path.join(os.path.dirname(f), final)
    if not os.path.exists(csvp): continue
    real = pd.read_csv(csvp)
    emu = plan_variant(dirty, fill)
    ok = len(real) == len(emu) and np.allclose(real["revenue"].fillna(-1).values, emu["revenue"].fillna(-1).values) and (real["units"].values == emu["units"].values).all()
    res.setdefault((fill), []).append((len(real), bool(ok)))
print("emulation vs real artifacts (fill_revenue -> [(rows, match)]):", {k: (len(v), set(v)) for k, v in res.items()})

class D(cg._Data):
    def __init__(self, df):
        super().__init__()
        x = df.copy(); x["date"] = pd.to_datetime(x["date"]); x["quarter"] = x["date"].dt.quarter; x["year"] = x["date"].dt.year
        self._sales = x

truth_d = cg._Data()
def answers(df):
    d = D(df); out = {}
    for cid, fn in cg.COMPUTERS.items():
        out[cid] = fn(d)
    return out

sample = json.load(open("tests/eval/h1_sample.json"))["case_ids"]
suite = {c["id"]: c for c in json.load(open("tests/eval/test_suite.json"))}
A = {"clean": answers(pd.read_csv("data/truth/sales_data_clean.csv")),
     "B": answers(dirty),
     "C_fill": answers(plan_variant(dirty, True)),
     "C_nofill": answers(plan_variant(dirty, False))}

def fmt(v):
    return f"{v:,}" if isinstance(v, int) else f"{v:,.2f}"
def score(cid, ans):
    gt = {k: v for k, v in A["clean"][cid].items() if isinstance(v, (int, float))}
    report = "Findings: " + "; ".join(f"{k.replace('_',' ')} is {fmt(ans[k]) if isinstance(ans[k], int) else fmt(round(float(ans[k]),2))}" for k in gt)
    return score_factual_accuracy(report, [], gt).score

typ = {}
for c in "D01 D04 DG01 DG03 C01 C07 C16 S01 S02 S05 S06 S09 S10 S13 S14 S15 S16 S19 S20 S21 S23 S28 S30".split(): typ[c] = "收入合计"
for c in "S03 S12 S24 S29".split(): typ[c] = "收入均值"
for c in "D11 C11 S04 S11 S17 S22".split(): typ[c] = "销量合计"
typ["S08"] = "销量均值"
for c in "S07 S25".split(): typ[c] = "计数"
typ["D10"] = "排名(销售员收入)"; typ["S18"] = "排名(销售员销量)"
typ["S26"] = "占比"; typ["S27"] = "增长率"
rows = []
for cid in sample:
    sb, sf, sn = score(cid, A["B"][cid]), score(cid, A["C_fill"][cid]), score(cid, A["C_nofill"][cid])
    rows.append(dict(case=cid, type=typ[cid], B=sb, C_fill=sf, C_nofill=sn))
df = pd.DataFrame(rows); df["d_fill"] = df.C_fill - df.B; df["d_nofill"] = df.C_nofill - df.B
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(df.round(2).to_string(index=False))
df.to_csv("data/sigma_prediction/predict.csv", index=False)
for col in ("d_fill", "d_nofill"):
    print(col, "win", (df[col] > 1e-9).sum(), "tie", (df[col].abs() <= 1e-9).sum(), "loss", (df[col] < -1e-9).sum(), "mean", round(df[col].mean(), 4))
    print(df.groupby("type")[col].agg(win=lambda s: (s > 1e-9).sum(), tie=lambda s: (s.abs() <= 1e-9).sum(), loss=lambda s: (s < -1e-9).sum(), n="size", mean="mean").round(3).to_string())
print("B mean", df.B.mean().round(3), "C_fill mean", df.C_fill.mean().round(3), "C_nofill mean", df.C_nofill.mean().round(3))
