"""Simulate whether a sigma run on the predicted per-case C and B scores would be judged valid by EJ.4(4) (GN.6 method, 附录 GX). Needs data/sigma_prediction/predict.csv from predict_c_minus_b.py. No API call.

    PYTHONPATH=. python scripts/simulate_sigma_validity.py
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.eval.noise import decompose_paired_variance, sigma_separates_n, required_n
S = ""
df = pd.read_csv("data/sigma_prediction/predict.csv")
# real B r1 vs ideal B
rows = [json.loads(l) for l in open("data/sigma_v1/runs.jsonl")]
real = {}
for r in rows:
    fa = [s for s in r["eval_result"]["scores"] if s["metric"] == "factual_accuracy"][0]
    if fa.get("valid", True): real[r["case_id"]] = fa["score"]
m = df[df.case.isin(real)]
agree = sum(abs(real[c] - b) < 0.01 for c, b in zip(m.case, m.B))
print("real B r1 valid cases", len(m), "equal to ideal B:", agree)
print([ (c, round(real[c],2), round(b,2)) for c,b in zip(m.case,m.B) if abs(real[c]-b)>=0.01])
rng = np.random.default_rng(1)
def sim(delta_col, sd, k=8, runs=100, n_cases=40):
    b = df.B.values; c = df.C_fill.values if delta_col == "fill" else df.C_nofill.values
    inval = 0; sig = []; ns = []
    for _ in range(runs):
        A = {}; Bm = {}
        for i, cid in enumerate(df.case):
            A[cid] = list(np.clip(c[i] + rng.normal(0, sd, k), 0, 1))
            Bm[cid] = list(np.clip(b[i] + rng.normal(0, sd, k), 0, 1))
        d = decompose_paired_variance(A, Bm, n_resamples=1000, seed=int(rng.integers(1e6)))
        inval += not sigma_separates_n(d, 5)
        sig.append(d.sigma_hetero); ns.append(required_n(d.var_at(5)))
    return inval / runs, float(np.mean(sig)), float(np.median(ns)), float(np.percentile(ns, 5)), float(np.percentile(ns, 95))
print("true sd of predicted d: fill", round(np.std(df.C_fill - df.B, ddof=1), 3), "nofill", round(np.std(df.C_nofill - df.B, ddof=1), 3))
for col in ("fill", "nofill"):
    for sd in (0.0, 0.08, 0.15):
        p, sg, nmed, n5, n95 = sim(col, sd)
        print(col, "noise sd", sd, "P(invalid)", round(p, 3), "mean sigma_hetero", round(sg, 3), "N median", nmed, "5-95%", n5, n95)
