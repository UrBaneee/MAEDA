#!/usr/bin/env python3
"""
Offline analysis of the reduced σ run (附录 GY.4, GZ, 预注册 §10.13). Reads the run files, makes no
API call.

Void rule (objective, applied to every saved row): the row's own `void`, OR the
rule-based fallback report header, OR elapsed_s above the run timeout (default 900 s).
Void runs are excluded. Guardrail refusals are NOT void: they score 0 (§7.2.1, §10.13).

    PYTHONPATH=. python scripts/analyze_sigma.py [--keep-timeout-rows]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from src.eval.noise import decompose_paired_variance, required_n, sigma_separates_n

SIG = "Analysis completed for query: *"
ARMS = ["B", "C", "R", "CR"]


def load(timeout: float, apply_timeout: bool = True) -> list[dict]:
    rows = [json.loads(l) for f in sorted(glob.glob("data/sigma_v1*/runs.jsonl")) for l in open(f)]
    for r in rows:
        why = r.get("void")
        if not why and SIG in (r.get("report") or ""):
            why = "llm_report_fallback"
        if not why and apply_timeout and (r["meta"].get("elapsed_s") or 0) > timeout:
            why = "timeout"
        r["eff_void"] = why
        fa = [s for s in r["eval_result"]["scores"] if s["metric"] == "factual_accuracy"]
        refused = r["meta"].get("terminal_state") == "safe_refusal"
        r["refused"] = refused
        r["fa"] = 0.0 if refused else (fa[0]["score"] if fa and fa[0].get("valid", True) else None)
    return rows


def scores_by_case(rows, arm):
    out = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and not r["eff_void"] and r["fa"] is not None:
            out[r["case_id"]].append(r["fa"])
    return out


def power_n(delta: float, sd: float, alpha: float, power: float = 0.8, two_sided: bool = False) -> int:
    for n in range(3, 100000):
        df = n - 1
        crit = stats.t.ppf(1 - (alpha / 2 if two_sided else alpha), df)
        nc = delta / (sd / math.sqrt(n))
        pw = 1 - stats.nct.cdf(crit, df, nc)
        if two_sided:
            pw += stats.nct.cdf(-crit, df, nc)
        if pw >= power:
            return n
    return 100000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--keep-timeout-rows", action="store_true", help="sensitivity: do not void long-elapsed rows")
    a = ap.parse_args()
    rows = load(a.timeout, apply_timeout=not a.keep_timeout_rows)
    print("== completions per arm and repeat (n total / success / refused / void)")
    for arm in ARMS:
        for rep in (1, 2, 3):
            sub = [r for r in rows if r["arm"] == arm and r["repeat"] == rep]
            print(f"{arm} r{rep}: {len(sub)} saved, success {sum(1 for r in sub if not r['eff_void'] and not r['refused'])}, "
                  f"refused {sum(1 for r in sub if not r['eff_void'] and r['refused'])}, "
                  f"void {sum(1 for r in sub if r['eff_void'])} {sorted({r['eff_void'] for r in sub if r['eff_void']})}")
    S = {arm: scores_by_case(rows, arm) for arm in ARMS}
    cases = sorted(set(S["B"]) & set(S["C"]))
    d = np.array([np.mean(S["C"][c]) - np.mean(S["B"][c]) for c in cases])
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    lb80 = mean - stats.t.ppf(0.8, n - 1) * se
    ci = (mean - stats.t.ppf(0.975, n - 1) * se, mean + stats.t.ppf(0.975, n - 1) * se)
    print(f"\n== H1: cases with both arms {n}; mean C-B {mean:.4f}; case SD {sd:.4f}; SE {se:.4f}")
    print(f"   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]; one-sided 80% lower bound {lb80:.4f}")
    print(f"   arm means: " + ", ".join(f"{k} {np.mean([np.mean(v) for v in S[k].values()]):.3f}" for k in ARMS))
    wins = int((d > 1e-9).sum()); ties = int((abs(d) <= 1e-9).sum()); loss = int((d < -1e-9).sum())
    print(f"   C wins {wins}, ties {ties}, losses {loss}")
    dec = decompose_paired_variance(S["C"], S["B"])
    print(f"   repeat noise (mean per-trial var s2_a+s2_b) {dec.repeat_var:.4f}; sigma_hetero {dec.sigma_hetero:.4f} "
          f"CI [{dec.sigma_hetero_ci[0]:.4f}, {dec.sigma_hetero_ci[1]:.4f}]; mean k {dec.mean_k:.2f}")
    for k in (3, 5):
        print(f"   sigma_separates_n(k={k}) = {sigma_separates_n(dec, k)}; required_n at mde 0.028 = {required_n(dec.var_at(k))}")
    print("\n== N by the §10.13 rule")
    if lb80 <= 0.028:
        print(f"   lower bound {lb80:.4f} is not above 0.028: STOP, hand to the user")
    else:
        N = power_n(lb80, sd, 0.0167)
        print(f"   power target (lower bound) {lb80:.4f}, SD {sd:.4f}, one-sided alpha 0.0167, power 80% -> N = {N}"
              + ("  (> 100: STOP, hand to the user)" if N > 100 else ""))
        Nm = power_n(mean, sd, 0.0167)
        print(f"   for reference, N against the point estimate {mean:.4f}: {Nm}")
    N = locals().get("N")
    # H3
    cs3 = sorted(set(S["B"]) & set(S["C"]) & set(S["R"]) & set(S["CR"]))
    if cs3:
        d3 = np.array([(np.mean(S["CR"][c]) - np.mean(S["R"][c])) - (np.mean(S["C"][c]) - np.mean(S["B"][c])) for c in cs3])
        sd3 = d3.std(ddof=1)
        print(f"\n== H3: cases with all four arms {len(cs3)}; mean (CR-R)-(C-B) {d3.mean():.4f}; case SD {sd3:.4f}")
        for nn in ([N] if N and N <= 100 else []) + [len(cs3)]:
            lo, hi = 0.001, 2.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if power_n(mid, sd3, 0.0167, two_sided=True) <= nn: hi = mid
                else: lo = mid
            print(f"   H3 minimum detectable effect at N={nn} (two-sided alpha 0.0167, power 80%): {hi:.3f}")
    # per-case table
    print("\n== per case: C-B (mean over repeats), B, C")
    for c, x in zip(cases, d):
        print(f"{c:5s} {x:+.2f}  B {np.mean(S['B'][c]):.2f} C {np.mean(S['C'][c]):.2f}  nB {len(S['B'][c])} nC {len(S['C'][c])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
