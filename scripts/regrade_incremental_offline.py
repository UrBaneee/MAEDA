"""Pre-registration §8.6.3 first row: offline incremental re-score of stored
Sol groundedness output (step 0 -> 1 -> 2). No API call, touches no holdout.

Unlike scripts/regrade_groundedness_offline.py (which approximates claim type
from wording), section attribution here is computed against the report text the
judge actually saw, recovered from the blind-relabel page.

Limits, by construction (§8.5): `section` and the failure-report tag are
APPROXIMATED, because the stored output has neither. Only the reasoning of the
median sample is stored, so all re-scoring is over that one sample.

    python scripts/regrade_incremental_offline.py classify   # no human labels read
    python scripts/regrade_incremental_offline.py score      # joins human reference
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.agreement import discretize  # noqa: E402

ANN = Path(__file__).resolve().parents[1] / "data" / "annotations"
SOL = ANN / "answer_key_openai_gpt_5_6_sol_groundedness_xhigh_16k_nonpredictive20_complete.json"
HTML = ANN / "dmxapi_calibration20_nonpredictive_blind_relabel.html"
MAP = ANN / "dmxapi_calibration20_nonpredictive_blind_mapping.json"
REFS = {
    "strict": ANN / "human_reference_nonpredictive20_strict_lower.json",
    "lenient": ANN / "human_reference_nonpredictive20_lenient_upper.json",
}
READ = {"Executive Summary", "Key Findings", "Analysis Details"}  # §9 R1
TIE_MARGIN = 0.15  # a claim within this containment of a read section stays read
# §9 R5 says chart claims enter neither score, but §8.6.2 has no step for it. This
# diagnostic was added AFTER steps 0-2 had been seen; it is not a candidate rule
# until adjudicated, and it counts as a selection degree of freedom (§8.5).
CHART = re.compile(r"\b(chart|plot|graph|visual\w*|histogram|heatmap)\b", re.I)
N_BOOT = 100_000  # FE.8
BAR = 0.7167  # §8.1

STOP = set("a an the of to in on for and or by with was were is are be been that this "
           "it its as at from had has have not no their there which than more".split())

# Rule frozen before any human label is read (附录 FF). Step 2 exempts a claim
# only when it asserts that a step/tool FAILED, or asserts inability AND the report
# sentence it was drawn from attributes that to a failure. A bare "could not be
# determined" stays a deduction (R2.2: C16, D05).
FAILURE = re.compile(
    r"\b(step|steps|analysis|analyses|tool|computation|calculation|query)\b[^.]*"
    r"\b(fail|failed|fails|failure|failures|error|errors|errored)\b"
    r"|\b(fail|failed|failure|failures|error|errors)\b[^.]*"
    r"\b(step|steps|analysis|analyses|tool|computation|calculation|query)\b",
    re.I,
)

INABILITY = re.compile(r"could not|cannot|unable|inability|not (be )?calculated", re.I)


def tokens(text: str) -> set[str]:
    out = set()
    for t in re.findall(r"\d[\d,]*\.?\d*|[a-zA-Z][a-zA-Z-]+", text.lower()):
        t = t.replace(",", "").rstrip(".")
        if t in STOP:
            continue
        out.add(t[:6] if t[0].isalpha() else t)  # crude stem
    return out


def split_sections(report: str) -> list[tuple[str, str]]:
    """(top-level section, sentence) pairs; ### subsections belong to their parent."""
    out, cur = [], None
    for line in report.splitlines():
        m = re.match(r"^(#{1,2}) (.+)$", line)
        if m:
            cur = m.group(2).strip()
            continue
        if cur and line.strip():
            for s in re.split(r"(?<=[.!?])\s+", line):
                if s.strip():
                    out.append((cur, s))
    return out


def parse_claims(reasoning: str) -> list[tuple[bool, str, str]]:
    body = reasoning[re.match(r"\d+/\d+ claims supported\.?\s*", reasoning).end():]
    body = re.sub(r"\s*\[[^\]]*spread=[^\]]*\]\s*$", "", body)
    claims = []
    for p in body.split(" | "):
        p = p.strip()
        m = re.match(r"^([✓✗])\s*(.*?)(?:\s*\((.*)\))?$", p, re.S)
        if not m:
            raise ValueError(f"unparsable claim: {p[:80]}")
        claims.append((m.group(1) == "✓", m.group(2), m.group(3) or ""))
    return claims


def attribute(claim: str, sents: list[tuple[str, str]]) -> dict:
    c = tokens(claim)
    best: dict[str, float] = {}
    src, src_score = "", -1.0
    for sec, s in sents:
        score = len(c & tokens(s)) / max(len(c), 1)
        best[sec] = max(best.get(sec, 0.0), score)
        if score > src_score:
            src, src_score = s, score
    br = max((v for k, v in best.items() if k in READ), default=0.0)
    bu = max((v for k, v in best.items() if k not in READ), default=0.0)
    top = max(best, key=best.get)
    return {"top": top, "best_read": br, "best_unread": bu, "read": bu - br <= TIE_MARGIN,
            "read_strict": bu <= br, "source": src}


def load_cases() -> dict:
    h = HTML.read_text()
    samples, _ = json.JSONDecoder().raw_decode(h[h.find("const SAMPLES = ") + 16:])
    mapping = json.loads(MAP.read_text())["mapping"]
    sol = json.loads(SOL.read_text())["scores"]
    cases = {}
    for s in samples:
        cid = mapping[s["case_id"]]
        g = sol[cid]["groundedness"]
        claims = parse_claims(g["reasoning"])
        expected = re.match(r"(\d+)/(\d+)", g["reasoning"])
        assert len(claims) == int(expected[2]) and sum(c[0] for c in claims) == int(expected[1]), cid
        assert abs(g["score"] - sum(c[0] for c in claims) / len(claims)) < 1e-9, cid
        sents = split_sections(s["report"])
        cases[cid] = {
            "score": g["score"],
            "claims": [
                {"ok": ok, "claim": t, "ev": ev, **attribute(t, sents),
                 "failure": False}
                for ok, t, ev in claims
            ],
        }
        for c in cases[cid]["claims"]:
            c["failure"] = (not c["ok"]) and bool(
                FAILURE.search(c["claim"])
                or (INABILITY.search(c["claim"]) and FAILURE.search(c["source"])))
    return cases


def ratio(cl: list[dict], default: float = 1.0) -> float:
    return sum(c["ok"] for c in cl) / len(cl) if cl else default


def variants(cases: dict) -> dict[str, dict[str, float]]:
    return {
        "0 baseline": {c: ratio(v["claims"]) for c, v in cases.items()},
        "1 read sections": {c: ratio([x for x in v["claims"] if x["read"]]) for c, v in cases.items()},
        "2 +failure exempt": {
            c: ratio([x for x in v["claims"] if x["read"] and not x["failure"]]) for c, v in cases.items()},
        "1s strict-argmax": {c: ratio([x for x in v["claims"] if x["read_strict"]]) for c, v in cases.items()},
        "2s strict+failure": {
            c: ratio([x for x in v["claims"] if x["read_strict"] and not x["failure"]]) for c, v in cases.items()},
    }


def qwk(h, j) -> float:
    return cohen_kappa_score(h, j, weights="quadratic", labels=[0, 1, 2, 3, 4])


def boot(h: np.ndarray, j: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = len(idx)
    conf = np.zeros((n, 5, 5))
    np.add.at(conf, (np.arange(n)[:, None], h[idx], j[idx]), 1)
    w = (np.arange(5)[:, None] - np.arange(5)) ** 2
    exp = conf.sum(2, keepdims=True) * conf.sum(1, keepdims=True) / h.size
    den = (w * exp).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - (w * conf).sum((1, 2)) / den  # NaN = degenerate resample


def main() -> int:
    cases = load_cases()
    if sys.argv[1:] == ["classify"]:
        for cid, v in cases.items():
            for c in v["claims"]:
                flag = ("U" if not c["read"] else "R") + ("F" if c["failure"] else " ")
                if not c["ok"] or not c["read"]:
                    print(f"{cid:5}{flag} {'✓' if c['ok'] else '✗'} {c['best_read']:.2f}/{c['best_unread']:.2f} "
                          f"{c['top'][:14]:14} {c['claim'][:95]}")
        return 0
    ids = sorted(cases)
    # Same resamples for every step, so step-to-step deltas are paired.
    idx = np.random.default_rng(20261031).integers(0, len(ids), (N_BOOT, len(ids)))
    out: dict = {"n_exempt": {c: sum(x["failure"] for x in v["claims"]) for c, v in cases.items() if
                              any(x["failure"] for x in v["claims"])},
                 "claims_kept": {}, "steps": {}}
    vs = variants(cases)
    for rn, path in REFS.items():
        raw = json.loads(path.read_text())["labels"]
        h = np.array([discretize(raw[c]["groundedness"]) for c in ids])
        hv = np.array([raw[c]["groundedness"] for c in ids])
        print(f"--- {rn}")
        prev = None
        for name, sc in vs.items():
            j = np.array([discretize(sc[c]) for c in ids])
            jv = np.array([sc[c] for c in ids])
            vals = boot(h, j, idx)
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            row = {"qwk": round(qwk(h, j), 4), "ci95": [round(lo, 3), round(hi, 3)],
                   "mae": round(float(abs(hv - jv).mean()), 4), "exact": float((h == j).mean()),
                   "bias": round(float((jv - hv).mean()), 4),
                   "p_ge_bar": round(float(np.nanmean(vals >= BAR)), 3)}
            line = (f"{name:20} QWK {row['qwk']:.3f} {row['ci95']} MAE {row['mae']:.3f} "
                    f"exact {row['exact']:.2f} bias {row['bias']:+.3f} P(>=bar) {row['p_ge_bar']:.2f}")
            if prev is not None and name[0] in "12" and not name.startswith("1s"):
                d = vals - prev
                dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
                row["delta_ci95"] = [round(dlo, 3), round(dhi, 3)]
                line += f"  delta CI [{dlo:+.3f},{dhi:+.3f}]"
            if not name.endswith("strict-argmax") and name[1] == " ":
                prev = vals
            out["steps"].setdefault(name, {"scores": {c: round(sc[c], 4) for c in ids}})[rn] = row
            print(line)
    diag = {c: ratio([x for x in cases[c]["claims"] if x["read"] and not x["failure"]
                      and not CHART.search(x["claim"])]) for c in ids}
    print("--- post-hoc diagnostic: R5 chart claims removed on top of step 2")
    n_chart = [x for v in cases.values() for x in v["claims"] if CHART.search(x["claim"])]
    print(f"chart claims {len(n_chart)}, unsupported {sum(not x['ok'] for x in n_chart)}")
    out["r5_diagnostic"] = {"scores": {c: round(diag[c], 4) for c in ids}}
    for rn, path in REFS.items():
        raw = json.loads(path.read_text())["labels"]
        h = np.array([discretize(raw[c]["groundedness"]) for c in ids])
        j = np.array([discretize(diag[c]) for c in ids])
        step2 = np.array([discretize(vs["2 +failure exempt"][c]) for c in ids])
        vals, base = boot(h, j, idx), boot(h, step2, idx)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        dlo, dhi = np.nanpercentile(vals - base, [2.5, 97.5])
        loo = [qwk(np.delete(h, i), np.delete(j, i)) for i in range(len(ids))]
        out["r5_diagnostic"][rn] = {
            "qwk": round(qwk(h, j), 4), "ci95": [round(lo, 3), round(hi, 3)],
            "delta_vs_step2_ci95": [round(dlo, 3), round(dhi, 3)],
            "p_ge_bar": round(float(np.nanmean(vals >= BAR)), 3),
            "leave_one_out": [round(min(loo), 3), round(max(loo), 3)]}
        print(rn, out["r5_diagnostic"][rn])
    pass1 = json.loads((ANN / "dmxapi_calibration20_nonpredictive_manifest.json").read_text())["labels"]
    pass2 = json.loads((ANN / "human_labels_dmxapi_calibration20_nonpredictive_relabel_decoded.json").read_text())["labels"]
    a = np.array([discretize(pass1[c]["groundedness"]) for c in ids])
    b = np.array([discretize(pass2[c]["groundedness"]) for c in ids])
    lo, hi = np.nanpercentile(boot(a, b, idx), [2.5, 97.5])
    out["human_ceiling"] = {"qwk": round(qwk(a, b), 4), "ci95": [round(lo, 3), round(hi, 3)]}
    print("human ceiling", out["human_ceiling"])
    print("exempted claims per case:", out["n_exempt"])
    for name in ("read", "read_strict"):
        print(name, "claims kept:", sum(x[name] for v in cases.values() for x in v["claims"]), "of",
              sum(len(v["claims"]) for v in cases.values()))
    print(f"bar (§8.1): {BAR}")
    (ANN / "regrade_incremental_offline_result.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
