"""
Exploratory trial (branch experiment/gemini-decision): can Gemini Decision 1
classify query_type as well as MAEDA's intent parser?

Why: in phase 4 the retrieval gate (auto mode retrieves only when the parsed
query_type is diagnostic or exploratory) had false positives that came from
query_type errors. A closed-set classifier is what Decision 1 is built for.

Gold: `query_type` in tests/eval/test_suite.json (130 dev cases).
Decision 1 sees the query only and the five labels with no definitions
(DECISION_ROUTER_QUESTIONS), the same information the intent parser prompt
gives. Pre-specified primary: top choice vs gold (accuracy) and the gate
decision (retrieve iff diagnostic or exploratory) vs the gate on gold.
Parser baseline, no new parser calls: (a) the recorded intent_accuracy
reasoning in the σ runs (data/sigma_*, majority over repeats, 40 cases);
(b) the replay cache's parsed_intent (100 cases, older pipeline).

  python scripts/decision_router_trial.py                 # dry run, no API call
  MAEDA_ALLOW_PAID_TESTS=1 python scripts/decision_router_trial.py --execute
  python scripts/decision_router_trial.py --analyze data/gemini_decision/router_dev130.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.agent_prompts import DECISION_ROUTER_QUESTIONS as QUESTIONS  # noqa: E402

MODEL = "gemini-decision-1"
PRICE_PER_M_INPUT = 0.10
SUITE = Path("tests/eval/test_suite.json")
OUT = Path("data/gemini_decision/router_dev130.json")
ALLOW_ENV = "MAEDA_ALLOW_PAID_TESTS"
TYPES = list(QUESTIONS["query_type"]["criteria"])
GATE = {"diagnostic", "exploratory"}
CHARS_PER_TOKEN = 3.5
REQUEST_OVERHEAD_TOKENS = 110
_MISMATCH = re.compile(r"query_type mismatch: got (\w+) expected (\w+)")
_CORRECT = re.compile(r"query_type correct: (\w+)")


def gold() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(SUITE.read_text()) if c.get("split") == "dev"}


def build_request(query: str) -> dict:
    return {"model": MODEL, "input": f"User question: {query}", "questions": QUESTIONS}


def estimate_tokens(req: dict) -> int:
    return math.ceil((len(req["input"]) + len(json.dumps(req["questions"]))) / CHARS_PER_TOKEN) + REQUEST_OVERHEAD_TOKENS


def parse_choice(ans: dict) -> dict:
    probs = {k: float(v) for k, v in ans["probabilities"].items()}
    return {"choice": ans["choice"], "probs": probs, "confidence": float(ans.get("confidence", max(probs.values())))}


def parser_from_reasoning(text: str) -> str | None:
    m = _MISMATCH.search(text) or _CORRECT.search(text)
    return m.group(1) if m else None


def parser_baselines() -> dict[str, dict[str, str]]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for f in sorted(glob.glob("data/sigma_*/runs.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for s in (r.get("eval_result") or {}).get("scores", []):
                if s.get("metric") == "intent_accuracy":
                    p = parser_from_reasoning(s.get("reasoning") or "")
                    if p:
                        votes[r["case_id"]][p] += 1
    sigma = {c: v.most_common(1)[0][0] for c, v in votes.items()}
    cache = {}
    try:
        from src.eval.replay_cache import ReplayCache
        for cid, e in ReplayCache()._entries.items():
            qt = (e.state.get("parsed_intent") or {}).get("query_type")
            if qt:
                cache[cid] = qt
    except Exception:  # cache is local and optional
        pass
    return {"parser_sigma": sigma, "parser_cache": cache}


def score(pred: dict[str, str], g: dict[str, dict]) -> dict:
    ids = sorted(c for c in pred if c in g)
    if not ids:
        return {"n": 0}
    acc = sum(pred[c] == g[c]["query_type"] for c in ids) / len(ids)
    gate_p = {c: pred[c] in GATE for c in ids}
    gate_g = {c: g[c]["query_type"] in GATE for c in ids}
    fp = sum(gate_p[c] and not gate_g[c] for c in ids)
    fn = sum(gate_g[c] and not gate_p[c] for c in ids)
    conf = Counter((g[c]["query_type"], pred[c]) for c in ids)
    return {"n": len(ids), "accuracy": acc, "gate_accuracy": 1 - (fp + fn) / len(ids),
            "gate_fp": fp, "gate_fn": fn,
            "confusion": {f"{a}->{b}": k for (a, b), k in sorted(conf.items()) if a != b}}


def analyze(results: dict, g: dict | None = None, baselines: dict | None = None) -> dict:
    g = g or gold()
    baselines = baselines if baselines is not None else parser_baselines()
    dec = {c: v["choice"] for c, v in results.items()}
    out = {"decision_all": score(dec, g)}
    for name, b in baselines.items():
        common = {c for c in b if c in dec}
        out[name] = score({c: b[c] for c in common}, g)
        out[f"decision_on_{name}_cases"] = score({c: dec[c] for c in common}, g)
    return out


def print_analysis(an: dict) -> None:
    for k, r in an.items():
        if not r.get("n"):
            print(f"  {k:30s} n 0")
            continue
        print(f"  {k:30s} n {r['n']:3d}  accuracy {r['accuracy']:.3f}  gate acc {r['gate_accuracy']:.3f}"
              f"  (FP {r['gate_fp']}, FN {r['gate_fn']})  errors {r['confusion']}")


def dry_run() -> dict:
    g = gold()
    reqs = {c: build_request(v["query"]) for c, v in g.items()}
    toks = sum(estimate_tokens(r) for r in reqs.values())
    print(f"DRY RUN, no API call. {len(reqs)} requests, ~{toks:,} input tokens, "
          f"${toks / 1e6 * PRICE_PER_M_INPUT:.4f} at paid price (free tier $0)")
    print("  example:", json.dumps(next(iter(reqs.values())))[:300])
    return {"n": len(reqs), "tokens": toks}


def execute(out_path: Path = OUT, client=None, sleep=time.sleep) -> dict:
    if os.environ.get(ALLOW_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit(f"Refusing API calls: set {ALLOW_ENV}=1 for this process")
    if client is None:
        from dotenv import dotenv_values
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or dotenv_values(".env").get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Refusing API calls: GEMINI_API_KEY missing")
        client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    done = json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    for cid, case in gold().items():
        if cid in done:
            continue
        req = build_request(case["query"])
        for attempt in range(4):
            try:
                resp = client.decisions.create(**req)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"  {cid}: {type(exc).__name__}, retry in {15 * (attempt + 1)}s")
                sleep(15 * (attempt + 1))
        d = resp.model_dump() if hasattr(resp, "model_dump") else resp
        done[cid] = {**parse_choice(d["answers"]["query_type"]),
                     "input_tokens": int(d["usage"]["total_input_tokens"])}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": {"model": MODEL, "primary": "choice"}, "results": done}, indent=1))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--analyze", default=None)
    a = ap.parse_args(argv)
    if a.analyze:
        print_analysis(analyze(json.loads(Path(a.analyze).read_text())["results"]))
        return 0
    if not a.execute:
        dry_run()
        return 0
    res = execute()
    print(f"{len(res)} cases, {sum(v['input_tokens'] for v in res.values())} input tokens")
    print_analysis(analyze(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
